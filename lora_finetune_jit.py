import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from denoiser import Denoiser
from engine_jit import train_one_epoch, evaluate
from util.lora_utils import (
    inject_lora,
    mark_only_lora_as_trainable,
    count_trainable_params,
    resolve_checkpoint_path,
    _is_lora_state_dict,
    add_lora_args,
    load_state_dict_with_font_embedding_resize,
)
from main_jit import FontSrcTargetRefsDataset, collate_src_target_refs, get_args_parser
from util.crop import resize_and_random_crop
from util.misc import save_model_no_ema
import util.misc as misc


def main(args):
    misc.init_distributed_mode(args)
    print("Job directory:", os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for float32 matmul and cuDNN.")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: resize_and_random_crop(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = FontSrcTargetRefsDataset(
        root=args.data_path,
        transform=transform_train,
        ref_size=128,
        max_chars_per_font=args.max_chars_per_font
    )
    print(f"Dataset: {len(dataset_train)} samples, {dataset_train.num_fonts} fonts")

    if dataset_train.num_fonts != args.num_fonts:
        print(f"Warning: Different num_fonts from args {args.num_fonts} to dataset {dataset_train.num_fonts}")
        assert args.num_fonts >= dataset_train.num_fonts
    if dataset_train.num_chars != args.num_chars:
        print(f"Warning: Different num_chars from args {args.num_chars} to dataset {dataset_train.num_chars}")
        assert args.num_chars >= dataset_train.num_chars

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_src_target_refs
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    model = Denoiser(args)
    model.update_ema = lambda: None

    base_ckpt_path = resolve_checkpoint_path(args.base_checkpoint) if args.base_checkpoint else None
    if args.resume and args.base_checkpoint:
        print("Both --resume and --base_checkpoint provided; ignoring --base_checkpoint.")
        base_ckpt_path = None

    base_state_dict = None
    base_is_lora = False
    if base_ckpt_path:
        if not os.path.exists(base_ckpt_path):
            raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}")
        checkpoint = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        base_state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        base_is_lora = _is_lora_state_dict(base_state_dict)
        del checkpoint

    if base_state_dict is not None and not base_is_lora:
        load_messages = load_state_dict_with_font_embedding_resize(model, base_state_dict, strict=True)
        for message in load_messages:
            print(message)
        print("Loaded vanilla base checkpoint from", base_ckpt_path)

    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    replaced = inject_lora(model.net, targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    print(f"LoRA injected into {replaced} Linear modules (targets={targets}).")

    if base_state_dict is not None and base_is_lora:
        load_messages = load_state_dict_with_font_embedding_resize(model, base_state_dict, strict=True)
        for message in load_messages:
            print(message)
        print("Loaded LoRA base checkpoint from", base_ckpt_path)

    mark_only_lora_as_trainable(
        model,
        train_font_emb=True,
        train_content_encoder=args.train_content_encoder,
        train_style_encoder=args.train_style_encoder,
    )

    n_trainable = count_trainable_params(model)
    trainable_parts = ["LoRA"]
    if args.train_content_encoder:
        trainable_parts.append("content encoder")
    if args.train_style_encoder:
        trainable_parts.append("style encoder")
    trainable_desc = " + ".join(trainable_parts)
    print("Trainable parameters ({}): {:.6f}M".format(trainable_desc, n_trainable / 1e6))

    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    checkpoint_path = resolve_checkpoint_path(args.resume) if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_messages = load_state_dict_with_font_embedding_resize(
            model_without_ddp,
            checkpoint["model"],
            strict=True,
        )
        for message in load_messages:
            print(message)

        model_without_ddp.ema_params1 = list(model_without_ddp.parameters())
        model_without_ddp.ema_params2 = list(model_without_ddp.parameters())

        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer state from", checkpoint_path)

        print("Resumed LoRA checkpoint from", checkpoint_path)
        del checkpoint
    else:
        model_without_ddp.ema_params1 = list(model_without_ddp.parameters())
        model_without_ddp.ema_params2 = list(model_without_ddp.parameters())
        if args.resume:
            print("Warning: resume path not found, training from scratch.")
        else:
            print("Training from base checkpoint ({}).".format(trainable_desc))

    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    print(f"Start LoRA training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer,
                        args=args)

        if epoch > 0 and (epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs):
            save_model_no_ema(
                args=args,
                model_without_ddp=model_without_ddp,
                epoch=epoch,
                epoch_name="last"
            )

        if args.online_eval and epoch > 0 and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time:", total_time_str)


if __name__ == "__main__":
    parser = get_args_parser()
    parser = add_lora_args(parser)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
