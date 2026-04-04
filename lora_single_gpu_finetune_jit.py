import argparse
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
from engine_jit import train_one_epoch_single_gpu, evaluate_single_gpu
from util.lora_utils import (
    inject_lora,
    mark_only_lora_as_trainable,
    count_trainable_params,
    resolve_checkpoint_path,
    _is_lora_state_dict,
    add_lora_args,
    load_state_dict_with_font_embedding_resize,
)
from main_jit import FontSrcTargetRefsDataset, collate_src_target_refs
from util.crop import resize_and_random_crop
from util.misc import save_model_no_ema, save_named_model_no_ema
import util.misc as misc
from util.unicode_labels import load_unicode_codepoints
from util.ids_utils import build_ids_resources


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('LoRA Fine-Tuning (Single GPU)', add_help=True)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling / evaluation
    parser.add_argument('--sampling_method', default='heun', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--cfg', default=1.0, type=float)
    parser.add_argument('--interval_min', default=0.0, type=float)
    parser.add_argument('--interval_max', default=1.0, type=float)
    parser.add_argument('--num_images', default=50000, type=int)
    parser.add_argument('--test_npz_path', default='test_set.npz', type=str)
    parser.add_argument('--eval_freq', type=int, default=40)
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--eval_step_folders', action='store_true',
                        help='Save evaluation outputs in per-step subfolders (step_N/)')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256)

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str)
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--num_fonts', default=100, type=int)
    parser.add_argument('--num_chars', default=500, type=int)
    parser.add_argument('--max_chars_per_font', default=None, type=int)
    parser.add_argument('--num_style_refs', default=1, type=int)
    parser.add_argument('--style_ref_mode', default='single', type=str, choices=['single', 'mean', 'max'])
    parser.add_argument('--use_unicode_char_labels', action='store_true')
    parser.add_argument('--unicode_codepoints_path', default='', type=str)
    parser.add_argument('--use_char_embedding', action='store_true')
    parser.add_argument('--use_ids_conditioning', action='store_true')
    parser.add_argument('--ids_path', default='', type=str)
    parser.add_argument('--binary_loss_weight', default=0.0, type=float)
    parser.add_argument('--edge_loss_weight', default=0.0, type=float)
    parser.add_argument('--projection_loss_weight', default=0.0, type=float)
    parser.add_argument('--char_loss_weight', default=0.0, type=float)
    parser.add_argument('--ids_loss_weight', default=0.0, type=float)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--resume', default='')
    parser.add_argument('--infinite', action='store_true',
                        help='Train indefinitely until manually stopped.')
    parser.add_argument('--save_last_freq', type=int, default=5)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda')

    return add_lora_args(parser)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("Job directory:", os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for float32 matmul and cuDNN.")

    if args.output_dir is not None:
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
        max_chars_per_font=args.max_chars_per_font,
        num_style_refs=args.num_style_refs,
        use_unicode_char_labels=args.use_unicode_char_labels,
        unicode_codepoints=load_unicode_codepoints(args.unicode_codepoints_path) if args.unicode_codepoints_path else None,
    )
    print(f"Dataset: {len(dataset_train)} samples, {dataset_train.num_fonts} fonts")

    if args.use_unicode_char_labels and not getattr(args, "unicode_codepoints", None):
        args.unicode_codepoints = list(dataset_train.unicode_codepoints)
        args.num_chars = dataset_train.num_chars

    if dataset_train.num_fonts != args.num_fonts:
        print(f"Warning: Different num_fonts from args {args.num_fonts} to dataset {dataset_train.num_fonts}")
        assert args.num_fonts >= dataset_train.num_fonts
    if dataset_train.num_chars != args.num_chars:
        print(f"Warning: Different num_chars from args {args.num_chars} to dataset {dataset_train.num_chars}")
        assert args.num_chars >= dataset_train.num_chars

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_src_target_refs
    )

    torch._dynamo.config.cache_size_limit = 128

    if args.use_ids_conditioning:
        if not args.use_unicode_char_labels:
            raise ValueError("--use_ids_conditioning requires --use_unicode_char_labels.")
        if not args.ids_path:
            raise ValueError("--use_ids_conditioning requires --ids_path.")
        ids_resources = build_ids_resources(args.unicode_codepoints, args.ids_path)
        args.ids_vocab_size = ids_resources["ids_vocab_size"]
        args.ids_max_len = ids_resources["ids_max_len"]
    else:
        ids_resources = None
        args.ids_vocab_size = 0
        args.ids_max_len = 0

    model = Denoiser(args)
    if ids_resources is not None:
        model.net.y_embedder.set_ids_lookup(ids_resources["ids_token_ids"], ids_resources["ids_token_mask"])
        if ids_resources["missing_codepoints"]:
            print(
                f"Warning: IDS file does not cover {len(ids_resources['missing_codepoints'])} training codepoints. "
                "Those characters will receive a zero IDS embedding."
            )
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
        model.refresh_semantic_consistency_encoder_from_content()

    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    replaced = inject_lora(model.net, targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    print(f"LoRA injected into {replaced} Linear modules (targets={targets}).")

    if base_state_dict is not None and base_is_lora:
        load_messages = load_state_dict_with_font_embedding_resize(model, base_state_dict, strict=True)
        for message in load_messages:
            print(message)
        print("Loaded LoRA base checkpoint from", base_ckpt_path)
        model.refresh_semantic_consistency_encoder_from_content()

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

    if args.lr is None:
        args.lr = args.blr

    print("Learning rate: {:.2e}".format(args.lr))
    print("Batch size: %d" % args.batch_size)

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    checkpoint_path = resolve_checkpoint_path(args.resume) if args.resume else None
    best_fid = float("inf")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_messages = load_state_dict_with_font_embedding_resize(model, checkpoint["model"], strict=True)
        for message in load_messages:
            print(message)
        model.refresh_semantic_consistency_encoder_from_content()
        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        if "best_fid" in checkpoint:
            best_fid = float(checkpoint["best_fid"])
        print("Resumed LoRA checkpoint from", checkpoint_path)
        del checkpoint
    elif args.resume:
        print("Warning: resume path not found, training from scratch.")
    else:
        print("Training from base checkpoint ({}).".format(trainable_desc))

    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(args.seed)
            with torch.no_grad():
                evaluate_single_gpu(model, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    max_epochs = None if args.infinite or args.epochs <= 0 else args.epochs
    train_target_desc = "infinite training" if max_epochs is None else f"{max_epochs} epochs"
    print(f"Start LoRA training for {train_target_desc}")
    start_time = time.time()
    epoch = args.start_epoch
    while True:
        train_one_epoch_single_gpu(model, data_loader_train, optimizer, device, epoch,
                                   log_writer=log_writer, args=args)

        completed_epoch = epoch + 1
        should_update_latest = (
            completed_epoch % args.save_last_freq == 0 or (max_epochs is not None and completed_epoch == max_epochs)
        )
        if should_update_latest:
            save_named_model_no_ema(
                args=args,
                model_without_ddp=model,
                epoch=epoch,
                checkpoint_name="latest",
                extra_state={"best_fid": best_fid},
            )
            save_model_no_ema(
                args=args,
                model_without_ddp=model,
                epoch=epoch,
                epoch_name="last"
            )

        if args.online_eval and (
            completed_epoch % args.eval_freq == 0 or (max_epochs is not None and completed_epoch == max_epochs)
        ):
            torch.cuda.empty_cache()
            with torch.no_grad():
                metrics = evaluate_single_gpu(model, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            if metrics["fid"] < best_fid:
                best_fid = float(metrics["fid"])
                save_named_model_no_ema(
                    args=args,
                    model_without_ddp=model,
                    epoch=epoch,
                    checkpoint_name="best",
                    extra_state={"best_fid": best_fid, "best_epoch": epoch},
                )
                print(f"Saved new best checkpoint at epoch {epoch} with FID {best_fid:.4f}")
            torch.cuda.empty_cache()

        if log_writer is not None:
            log_writer.flush()

        if max_epochs is not None and epoch + 1 >= max_epochs:
            break
        epoch += 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time:", total_time_str)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
