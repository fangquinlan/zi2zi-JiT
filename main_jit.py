import argparse
import datetime
import numpy as np
import os
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image

from util.crop import resize_and_random_crop
import util.misc as misc
from util.unicode_labels import (
    load_unicode_codepoints,
    parse_unicode_codepoint_from_name,
)
from util.ids_utils import build_ids_resources

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser


class FontSrcTargetRefsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        ref_size=128,
        max_chars_per_font=None,
        num_style_refs=1,
        use_unicode_char_labels=False,
        unicode_codepoints=None,
    ):
        self.root = root
        self.transform = transform
        self.ref_size = ref_size
        self.max_chars_per_font = max_chars_per_font
        self.num_style_refs = num_style_refs
        self.use_unicode_char_labels = use_unicode_char_labels
        if not (1 <= self.num_style_refs <= 8):
            raise ValueError(f"num_style_refs must be between 1 and 8, got {self.num_style_refs}")

        self.samples = []

        font_dirs = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')
        ])
        self.font_to_idx = {}
        for font in font_dirs:
            idx_str = font.split('_')[0]
            if idx_str.startswith("'"):
                idx_str = idx_str[1:]
            font_idx = int(idx_str) - 1
            self.font_to_idx[font] = font_idx
        self.idx_to_font = {idx: font for font, idx in self.font_to_idx.items()}
        self.num_fonts = max(self.font_to_idx.values()) + 1 if self.font_to_idx else 0

        char_indices_set = set()
        unicode_codepoints_set = set()
        for font_name in font_dirs:
            font_path = os.path.join(root, font_name)
            font_idx = self.font_to_idx[font_name]

            font_samples = []
            for filename in os.listdir(font_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        char_idx = int(filename.split('_')[0])
                    except (ValueError, IndexError):
                        continue
                    unicode_cp = parse_unicode_codepoint_from_name(filename)
                    if self.use_unicode_char_labels and unicode_cp is None:
                        continue
                    img_path = os.path.join(font_path, filename)
                    font_samples.append((img_path, font_idx, char_idx, unicode_cp))
                    char_indices_set.add(char_idx)
                    if unicode_cp is not None:
                        unicode_codepoints_set.add(unicode_cp)

            if max_chars_per_font is not None and len(font_samples) > max_chars_per_font:
                rng = random.Random(font_idx)
                rng.shuffle(font_samples)
                font_samples = font_samples[:max_chars_per_font]

            self.samples.extend(font_samples)

        if self.use_unicode_char_labels:
            if unicode_codepoints is None:
                self.unicode_codepoints = sorted(unicode_codepoints_set)
            else:
                self.unicode_codepoints = [int(cp) for cp in unicode_codepoints]
            self.unicode_to_idx = {cp: idx for idx, cp in enumerate(self.unicode_codepoints)}
            self.num_chars = len(self.unicode_codepoints)
        else:
            self.unicode_codepoints = []
            self.unicode_to_idx = {}
            self.num_chars = max(char_indices_set) + 1 if char_indices_set else 0

    def __len__(self):
        return len(self.samples)

    def _extract_ref_from_grid(self, img, grid_idx, ref_idx):
        grid_x = 512 if grid_idx == 0 else 768
        row = ref_idx // 2
        col = ref_idx % 2
        x1 = grid_x + col * 128
        y1 = row * 128
        x2 = x1 + 128
        y2 = y1 + 128
        return img.crop((x1, y1, x2, y2))

    def _extract_refs(self, img):
        if self.num_style_refs == 1:
            ref_global_idx = random.randint(0, 7)
            grid_idx = ref_global_idx // 4
            ref_idx = ref_global_idx % 4
            ref = self._extract_ref_from_grid(img, grid_idx, ref_idx)
            if ref.size[0] != self.ref_size or ref.size[1] != self.ref_size:
                ref = ref.resize((self.ref_size, self.ref_size), Image.LANCZOS)
            return ref

        ref_indices = random.sample(range(8), self.num_style_refs)
        refs = []
        for ref_global_idx in ref_indices:
            grid_idx = ref_global_idx // 4
            ref_idx = ref_global_idx % 4
            ref = self._extract_ref_from_grid(img, grid_idx, ref_idx)
            if ref.size[0] != self.ref_size or ref.size[1] != self.ref_size:
                ref = ref.resize((self.ref_size, self.ref_size), Image.LANCZOS)
            refs.append(ref)
        return refs

    def __getitem__(self, index):
        img_path, font_idx, char_idx, unicode_cp = self.samples[index]

        img = Image.open(img_path).convert('RGB')

        source = img.crop((0, 0, 256, 256))
        target = img.crop((256, 0, 512, 256))

        ref = self._extract_refs(img)

        to_tensor = transforms.PILToTensor()
        if self.transform is not None:
            rng_state = torch.get_rng_state()
            source = self.transform(source)
            torch.set_rng_state(rng_state)
            target = self.transform(target)
        else:
            source = to_tensor(source)
            target = to_tensor(target)

        if isinstance(ref, list):
            ref = torch.stack([to_tensor(item) for item in ref])
        else:
            ref = to_tensor(ref)

        if self.use_unicode_char_labels:
            char_idx = self.unicode_to_idx[unicode_cp]

        return source, target, ref, (font_idx, char_idx)


def collate_src_target_refs(batch):
    sources = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    refs = torch.stack([item[2] for item in batch])
    font_labels = torch.tensor([item[3][0] for item in batch], dtype=torch.long)
    char_labels = torch.tensor([item[3][1] for item in batch], dtype=torch.long)

    labels = (font_labels, char_labels, refs, sources)
    return targets, labels


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--test_npz_path', default='test_set.npz', type=str,
                        help='Path to test set npz file for evaluation')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--eval_step_folders', action='store_true',
                        help='Save evaluation outputs in per-step subfolders (step_N/)')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--num_fonts', default=100, type=int,
                        help='Number of font classes')
    parser.add_argument('--num_chars', default=500, type=int,
                        help='Number of character classes')
    parser.add_argument('--max_chars_per_font', default=None, type=int,
                        help='Max characters per font (None for no limit)')
    parser.add_argument('--num_style_refs', default=1, type=int,
                        help='Number of style reference glyphs sampled from the 8-grid per training sample.')
    parser.add_argument('--style_ref_mode', default='single', type=str,
                        choices=['single', 'mean', 'max'],
                        help='How to aggregate multiple style reference embeddings when num_style_refs > 1.')
    parser.add_argument('--use_unicode_char_labels', action='store_true',
                        help='Use Unicode-aware shared character labels instead of per-font local indices.')
    parser.add_argument('--unicode_codepoints_path', default='', type=str,
                        help='Optional JSON/TXT file that stores the shared Unicode codepoint list.')
    parser.add_argument('--use_char_embedding', action='store_true',
                        help='Add an explicit character embedding/token into the conditioning path.')
    parser.add_argument('--use_ids_conditioning', action='store_true',
                        help='Inject IDS token conditioning derived from Unicode labels.')
    parser.add_argument('--ids_path', default='', type=str,
                        help='Path to an ids_lv*.txt file from yi-bai/ids.')
    parser.add_argument('--binary_loss_weight', default=0.0, type=float,
                        help='Weight for auxiliary ink-mask reconstruction loss.')
    parser.add_argument('--edge_loss_weight', default=0.0, type=float,
                        help='Weight for auxiliary edge-structure loss.')
    parser.add_argument('--projection_loss_weight', default=0.0, type=float,
                        help='Weight for row/column projection consistency loss.')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
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

    # Create denoiser
    model = Denoiser(args)
    if ids_resources is not None:
        model.net.y_embedder.set_ids_lookup(ids_resources["ids_token_ids"], ids_resources["ids_token_mask"])
        if ids_resources["missing_codepoints"]:
            print(
                f"Warning: IDS file does not cover {len(ids_resources['missing_codepoints'])} training codepoints. "
                "Those characters will receive a zero IDS embedding."
            )

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
