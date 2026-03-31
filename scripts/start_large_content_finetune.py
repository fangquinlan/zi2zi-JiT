#!/usr/bin/env python3
"""Launch JiT-L/16 LoRA finetuning with content encoder updates enabled."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

MODEL_FOLDER_URL = "https://drive.google.com/drive/folders/1QJi2ihxDBK2NF-jCE07g59YwuUTAd-iY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start JiT-L/16 LoRA finetuning with content encoder finetuning enabled.",
    )
    parser.add_argument("--dataset-dir", default="data/mixed_finetune_dataset")
    parser.add_argument("--output-dir", default="run/lora_ft_large_content_encoder")
    parser.add_argument("--base-checkpoint", default="models/zi2zi-JiT-L-16.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--blr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default="qkv,proj,w12,w3")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-last-freq", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--gen-bsz", type=int, default=8)
    parser.add_argument("--num-images", type=int, default=400)
    parser.add_argument("--cfg", type=float, default=2.4)
    parser.add_argument("--sampling-method", default="ab2")
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--proj-dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-chars-per-font", type=int, default=None)
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--no-auto-download-checkpoint", action="store_false", dest="auto_download_checkpoint")
    parser.set_defaults(auto_download_checkpoint=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_dataset_info(dataset_dir: Path) -> dict:
    dataset_info_path = dataset_dir / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(
            f"dataset_info.json not found under {dataset_dir}. Run prepare_mixed_finetune_dataset.py first."
        )
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_or_install_gdown():
    try:
        import gdown  # type: ignore
        return gdown
    except ImportError:
        print("`gdown` not found. Installing it automatically...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown  # type: ignore
        return gdown


def maybe_download_checkpoint(base_checkpoint: Path, dry_run: bool) -> None:
    if base_checkpoint.is_file():
        return
    if dry_run:
        return

    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    gdown = import_or_install_gdown()

    print(f"Checkpoint not found locally. Downloading from: {MODEL_FOLDER_URL}")
    downloaded = gdown.download_folder(
        url=MODEL_FOLDER_URL,
        output=str(base_checkpoint.parent),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    if not downloaded:
        raise RuntimeError("gdown did not download any files from the model folder.")
    if not base_checkpoint.is_file():
        downloaded_names = ", ".join(Path(path).name for path in downloaded)
        raise FileNotFoundError(
            f"Downloaded files do not include {base_checkpoint.name}. Got: {downloaded_names}"
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = (repo_root / args.dataset_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    base_checkpoint = (repo_root / args.base_checkpoint).resolve()
    train_script = repo_root / "lora_single_gpu_finetune_jit.py"

    dataset_info = load_dataset_info(dataset_dir)
    train_root = dataset_dir / "train"
    test_npz = dataset_dir / "test.npz"

    if not train_root.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_root}")
    if not test_npz.is_file():
        raise FileNotFoundError(f"test.npz not found: {test_npz}")
    if not train_script.is_file():
        raise FileNotFoundError(f"Training entrypoint not found: {train_script}")
    if not base_checkpoint.is_file() and args.auto_download_checkpoint:
        maybe_download_checkpoint(base_checkpoint, dry_run=args.dry_run)
    if not base_checkpoint.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")

    num_fonts = int(dataset_info["num_fonts"])
    num_chars = int(dataset_info["num_chars"])

    cmd = [
        sys.executable,
        str(train_script),
        "--data_path", str(train_root),
        "--test_npz_path", str(test_npz),
        "--output_dir", str(output_dir),
        "--base_checkpoint", str(base_checkpoint),
        "--model", "JiT-L/16",
        "--num_fonts", str(num_fonts),
        "--num_chars", str(num_chars),
        "--img_size", "256",
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--lora_targets", args.lora_targets,
        "--train_content_encoder",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--blr", str(args.blr),
        "--warmup_epochs", str(args.warmup_epochs),
        "--weight_decay", str(args.weight_decay),
        "--save_last_freq", str(args.save_last_freq),
        "--proj_dropout", str(args.proj_dropout),
        "--attn_dropout", str(args.attn_dropout),
        "--cfg", str(args.cfg),
        "--sampling_method", args.sampling_method,
        "--num_sampling_steps", str(args.num_sampling_steps),
        "--online_eval",
        "--eval_step_folders",
        "--eval_freq", str(args.eval_freq),
        "--gen_bsz", str(args.gen_bsz),
        "--num_images", str(args.num_images),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--device", args.device,
    ]

    if args.max_chars_per_font is not None:
        cmd.extend(["--max_chars_per_font", str(args.max_chars_per_font)])
    if args.resume:
        cmd.extend(["--resume", args.resume])

    command_text = " ".join(shlex.quote(part) for part in cmd)
    print("Resolved dataset:")
    print(f"  dataset_dir = {dataset_dir}")
    print(f"  num_fonts   = {num_fonts}")
    print(f"  num_chars   = {num_chars}")
    print(f"  checkpoint  = {base_checkpoint}")
    if not base_checkpoint.is_file():
        print("  checkpoint_status = missing on this machine (dry-run only)")
        if args.auto_download_checkpoint:
            print(f"  checkpoint_source = {MODEL_FOLDER_URL}")
    print("\nTraining command:")
    print(command_text)

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
