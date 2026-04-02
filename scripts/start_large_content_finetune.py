#!/usr/bin/env python3
"""Launch JiT-L/16 LoRA finetuning with content encoder updates enabled."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _launcher_utils import maybe_reexec_with_repo_python

maybe_reexec_with_repo_python(__file__, REPO_ROOT)

MODEL_FOLDER_URL = "https://drive.google.com/drive/folders/1QJi2ihxDBK2NF-jCE07g59YwuUTAd-iY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start JiT-L/16 LoRA finetuning with content encoder finetuning enabled.",
    )
    parser.add_argument("--dataset-dir", default="data/mixed_finetune_dataset")
    parser.add_argument("--output-dir", default="run/lora_ft_large_content_encoder")
    parser.add_argument("--base-checkpoint", default="models/zi2zi-JiT-L-16.pth")
    parser.add_argument("--upgrade-anti-hallucination", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--infinite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--blr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default="qkv,proj,w12,w3")
    parser.add_argument("--train-style-encoder", action="store_true")
    parser.add_argument("--use-unicode-char-labels", action="store_true")
    parser.add_argument("--use-char-embedding", action="store_true")
    parser.add_argument("--use-ids-conditioning", action="store_true")
    parser.add_argument("--ids-path", default="")
    parser.add_argument("--num-style-refs", type=int, default=None)
    parser.add_argument("--style-ref-mode", choices=["single", "mean", "max"], default=None)
    parser.add_argument("--binary-loss-weight", type=float, default=None)
    parser.add_argument("--edge-loss-weight", type=float, default=None)
    parser.add_argument("--projection-loss-weight", type=float, default=None)
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
    parser.add_argument("--checkpoint-url", default="")
    parser.add_argument("--download-tool", choices=["auto", "aria2c", "gdown"], default="auto")
    parser.add_argument("--download-connections", type=int, default=16)
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


def download_with_aria2c(url: str, output_path: Path, connections: int) -> None:
    aria2c = shutil.which("aria2c")
    if aria2c is None:
        raise FileNotFoundError("`aria2c` is not installed or not on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        aria2c,
        "--continue=true",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--summary-interval=5",
        "--dir", str(output_path.parent),
        "--out", output_path.name,
        url,
    ]
    subprocess.run(cmd, check=True)


def download_with_python(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as f:
        shutil.copyfileobj(response, f)


def maybe_download_checkpoint(
    base_checkpoint: Path,
    *,
    dry_run: bool,
    checkpoint_url: str,
    download_tool: str,
    download_connections: int,
) -> None:
    if base_checkpoint.is_file():
        return
    if dry_run:
        return

    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_url:
        if download_tool in {"auto", "aria2c"} and shutil.which("aria2c") is not None:
            print(
                f"Checkpoint not found locally. Downloading with aria2c "
                f"({download_connections} connections) from: {checkpoint_url}"
            )
            download_with_aria2c(checkpoint_url, base_checkpoint, download_connections)
        elif download_tool == "aria2c":
            raise FileNotFoundError("Requested --download-tool aria2c, but `aria2c` is not installed.")
        else:
            print(f"Checkpoint not found locally. Downloading with Python from: {checkpoint_url}")
            download_with_python(checkpoint_url, base_checkpoint)

        if not base_checkpoint.is_file():
            raise FileNotFoundError(f"Download finished, but checkpoint is still missing: {base_checkpoint}")
        return

    gdown = import_or_install_gdown()
    print(
        "Checkpoint not found locally. Falling back to gdown folder download from "
        f"{MODEL_FOLDER_URL}. This path is not multi-thread accelerated."
    )
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
    repo_root = REPO_ROOT
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
        maybe_download_checkpoint(
            base_checkpoint,
            dry_run=args.dry_run,
            checkpoint_url=args.checkpoint_url,
            download_tool=args.download_tool,
            download_connections=args.download_connections,
        )
    if not base_checkpoint.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")

    upgrade_enabled = args.upgrade_anti_hallucination
    use_unicode_char_labels = args.use_unicode_char_labels or upgrade_enabled or args.use_ids_conditioning
    use_char_embedding = args.use_char_embedding or upgrade_enabled
    use_ids_conditioning = args.use_ids_conditioning
    if use_ids_conditioning and not args.ids_path:
        raise ValueError("--use-ids-conditioning requires --ids-path.")
    num_style_refs = args.num_style_refs if args.num_style_refs is not None else (8 if upgrade_enabled else 1)
    style_ref_mode = args.style_ref_mode or ("mean" if upgrade_enabled else "single")
    binary_loss_weight = args.binary_loss_weight if args.binary_loss_weight is not None else (0.15 if upgrade_enabled else 0.0)
    edge_loss_weight = args.edge_loss_weight if args.edge_loss_weight is not None else (0.10 if upgrade_enabled else 0.0)
    projection_loss_weight = args.projection_loss_weight if args.projection_loss_weight is not None else (0.05 if upgrade_enabled else 0.0)

    num_fonts = int(dataset_info["num_fonts"])
    if use_unicode_char_labels:
        num_chars = int(dataset_info.get("num_unicode_chars", dataset_info["num_chars"]))
    else:
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
        "--num_style_refs", str(num_style_refs),
        "--style_ref_mode", style_ref_mode,
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
    if args.infinite:
        cmd.append("--infinite")
    if args.train_style_encoder:
        cmd.append("--train_style_encoder")
    if use_unicode_char_labels:
        cmd.extend(["--use_unicode_char_labels", "--unicode_codepoints_path", str(dataset_dir / "dataset_info.json")])
    if use_char_embedding:
        cmd.append("--use_char_embedding")
    if use_ids_conditioning:
        cmd.extend(["--use_ids_conditioning", "--ids_path", args.ids_path])
    if binary_loss_weight > 0:
        cmd.extend(["--binary_loss_weight", str(binary_loss_weight)])
    if edge_loss_weight > 0:
        cmd.extend(["--edge_loss_weight", str(edge_loss_weight)])
    if projection_loss_weight > 0:
        cmd.extend(["--projection_loss_weight", str(projection_loss_weight)])
    if args.resume:
        cmd.extend(["--resume", args.resume])

    command_text = " ".join(shlex.quote(part) for part in cmd)
    print("Resolved dataset:")
    print(f"  dataset_dir = {dataset_dir}")
    print(f"  num_fonts   = {num_fonts}")
    print(f"  num_chars   = {num_chars}")
    print(f"  checkpoint  = {base_checkpoint}")
    print(f"  unicode_labels = {use_unicode_char_labels}")
    print(f"  char_embedding = {use_char_embedding}")
    print(f"  ids_conditioning = {use_ids_conditioning}")
    print(f"  num_style_refs = {num_style_refs}")
    print(f"  style_ref_mode = {style_ref_mode}")
    print(f"  binary_loss_weight = {binary_loss_weight}")
    print(f"  edge_loss_weight = {edge_loss_weight}")
    print(f"  projection_loss_weight = {projection_loss_weight}")
    if not base_checkpoint.is_file():
        print("  checkpoint_status = missing on this machine (dry-run only)")
        if args.auto_download_checkpoint and args.checkpoint_url:
            print(f"  checkpoint_source = {args.checkpoint_url}")
            print(f"  download_tool = {args.download_tool}")
            print(f"  download_connections = {args.download_connections}")
        elif args.auto_download_checkpoint:
            print(f"  checkpoint_source = {MODEL_FOLDER_URL}")
    print("\nTraining command:")
    print(command_text)

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
