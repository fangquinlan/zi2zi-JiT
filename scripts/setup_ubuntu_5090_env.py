#!/usr/bin/env python3
"""Prepare an isolated Ubuntu environment for RTX 5090 finetuning."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the .venv-5090 environment and install Ubuntu training dependencies.",
    )
    parser.add_argument("--env-dir", default=".venv-5090")
    parser.add_argument("--torch-version", default=os.environ.get("TORCH_VERSION", "2.7.1"))
    parser.add_argument("--torchvision-version", default=os.environ.get("TORCHVISION_VERSION", "0.22.1"))
    parser.add_argument(
        "--pytorch-index-url",
        default=os.environ.get("PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128"),
    )
    parser.add_argument("--download-connections", type=int, default=int(os.environ.get("DOWNLOAD_CONNECTIONS", "16")))
    parser.add_argument("--skip-apt", action="store_true")
    parser.add_argument("--skip-driver-check", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def version_ge(current: str, minimum: str) -> bool:
    def to_tuple(text: str) -> tuple[int, ...]:
        return tuple(int(part) for part in text.split("."))

    return to_tuple(current) >= to_tuple(minimum)


def install_apt_packages(skip_apt: bool) -> None:
    if skip_apt:
        print("Skipping apt package installation by request.")
        return
    if shutil.which("apt-get") is None:
        print("apt-get not found; skipping Ubuntu package installation.")
        return

    prefix: list[str] = []
    if os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if sudo is None:
            raise RuntimeError("apt-get is available, but this user is not root and `sudo` is missing.")
        prefix = [sudo]

    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    run([*prefix, "apt-get", "update"], env=env)
    run(
        [
            *prefix,
            "apt-get",
            "install",
            "-y",
            "aria2",
            "fontforge",
            "git",
            "libgl1",
            "libglib2.0-0",
            "potrace",
            "python3",
            "python3-fontforge",
            "python3-pip",
            "python3-venv",
        ],
        env=env,
    )


def check_nvidia_driver(skip_driver_check: bool) -> None:
    if skip_driver_check:
        print("Skipping NVIDIA driver version check by request.")
        return
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found. Install an NVIDIA driver before training on RTX 5090.")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    driver_version = result.stdout.strip().splitlines()[0].strip()
    if not driver_version:
        raise RuntimeError("Could not read NVIDIA driver version from nvidia-smi.")

    print(f"Detected NVIDIA driver: {driver_version}")
    if not version_ge(driver_version, "570.26"):
        raise RuntimeError(
            f"Driver {driver_version} is older than 570.26. "
            "Update the driver before training on RTX 5090 with CUDA 12.8 wheels."
        )


def create_venv(repo_root: Path, env_dir: Path) -> tuple[Path, Path]:
    run([sys.executable, "-m", "venv", str(env_dir)], cwd=repo_root)
    venv_python = env_dir / "bin" / "python"
    venv_pip = env_dir / "bin" / "pip"
    if not venv_python.is_file():
        raise FileNotFoundError(f"Virtualenv python not found: {venv_python}")
    return venv_python, venv_pip


def install_python_packages(
    repo_root: Path,
    venv_python: Path,
    venv_pip: Path,
    *,
    torch_version: str,
    torchvision_version: str,
    pytorch_index_url: str,
) -> None:
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=repo_root)
    run(
        [
            str(venv_pip),
            "install",
            f"torch=={torch_version}",
            f"torchvision=={torchvision_version}",
            "--index-url",
            pytorch_index_url,
        ],
        cwd=repo_root,
    )
    run(
        [
            str(venv_pip),
            "install",
            "numpy<2",
            "opencv-python==4.11.0.86",
            "brotli",
            "timm==0.9.12",
            "tensorboard>=2.16,<2.21",
            "scipy>=1.11,<1.16",
            "einops==0.8.1",
            "gdown==5.2.0",
            "fonttools",
            "Pillow",
            "pytorch-msssim",
            "lpips",
            "tqdm",
            "matplotlib",
            "torch-fidelity @ git+https://github.com/LTH14/torch-fidelity.git@master",
        ],
        cwd=repo_root,
    )
    run([str(venv_pip), "install", "-e", "."], cwd=repo_root)


def verify_environment(venv_python: Path) -> None:
    print()
    print("Environment ready.")
    print("Verifying PyTorch + CUDA:")
    run(
        [
            str(venv_python),
            "-c",
            (
                "import torch; "
                "print('torch:', torch.__version__); "
                "print('torch cuda runtime:', torch.version.cuda); "
                "print('cuda available:', torch.cuda.is_available()); "
                "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    env_dir = (repo_root / args.env_dir).resolve()

    install_apt_packages(args.skip_apt)
    check_nvidia_driver(args.skip_driver_check)
    venv_python, venv_pip = create_venv(repo_root, env_dir)
    install_python_packages(
        repo_root,
        venv_python,
        venv_pip,
        torch_version=args.torch_version,
        torchvision_version=args.torchvision_version,
        pytorch_index_url=args.pytorch_index_url,
    )
    verify_environment(venv_python)

    print()
    print("Recommended next steps:")
    print("  python scripts/prepare_mixed_finetune_dataset.py --input-root train --output-dir data/mixed_finetune_dataset --force")
    print("  python scripts/start_large_content_finetune.py")
    print()
    print("If you have a direct checkpoint URL and want multi-threaded download:")
    print(
        "  python scripts/start_large_content_finetune.py "
        f"--checkpoint-url '<URL>' --download-tool aria2c --download-connections {args.download_connections}"
    )


if __name__ == "__main__":
    main()
