#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${REPO_ROOT}/.venv-5090"

TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
DOWNLOAD_CONNECTIONS="${DOWNLOAD_CONNECTIONS:-16}"

choose_python() {
  for candidate in python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

version_ge() {
  local a="$1"
  local b="$2"
  local first
  first="$(printf '%s\n%s\n' "${a}" "${b}" | sort -V | head -n1)"
  [[ "${first}" == "${b}" ]]
}

install_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found; skipping Ubuntu package installation."
    return 0
  fi

  local sudo_cmd=()
  if [[ "${EUID}" -ne 0 ]]; then
    sudo_cmd=(sudo)
  fi

  "${sudo_cmd[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${sudo_cmd[@]}" apt-get install -y \
    aria2 \
    git \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-venv
}

check_nvidia_driver() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Install an NVIDIA driver before training on RTX 5090." >&2
    return 1
  fi

  local driver_version
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
  if [[ -z "${driver_version}" ]]; then
    echo "Could not read NVIDIA driver version from nvidia-smi." >&2
    return 1
  fi

  echo "Detected NVIDIA driver: ${driver_version}"
  if ! version_ge "${driver_version}" "570.26"; then
    echo "Driver ${driver_version} is older than 570.26." >&2
    echo "For RTX 5090 / Blackwell with CUDA 12.8 wheels, update the driver first." >&2
    return 1
  fi
}

main() {
  cd "${REPO_ROOT}"

  install_apt_packages
  check_nvidia_driver

  local python_bin
  python_bin="$(choose_python)" || {
    echo "No suitable python3 interpreter found." >&2
    exit 1
  }

  echo "Using Python interpreter: ${python_bin}"
  "${python_bin}" -m venv "${ENV_DIR}"

  local venv_python="${ENV_DIR}/bin/python"
  local venv_pip="${ENV_DIR}/bin/pip"

  "${venv_python}" -m pip install --upgrade pip setuptools wheel

  "${venv_pip}" install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${PYTORCH_INDEX_URL}"

  "${venv_pip}" install \
    "numpy<2" \
    "opencv-python==4.11.0.86" \
    "timm==0.9.12" \
    "tensorboard>=2.16,<2.21" \
    "scipy>=1.11,<1.16" \
    "einops==0.8.1" \
    "gdown==5.2.0" \
    "fonttools" \
    "Pillow" \
    "pytorch-msssim" \
    "lpips" \
    "tqdm" \
    "matplotlib" \
    "torch-fidelity @ git+https://github.com/LTH14/torch-fidelity.git@master"

  "${venv_pip}" install -e .

  echo
  echo "Environment ready at: ${ENV_DIR}"
  echo "Verifying PyTorch + CUDA:"
  "${venv_python}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

  echo
  echo "Recommended next steps:"
  echo "  bash scripts/prepare_mixed_finetune_dataset.sh"
  echo "  bash scripts/start_large_content_finetune.sh"
  echo
  echo "If you have a direct checkpoint URL and want multi-threaded download:"
  echo "  bash scripts/start_large_content_finetune.sh --checkpoint-url '<URL>' --download-tool aria2c --download-connections ${DOWNLOAD_CONNECTIONS}"
}

main "$@"
