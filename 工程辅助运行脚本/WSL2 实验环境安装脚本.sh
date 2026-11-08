#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -d "${PROJECT_ROOT}/WSL2 与 Python 依赖文件" ]; then
  ENVIRONMENT_ROOT="${PROJECT_ROOT}/WSL2 与 Python 依赖文件"
else
  ENVIRONMENT_ROOT="${PROJECT_ROOT}/environment"
fi

REQUESTED_KIND="all"
VERIFY_ONLY="false"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --kind)
      REQUESTED_KIND="${2:?--kind requires main, yolo, stats, or all}"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY="true"
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "${REQUESTED_KIND}" in
  main|yolo|stats|all) ;;
  *)
    echo "ERROR: --kind must be main, yolo, stats, or all." >&2
    exit 2
    ;;
esac

if ! grep -qi microsoft /proc/version; then
  echo "ERROR: this setup script must run inside WSL2 Ubuntu." >&2
  exit 2
fi

if [ "${REQUESTED_KIND}" != "stats" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable inside WSL2. Update the Windows NVIDIA driver." >&2
  exit 2
fi

VENV_ROOT="${HOME}/.morphoroute/venvs"
TOOLS_ROOT="${HOME}/.morphoroute/tools"
UV_PYTHON_INSTALL_DIR="${HOME}/.morphoroute/python"
mkdir -p "${VENV_ROOT}"
mkdir -p "${TOOLS_ROOT}"
export UV_PYTHON_INSTALL_DIR

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi
  if [ ! -x "${TOOLS_ROOT}/uv" ]; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "ERROR: curl is required to bootstrap isolated Python 3.10." >&2
      exit 2
    fi
    curl -LsSf https://astral.sh/uv/install.sh |
      env UV_UNMANAGED_INSTALL="${TOOLS_ROOT}" sh >&2
  fi
  printf '%s/uv' "${TOOLS_ROOT}"
}

UV_EXECUTABLE="$(ensure_uv)"

requirements_for() {
  printf '%s/requirements-%s.txt' "${ENVIRONMENT_ROOT}" "$1"
}

python_for() {
  printf '%s/%s/bin/python' "${VENV_ROOT}" "$1"
}

lock_for() {
  printf '%s/environment-lock-%s.json' "${ENVIRONMENT_ROOT}" "$1"
}

verify_environment() {
  local kind="$1"
  local python_path
  python_path="$(python_for "${kind}")"
  [ -x "${python_path}" ] || return 1
  case "${kind}" in
    main)
      "${python_path}" -c "import torch, mmcv, mmengine, mmdet; assert torch.cuda.is_available()"
      ;;
    yolo)
      "${python_path}" -c "import torch, ultralytics; assert torch.cuda.is_available()"
      ;;
    stats)
      "${python_path}" -c "import numpy, pandas, scipy, sklearn, skimage, cv2, statsmodels, matplotlib"
      ;;
  esac
}

write_lock() {
  local kind="$1"
  local python_path
  local lock_path
  python_path="$(python_for "${kind}")"
  lock_path="$(lock_for "${kind}")"
  "${python_path}" - "${kind}" "${lock_path}" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

kind, lock_path = sys.argv[1:3]
packages = {
    distribution.metadata["Name"]: distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
try:
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
except (FileNotFoundError, subprocess.CalledProcessError):
    gpu = None
payload = {
    "environment": kind,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "gpu": gpu,
    "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
}
canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
payload["lock_sha256"] = hashlib.sha256(canonical).hexdigest()
Path(lock_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

install_environment() {
  local kind="$1"
  local environment_path="${VENV_ROOT}/${kind}"
  local python_path="${environment_path}/bin/python"
  local requirements_path
  local pypi_index
  requirements_path="$(requirements_for "${kind}")"
  pypi_index="${MORPHOROUTE_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

  if verify_environment "${kind}" >/dev/null 2>&1; then
    echo "ENV-${kind}: already healthy; reusing."
    [ -f "$(lock_for "${kind}")" ] || write_lock "${kind}"
    return 0
  fi
  if [ "${VERIFY_ONLY}" = "true" ]; then
    echo "ERROR: ENV-${kind} is missing or unhealthy." >&2
    return 3
  fi

  # A previous installation may have been interrupted after creating the
  # virtual environment.  Reuse that valid interpreter so rerunning this
  # script is resumable instead of failing (or destructively clearing it).
  if [ ! -x "${python_path}" ]; then
    "${UV_EXECUTABLE}" venv \
      --python 3.10 \
      --managed-python \
      --seed \
      "${environment_path}"
  else
    echo "ENV-${kind}: reusing partial environment and resuming installation."
  fi
  "${python_path}" -m pip install --upgrade \
    pip==23.3.2 setuptools==69.0.3 wheel==0.42.0 \
    --index-url "${pypi_index}"

  if [ "${kind}" = "main" ] || [ "${kind}" = "yolo" ]; then
    # Keep the exact CUDA 11.8 builds from the original environment design.
    # Reuse the verified local wheel when present so ENV-MAIN and ENV-YOLO do
    # not each download the same 2.3 GiB artifact.
    local torch_local_wheel="${MORPHOROUTE_TORCH_LOCAL_WHEEL:-/mnt/c/tmp/torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl}"
    local torch_expected_sha256="60396358193f238888540f4a38d78485f161e28ec17fa445f0373b5350ef21f0"
    local torch_requirement="torch==2.1.2+cu118"
    if [ -f "${torch_local_wheel}" ]; then
      echo "${torch_expected_sha256}  ${torch_local_wheel}" | sha256sum --check -
      torch_requirement="${torch_local_wheel}"
      echo "ENV-${kind}: installing verified local PyTorch CUDA 11.8 wheel."
    fi
    "${python_path}" -m pip install \
      "${torch_requirement}" torchvision==0.16.2+cu118 \
      --index-url "${pypi_index}" \
      --find-links "${MORPHOROUTE_PYTORCH_CU118_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu118/}"
  fi
  "${python_path}" -m pip install \
    -r "${requirements_path}" \
    --index-url "${pypi_index}"
  if [ "${kind}" = "main" ]; then
    "${python_path}" -m pip install "mmcv==2.1.0" \
      --index-url "${pypi_index}" \
      --find-links "${MORPHOROUTE_MMCV_CU118_LINKS:-https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html}"
  fi
  verify_environment "${kind}"
  write_lock "${kind}"
  echo "ENV-${kind}: installation and verification complete."
}

if [ "${REQUESTED_KIND}" = "all" ]; then
  install_environment "main"
  install_environment "yolo"
  install_environment "stats"
else
  install_environment "${REQUESTED_KIND}"
fi
