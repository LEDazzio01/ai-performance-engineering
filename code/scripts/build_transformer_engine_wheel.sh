#!/usr/bin/env bash

set -euo pipefail

VERSION="${1:-v2.8}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="${REPO_ROOT}/third_party/TransformerEngine"

if [[ ! -d "${SRC_DIR}" ]]; then
  git clone --depth 1 --branch "${VERSION}" https://github.com/NVIDIA/TransformerEngine.git "${SRC_DIR}"
else
  git -C "${SRC_DIR}" fetch --tags
  git -C "${SRC_DIR}" checkout "${VERSION}"
fi

git -C "${SRC_DIR}" submodule update --init --recursive

PYTHON_BIN="${PYTHON_BIN:-python}"
CMAKE_PREFIX_PATH="$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

# Locate the Python-installed cuDNN headers/libraries (from nvidia-cudnn-cu13 wheel).
CUDNN_BASE="$(${PYTHON_BIN} - <<'PY'
import pathlib
import sys
import pkgutil
site = pathlib.Path(sys.executable).resolve().parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
wheel_path = site / "nvidia" / "cudnn"
if not wheel_path.exists():
    wheel_path = pathlib.Path.home() / ".local/lib/python{0}.{1}/site-packages/nvidia/cudnn".format(sys.version_info.major, sys.version_info.minor)
print(wheel_path)
PY
)"

export NVTE_FRAMEWORK=pytorch
export CMAKE_PREFIX_PATH
export CUDNN_INCLUDE_PATH="${CUDNN_INCLUDE_PATH:-${CUDNN_BASE}/include}"
export CUDNN_LIBRARY_PATH="${CUDNN_LIBRARY_PATH:-${CUDNN_BASE}/lib}"
export CPATH="${CUDNN_INCLUDE_PATH}:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_LIBRARY_PATH}:${LIBRARY_PATH:-}"

pushd "${SRC_DIR}" >/dev/null
${PYTHON_BIN} -m pip install --upgrade pip setuptools wheel cmake ninja > /dev/null
${PYTHON_BIN} setup.py bdist_wheel
popd >/dev/null

# Build the transformer_engine_torch wheel as well (PyTorch C++/CUDA extension)
PYTORCH_DIR="${SRC_DIR}/transformer_engine/pytorch"
pushd "${PYTORCH_DIR}" >/dev/null
${PYTHON_BIN} setup.py bdist_wheel
popd >/dev/null

echo "Wheel(s) generated under ${SRC_DIR}/dist:"
ls "${SRC_DIR}/dist"
echo
echo "Wheel(s) generated under ${PYTORCH_DIR}/dist:"
ls "${PYTORCH_DIR}/dist"
