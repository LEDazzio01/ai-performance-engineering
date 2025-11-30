#!/usr/bin/env bash
#
# Build helper for the CUDA 13 Blackwell TMA demonstrations.
# This script compiles:
#   - ch07/async_prefetch_tma  (1D double-buffered TMA pipeline)
#   - ch10/tma_2d_pipeline_blackwell (2D TMA tile pipeline)
#
# Usage:
#   ./core/scripts/build_tma_demos.sh [--arch sm_100]
#
# Requires CUDA 13.0+ with nvcc on PATH.

set -euo pipefail

ARCH="sm_100"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="${2:?--arch requires an argument (e.g. sm_100)}"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      echo "Unexpected positional argument: $1" >&2
      exit 1
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> Building Blackwell TMA demos (ARCH=${ARCH})"

pushd "${repo_root}/ch07" >/dev/null
echo "-> ch07/async_prefetch_tma"
make ARCH="${ARCH}" async_prefetch_tma
popd >/dev/null

pushd "${repo_root}/ch10" >/dev/null
echo "-> ch10/tma_2d_pipeline_blackwell"
make ARCH="${ARCH}" tma_2d_pipeline_blackwell
popd >/dev/null

echo "Builds complete. Binaries:"
echo "  - ${repo_root}/ch07/async_prefetch_tma"
echo "  - ${repo_root}/ch10/tma_2d_pipeline_blackwell"
