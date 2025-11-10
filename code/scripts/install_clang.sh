#!/usr/bin/env bash
# Fetch and install a self-contained LLVM/Clang toolchain under third_party/llvm.
set -euo pipefail

VERSION="${1:-${LLVM_VERSION:-18.1.8}}"
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)
        LLVM_SUFFIX="${LLVM_SUFFIX:-x86_64-linux-gnu-ubuntu-22.04}"
        ;;
    aarch64|arm64)
        LLVM_SUFFIX="${LLVM_SUFFIX:-aarch64-linux-gnu}"
        ;;
    *)
        echo "Unsupported architecture: ${ARCH}. Please install clang manually." >&2
        exit 1
        ;;
esac

ARCHIVE_NAME="${LLVM_ARCHIVE:-clang+llvm-${VERSION}-${LLVM_SUFFIX}.tar.xz}"
BASE_URL="${LLVM_BASE_URL:-https://github.com/llvm/llvm-project/releases/download}"
DOWNLOAD_URL="${LLVM_URL:-${BASE_URL}/llvmorg-${VERSION}/${ARCHIVE_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/third_party/llvm"

echo "Installing LLVM/Clang (${VERSION}) into ${DEST_DIR}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/llvm.XXXXXX")"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

ARCHIVE_PATH="${tmp_dir}/${ARCHIVE_NAME}"
echo "Downloading ${DOWNLOAD_URL}"
curl -L --retry 3 --retry-delay 2 -o "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
tar -xJf "${ARCHIVE_PATH}" --strip-components=1 -C "${DEST_DIR}"

echo "LLVM/Clang installed at ${DEST_DIR}"
