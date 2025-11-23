"""Torch extension loader for the inline tcgen05 kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load

try:  # Ensure arch_config clamps TORCH_CUDA_ARCH_LIST before building.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional when module unavailable
    arch_config = None  # type: ignore[assignment]

_MODULE = None
_BUILD_ERROR: Optional[str] = None


def load_tcgen05_module():
    """Compile (if needed) and return the inline tcgen05 extension."""
    global _MODULE, _BUILD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _BUILD_ERROR is not None:
        raise RuntimeError(f"tcgen05 inline extension unavailable: {_BUILD_ERROR}")

    repo_root = Path(__file__).resolve().parents[2]
    cutlass_include = repo_root / "third_party" / "cutlass" / "include"
    clang_host = repo_root / "third_party" / "llvm" / "bin" / "clang++"
    ccbin_flag = f"-ccbin={clang_host}" if clang_host.exists() else None
    src = repo_root / "labs" / "fullstack_cluster" / "capstone_kernels_tcgen05.cu"

    extra_cuda_cflags = [
        "-std=c++20",
        "-gencode=arch=compute_100,code=sm_100",
        "-lineinfo",
        f"-I{cutlass_include}",
        "-DCUTE_PREFETCH_COPY_ATOM_DISABLED",
    ]
    if ccbin_flag:
        extra_cuda_cflags.append(ccbin_flag)
    extra_cflags = ["-std=c++20"]

    try:
        _MODULE = load(
            name="blackwell_capstone_tcgen05_inline_ext",
            sources=[str(src)],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on toolchain
        _BUILD_ERROR = (
            "failed to build inline tcgen05 extension. "
            "Requires SM100 hardware and CUDA toolchain with tcgen05 support. "
            f"Original error: {exc}"
        )
        raise RuntimeError(_BUILD_ERROR) from exc

    return _MODULE


__all__ = ["load_tcgen05_module"]
