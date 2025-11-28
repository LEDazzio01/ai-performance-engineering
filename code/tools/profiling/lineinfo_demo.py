"""
Build and run a tiny CUDA extension compiled with -lineinfo to verify source/line capture.
Usage:
  python -m tools.profiling.lineinfo_demo
This builds a torch extension in a temp dir, runs a single kernel, and prints the output path.
Use with Nsight:
  ncu --set full --section SourceCounters --section SpeedOfLight --force-overwrite --export /tmp/lineinfo_demo.ncu-rep \\
      python -m tools.profiling.lineinfo_demo
  ncu --import /tmp/lineinfo_demo.ncu-rep --csv --page source | head
"""
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def build_extension():
    src = Path(__file__).with_suffix(".cu")
    build_dir = Path(tempfile.mkdtemp(prefix="lineinfo_demo_build_"))
    ext = load(
        name="lineinfo_demo_ext",
        sources=[str(src)],
        verbose=False,
        extra_cuda_cflags=["-lineinfo"],
        extra_include_paths=[],
        build_directory=str(build_dir),
    )
    return ext


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this demo")
    ext = build_extension()
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    torch.cuda.nvtx.range_push("lineinfo_demo")
    out = ext.forward(a, b)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print("Output checksum:", out.sum().item())


if __name__ == "__main__":
    main()
