#!/usr/bin/env python3
"""Generate a minimal repro bundle for torch.compile issues."""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def detect_gpu_info() -> dict:
    """Collect basic GPU diagnostics using torch if available."""
    info: dict = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count()
        info["torch_version"] = torch.__version__
        info["cuda_version_reported"] = getattr(torch.version, "cuda", "unknown")

        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["devices"].append(
                {
                    "id": idx,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    except ImportError:
        info["error"] = "torch not importable"
    except Exception as exc:  # pragma: no cover - diagnostics only
        info["error"] = f"torch probe failed: {exc}"

    return info


def detect_driver_versions() -> dict:
    """Capture NVIDIA driver versions via nvidia-smi if available."""
    info: dict = {}
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-driver-version", "--format=csv,noheader"], text=True)
        info["nvidia_driver_version"] = output.strip()
    except FileNotFoundError:
        info["nvidia_driver_version"] = "nvidia-smi not found"
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        info["nvidia_driver_version"] = f"nvidia-smi failed: {exc}"
    return info


def build_command(args: argparse.Namespace) -> str:
    base = [
        "python",
        "ch16/test_gpt_large_optimized.py",
        f"--model {args.model}",
        f"--sequence-length {args.sequence_length}",
        f"--batch-size {args.batch_size}",
        "--compile-mode reduce-overhead",
    ]
    if args.tensor_parallel_gpus:
        base.append(f"--tensor-parallel-gpus {args.tensor_parallel_gpus}")
    if args.skip_compile:
        base.append("--skip-compile")
    if args.extra_flags:
        base.append(args.extra_flags)
    return " \\\n+  ".join(base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a torch.compile repro bundle.")
    parser.add_argument("--model", default="llama40b", help="Model identifier for the repro notes")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-gpus", type=int, default=8)
    parser.add_argument("--skip-compile", action="store_true", help="Generate eager baseline command instead")
    parser.add_argument("--extra-flags", default="", help="Additional flags to append to the benchmark command")
    parser.add_argument("--output-dir", required=True, help="Directory where the repro bundle will be written")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "arguments": vars(args),
        "command_template": build_command(args),
    }
    metadata.update(detect_driver_versions())
    metadata["torch_cuda_info"] = detect_gpu_info()

    output_path = out_dir / "torch_compile_repro.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    readme_path = out_dir / "README.txt"
    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "Torch Compile Repro Bundle\n"
            "==========================\n\n"
            "Command Template:\n"
        )
        handle.write(metadata["command_template"] + "\n\n")
        handle.write(
            "Next Steps:\n"
            "1. Run the command above with and without --skip-compile.\n"
            "2. Capture logs (stdout/stderr) and Nsight traces if the process hangs.\n"
            "3. Attach this directory when filing upstream issues.\n"
        )

    print(f"Repro bundle written to {out_dir}")


if __name__ == "__main__":
    main()
