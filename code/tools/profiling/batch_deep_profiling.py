#!/usr/bin/env python3
"""
Batch Nsight profiling + deep-report generator across representative chapter workloads.

Examples:
    # List available workloads
    python tools/batch_deep_profiling.py --list

    # Profile a subset
    python tools/batch_deep_profiling.py --workload ch10_double_buffered_pipeline

    # Profile everything (writes into profiles/)
    python tools/batch_deep_profiling.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = REPO_ROOT / "profiles"
DEEP_PROFILER = REPO_ROOT / "tools" / "deep_profiling_report.py"

# Nsight metrics tuned for Blackwell (SM 12.x instruction counters + L1 bytes).
DEFAULT_NCU_METRICS = [
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "gpu__time_duration.sum",
]


@dataclass(slots=True)
class Workload:
    name: str
    chapter: str
    command: List[str]
    workdir: Path
    description: str
    metrics: List[str] = field(default_factory=list)

    def effective_metrics(self) -> List[str]:
        return self.metrics or DEFAULT_NCU_METRICS


WORKLOADS: List[Workload] = [
    Workload(
        name="ch07_hbm3e_copy",
        chapter="ch7",
        command=["./hbm3e_optimized_copy"],
        workdir=REPO_ROOT / "ch7",
        description="HBM3e-optimised memcpy (memory-bound baseline).",
    ),
    Workload(
        name="ch08_occupancy_tuning",
        chapter="ch8",
        command=["./occupancy_tuning"],
        workdir=REPO_ROOT / "ch8",
        description="Occupancy tuning microbenchmark (launch configuration study).",
    ),
    Workload(
        name="ch09_cutlass_gemm",
        chapter="ch9",
        command=["./cutlass_gemm_example"],
        workdir=REPO_ROOT / "ch9",
        description="CUTLASS GEMM example (tensor core throughput).",
    ),
    Workload(
        name="ch10_double_buffered_pipeline",
        chapter="ch10",
        command=["./double_buffered_pipeline", "512", "512", "512"],
        workdir=REPO_ROOT / "ch10",
        description="Double-buffered cooperative GEMM (async pipeline showcase).",
    ),
    Workload(
        name="ch11_basic_streams",
        chapter="ch11",
        command=["./basic_streams"],
        workdir=REPO_ROOT / "ch11",
        description="Multi-stream overlap example (command queueing).",
    ),
    Workload(
        name="ch12_cuda_graphs",
        chapter="ch12",
        command=["./cuda_graphs"],
        workdir=REPO_ROOT / "ch12",
        description="CUDA Graphs launch amortisation example.",
    ),
]


def run_command(cmd: List[str], *, cwd: Path, env: dict[str, str] | None = None, capture: Path | None = None) -> None:
    if capture:
        capture.parent.mkdir(parents=True, exist_ok=True)
        with capture.open("w") as fh:
            subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=fh, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_tools_available() -> None:
    for tool in ("nsys", "ncu"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found in PATH; install NVIDIA Nsight Systems/Compute.")
    if not DEEP_PROFILER.exists():
        raise SystemExit(f"Deep profiling script missing: {DEEP_PROFILER}")


def format_metrics(metrics: Iterable[str]) -> str:
    return ",".join(metrics)


def profile_workload(
    workload: Workload,
    *,
    profiles_dir: Path,
    dry_run: bool,
    skip_nsys: bool,
    skip_ncu: bool,
    skip_report: bool,
) -> None:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    base = profiles_dir / workload.name

    nsys_base = base
    ncu_csv = profiles_dir / f"{workload.name}_metrics.csv"
    report_md = profiles_dir / f"{workload.name}_report.md"
    report_json = profiles_dir / f"{workload.name}_analysis.json"

    print(f"\n=== Profiling {workload.name} ({workload.description}) ===")
    print(f"Command: {' '.join(workload.command)}")

    if not skip_nsys:
        nsys_cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--cuda-memory-usage=true",
            "--sample=none",
            "--force-overwrite=true",
            f"--output={nsys_base}",
            *workload.command,
        ]
        print("-> Nsight Systems:", " ".join(nsys_cmd))
        if not dry_run:
            run_command(nsys_cmd, cwd=workload.workdir)
    else:
        print("-> Nsight Systems: skipped")

    if not skip_ncu:
        metrics = format_metrics(workload.effective_metrics())
        ncu_cmd = [
            "ncu",
            "--metrics",
            metrics,
            "--csv",
            "--force-overwrite",
            *workload.command,
        ]
        print("-> Nsight Compute:", " ".join(ncu_cmd))
        if not dry_run:
            run_command(ncu_cmd, cwd=workload.workdir, capture=ncu_csv)
    else:
        print("-> Nsight Compute: skipped")

    if not skip_report:
        report_cmd = [
            sys.executable,
            str(DEEP_PROFILER),
            "--ncu-csv",
            str(ncu_csv),
            "--nsys-report",
            str(nsys_base) + ".nsys-rep",
            "--output-json",
            str(report_json),
            "--print-markdown",
        ]
        print("-> Deep profiling report:", " ".join(report_cmd))
        if not dry_run:
            output = subprocess.run(
                report_cmd,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            report_md.write_text(output.stdout)
            print(f"   Markdown saved to {report_md}")
            print(f"   JSON saved to {report_json}")
    else:
        print("-> Deep profiling report: skipped")


def list_workloads() -> None:
    for workload in WORKLOADS:
        print(f"{workload.name:>32}  ({workload.chapter})  {workload.description}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Nsight profiling across chapter workloads.")
    parser.add_argument(
        "--workload",
        action="append",
        help="Specific workload(s) to process (default: all).",
    )
    parser.add_argument("--list", action="store_true", help="List workloads and exit.")
    parser.add_argument("--profiles-dir", type=Path, default=PROFILES_DIR, help="Output directory for reports/metrics.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--skip-nsys", action="store_true", help="Skip Nsight Systems capture.")
    parser.add_argument("--skip-ncu", action="store_true", help="Skip Nsight Compute capture.")
    parser.add_argument("--skip-report", action="store_true", help="Skip deep profiling report generation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_workloads()
        return 0

    ensure_tools_available()

    selected = WORKLOADS
    if args.workload:
        names = set(args.workload)
        selected = [wl for wl in WORKLOADS if wl.name in names]
        missing = names - {wl.name for wl in selected}
        if missing:
            print(f"Unknown workload(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    for workload in selected:
        try:
            profile_workload(
                workload,
                profiles_dir=args.profiles_dir,
                dry_run=args.dry_run,
                skip_nsys=args.skip_nsys,
                skip_ncu=args.skip_ncu,
                skip_report=args.skip_report,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[error] Failed: {' '.join(exc.cmd)} (exit {exc.returncode})", file=sys.stderr)
            return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
