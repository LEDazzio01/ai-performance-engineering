"""Unified harness for running Nsight Systems, Nsight Compute, and torch.profiler."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from example_registry import (
    EXAMPLE_BY_NAME,
    EXAMPLES,
    BuildStep,
    Example,
    ExampleKind,
)
from metrics_config import (
    BASE_NCU_METRICS,
    BASE_NSYS_EXTRA_ARGS,
    BASE_NSYS_TRACE_MODULES,
    ProfilerOverrides,
    resolve_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
DEFAULT_TIMEOUT = 900  # seconds

# Add repo root to path for imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUDA_BIN_DIRS = [
    "/usr/local/cuda-13.0/bin",
    "/usr/local/cuda-13/bin",
    "/usr/local/cuda/bin",
]

CUDA_LIB_DIRS = [
    "/usr/local/cuda-13.0/lib64",
    "/usr/local/cuda-13/lib64",
    "/usr/local/cuda/lib64",
]

def log_progress(*parts: str) -> None:
    """Emit a timestamped, flushed progress line for streaming logs."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(part for part in parts if part)
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class RunResult:
    profiler: str
    example: Example
    command: List[str]
    output_dir: Path
    stdout_path: Path
    stderr_path: Path
    duration: float
    exit_code: int
    skipped: bool
    skip_reason: Optional[str] = None
    metrics_path: Optional[Path] = None  # Path to extracted metrics file (CSV, JSON, etc.)


def unique_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def example_run_command(example: Example, repo_root: Path) -> List[str]:
    if example.run_command:
        base_command = list(example.run_command)
    elif example.kind is ExampleKind.PYTHON:
        base_command = [sys.executable, str(example.resolved_path(repo_root))]
    elif example.kind is ExampleKind.CUDA:
        base_command = [str(example.resolved_path(repo_root))]
    elif example.kind is ExampleKind.SHELL:
        base_command = ["bash", str(example.resolved_path(repo_root))]
    else:
        base_command = [str(example.resolved_path(repo_root))]

    return base_command + list(example.default_args)


def resolved_build_steps(example: Example, repo_root: Path) -> List[BuildStep]:
    if example.build_steps:
        return list(example.build_steps)

    if example.kind is ExampleKind.PYTHON:
        return [
            BuildStep(
                command=(
                    sys.executable,
                    "-m",
                    "py_compile",
                    str(example.resolved_source(repo_root)),
                ),
                workdir=repo_root,
                description="Python syntax check",
            )
        ]

    return []


def should_run_build_step(
    example: Example,
    step: BuildStep,
    repo_root: Path,
    force_build: bool,
) -> bool:
    if force_build:
        return True
    if not step.outputs:
        return True

    outputs = [(repo_root / output).resolve() for output in step.outputs]
    if any(not output.exists() for output in outputs):
        return True

    try:
        source_mtime = example.resolved_source(repo_root).stat().st_mtime
    except FileNotFoundError:
        return False

    for output in outputs:
        try:
            if output.stat().st_mtime < source_mtime:
                return True
        except FileNotFoundError:
            return True

    return False


def preparation_output_dir(session_dir: Path, example: Example, category: str) -> Path:
    return session_dir / "prep" / example.name / category


def execute_build_step(
    example: Example,
    step: BuildStep,
    session_dir: Path,
    repo_root: Path,
    description: str,
    context: argparse.Namespace,
    force_build: bool,
) -> RunResult:
    label = description
    out_dir = preparation_output_dir(session_dir, example, label)
    out_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    for output in step.outputs:
        (repo_root / output).parent.mkdir(parents=True, exist_ok=True)

    env = base_env(example)
    env.update(step.env)

    should_run = should_run_build_step(example, step, repo_root, force_build)
    if not should_run:
        log_progress(
            "build",
            example.name,
            label,
            "skip",
            step.description or "",
            "(up-to-date)",
        )
        return RunResult(
            profiler="build",
            example=example,
            command=list(step.command),
            output_dir=out_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            duration=0.0,
            exit_code=0,
            skipped=True,
            skip_reason="up-to-date",
        )

    log_progress("build", example.name, label, "start", format_command(step.command))
    exit_code, duration = run_command(
        list(step.command),
        cwd=step.workdir,
        env=env,
        timeout=DEFAULT_TIMEOUT,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        dry_run=context.dry_run,
    )
    if context.dry_run:
        log_progress("build", example.name, label, "dry-run")
    else:
        status = "ok" if exit_code == 0 else f"fail(exit={exit_code})"
        log_progress(
            "build",
            example.name,
            label,
            status,
            f"duration={duration:.2f}s",
        )

    return RunResult(
        profiler="build",
        example=example,
        command=list(step.command),
        output_dir=out_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration=duration,
        exit_code=exit_code,
        skipped=context.dry_run,
        skip_reason="dry-run" if context.dry_run else None,
    )


def prepare_example(
    example: Example,
    session_dir: Path,
    repo_root: Path,
    context: argparse.Namespace,
) -> Tuple[List[RunResult], bool]:
    build_steps = resolved_build_steps(example, repo_root)
    results: List[RunResult] = []

    for index, step in enumerate(build_steps, start=1):
        label = f"build_{index:02d}"
        result = execute_build_step(
            example,
            step,
            session_dir,
            repo_root,
            label,
            context,
            force_build=context.force_build,
        )
        results.append(result)
        if not result.skipped and result.exit_code != 0:
            break

    success = all(r.skipped or r.exit_code == 0 for r in results)
    return results, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profiling harness for all chapter examples")
    parser.add_argument(
        "--examples",
        nargs="*",
        default=["all"],
        help="Example names to run (default: all). Use --list to view names",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Only run examples matching these tags",
    )
    parser.add_argument(
        "--profile",
        nargs="*",
        default=["all"],
        choices=["all", "nsys", "ncu", "pytorch"],
        help="Profilers to run (default: all)",
    )
    parser.add_argument(
        "--profile-mode",
        action="append",
        choices=["full", "memory", "flops", "modules", "blackwell"],
        help="PyTorch profiler mode (repeat to collect multiple modes)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output",
        help="Root directory for profiler outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs when the target output directory already exists",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Limit the number of examples executed",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples and exit",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force rebuild of all examples before profiling",
    )
    return parser.parse_args()


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)



def missing_modules(modules: Iterable[str]) -> List[str]:
    import importlib.util

    missing: List[str] = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def select_examples(names: List[str], tags: Optional[Iterable[str]]) -> List[Example]:
    if "all" in names:
        selected = list(EXAMPLES)
    else:
        unknown = [name for name in names if name not in EXAMPLE_BY_NAME]
        if unknown:
            raise SystemExit(f"Unknown example name(s): {', '.join(sorted(unknown))}")
        selected = [EXAMPLE_BY_NAME[name] for name in names]

    if tags:
        tags_set = set(tags)
        selected = [ex for ex in selected if tags_set.intersection(ex.tags)]

    return selected


def resolve_profilers(requested: List[str]) -> List[str]:
    if "all" in requested:
        return ["nsys", "ncu", "pytorch"]
    dedup: List[str] = []
    for item in requested:
        if item not in dedup and item != "all":
            dedup.append(item)
    return dedup


def session_directory(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    directory = root / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_command(
    command: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
) -> Tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] {format_command(command)}")
        return 0, 0.0

    start = time.time()
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            exit_code = -1
    duration = time.time() - start
    return exit_code, duration


def base_env(example: Example) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(example.env)

    # Ensure CUDA/Nsight binaries and shared libraries are on PATH/LD_LIBRARY_PATH
    def _prepend_unique(var: str, candidates: Iterable[str]) -> None:
        current = env.get(var, "")
        parts = [p for p in current.split(os.pathsep) if p]
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in parts:
                parts.insert(0, candidate)
        if parts:
            env[var] = os.pathsep.join(parts)

    _prepend_unique("PATH", CUDA_BIN_DIRS)
    _prepend_unique("LD_LIBRARY_PATH", CUDA_LIB_DIRS)

    tmp_dir = Path(env.get("TMPDIR", REPO_ROOT / "output" / "tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_dir)

    chapter_tags = {
        "ch01",
        "ch02",
        "ch13",
        "ch14",
        "ch15",
        "ch16",
        "ch17",
        "ch18",
        "ch19",
        "ch20",
    }
    if chapter_tags.intersection(example.tags):
        env.setdefault("TORCHINDUCTOR_AUTOTUNE", "0")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
    # Always enable Python fault handler for stack traces on crashes
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


def _find_command(cmd: str) -> Optional[str]:
    """Return absolute path to a command, searching CUDA directories when needed."""
    import shutil

    path = shutil.which(cmd)
    if path:
        return path

    for directory in CUDA_BIN_DIRS:
        candidate = Path(directory) / cmd
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _terminate_lingering_nsys() -> None:
    """Best-effort cleanup for stray Nsight Systems agents before NCU runs."""
    subprocess.run(["pkill", "-f", "nsys"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_preconditions(example: Example, profiler: str) -> Optional[str]:
    missing = missing_modules(example.requires_modules)
    if missing:
        return f"missing modules: {', '.join(missing)}"

    command_requirements = list(example.requires_commands)
    if profiler == "nsys":
        command_requirements.append("nsys")
    elif profiler == "ncu":
        command_requirements.append("ncu")

    unavailable = [cmd for cmd in command_requirements if _find_command(cmd) is None]
    if unavailable:
        return f"missing commands: {', '.join(sorted(set(unavailable)))}"

    if example.requires_cuda and not torch.cuda.is_available():
        if example.allow_cpu_fallback:
            return "CUDA device unavailable"
        return "CUDA device required"

    if example.min_cuda_gpus:
        if not torch.cuda.is_available() or torch.cuda.device_count() < example.min_cuda_gpus:
            return f"needs >= {example.min_cuda_gpus} CUDA device(s)"

    return None


def profiler_output_dir(session_dir: Path, profiler: str, example: Example) -> Path:
    return session_dir / profiler / example.name


def extract_nsys_metrics(nsys_rep_path: Path, output_csv: Path) -> bool:
    """Extract metrics from a .nsys-rep file and append to CSV.
    
    Returns True if extraction succeeded, False otherwise.
    """
    if not nsys_rep_path.exists():
        return False
    
    try:
        # Import extraction functions (add REPO_ROOT to path for import)
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from core.profiling.extract_nsys_summary import harvest
        
        # Extract metrics from this file
        metrics = harvest(nsys_rep_path)
        if not metrics:
            log_progress("extract", "nsys", nsys_rep_path.name, "no-metrics")
            return False
        
        # Prepare CSV data
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        tag = nsys_rep_path.stem
        
        # Read existing CSV if it exists, otherwise create new
        existing_rows = []
        if output_csv.exists():
            import csv
            with output_csv.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                existing_rows = list(reader)
        
        # Append new metrics
        new_rows = []
        for entry in metrics:
            record = {"tag": tag, "section": entry.get("section", ""), 
                     "metric": entry.get("metric", ""), "value": entry.get("value", "")}
            new_rows.append(record)
        
        # Write combined CSV
        import csv
        with output_csv.open("w", newline="") as fh:
            fieldnames = ["tag", "section", "metric", "value"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows + new_rows)
        
        log_progress("extract", "nsys", nsys_rep_path.name, "ok", f"{len(new_rows)} metrics")
        return True
    except Exception as exc:
        log_progress("extract", "nsys", nsys_rep_path.name, "failed", str(exc))
        return False


def run_nsys(
    example: Example,
    session_dir: Path,
    context: argparse.Namespace,
    timeout: int,
    overrides: ProfilerOverrides,
) -> RunResult:
    out_dir = profiler_output_dir(session_dir, "nsys", example)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / f"nsys_{example.name}"
    target_command = example_run_command(example, REPO_ROOT)
    trace_modules = overrides.nsys_trace or BASE_NSYS_TRACE_MODULES
    trace_modules = unique_preserve(trace_modules)
    command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "-o",
        str(out_base),
        "-t",
        ",".join(trace_modules),
        "-s",
        "cpu",
        "--python-sampling=true",
        "--python-sampling-frequency=1000",
        "--cudabacktrace=true",
        "--stats=true",
    ]
    nsys_extra = unique_preserve([*BASE_NSYS_EXTRA_ARGS, *overrides.nsys_extra_args])
    command.extend(nsys_extra)
    command.extend(target_command)

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    log_progress("profile", "nsys", example.name, "start", format_command(command))
    exit_code, duration = run_command(
        command,
        cwd=example.resolved_workdir(REPO_ROOT),
        env=base_env(example),
        timeout=timeout,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        dry_run=context.dry_run,
    )
    (out_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))

    if context.dry_run:
        log_progress("profile", "nsys", example.name, "dry-run")
    else:
        status = "ok" if exit_code == 0 else f"fail(exit={exit_code})"
        log_progress(
            "profile",
            "nsys",
            example.name,
            status,
            f"duration={duration:.2f}s",
        )

    # Automatically extract metrics from .nsys-rep file (before creating result)
    metrics_path = None
    if not context.dry_run and exit_code == 0:
        nsys_rep_path = Path(f"{out_base}.nsys-rep")
        session_metrics_csv = session_dir / "nsys_metrics.csv"
        if extract_nsys_metrics(nsys_rep_path, session_metrics_csv):
            metrics_path = session_metrics_csv
    
    result = RunResult(
        profiler="nsys",
        example=example,
        command=command,
        output_dir=out_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration=duration,
        exit_code=exit_code,
        skipped=context.dry_run,
        skip_reason="dry-run" if context.dry_run else None,
        metrics_path=metrics_path,
    )
    if not context.dry_run:
        write_status(out_dir, exit_code=exit_code, duration=duration)
    
    return result


def run_ncu(
    example: Example,
    session_dir: Path,
    context: argparse.Namespace,
    timeout: int,
    overrides: ProfilerOverrides,
) -> RunResult:
    out_dir = profiler_output_dir(session_dir, "ncu", example)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / f"ncu_{example.name}"
    _terminate_lingering_nsys()
    target_command = example_run_command(example, REPO_ROOT)
    command = [
        "ncu",
        "--set",
        "full",
        "-o",
        str(out_base),
    ]
    metrics = unique_preserve([*BASE_NCU_METRICS, *overrides.ncu_metrics])
    if metrics:
        command.extend(["--metrics", ",".join(metrics)])
        command.extend(["--replay-mode", "kernel"])
    command.extend(target_command)

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    log_progress("profile", "ncu", example.name, "start", format_command(command))
    exit_code, duration = run_command(
        command,
        cwd=example.resolved_workdir(REPO_ROOT),
        env=base_env(example),
        timeout=timeout,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        dry_run=context.dry_run,
    )
    (out_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))

    if context.dry_run:
        log_progress("profile", "ncu", example.name, "dry-run")
    else:
        status = "ok" if exit_code == 0 else f"fail(exit={exit_code})"
        log_progress(
            "profile",
            "ncu",
            example.name,
            status,
            f"duration={duration:.2f}s",
        )

    result = RunResult(
        profiler="ncu",
        example=example,
        command=command,
        output_dir=out_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration=duration,
        exit_code=exit_code,
        skipped=context.dry_run,
        skip_reason="dry-run" if context.dry_run else None,
        metrics_path=None,  # NCU metrics extraction can be added later if needed
    )
    if not context.dry_run:
        write_status(out_dir, exit_code=exit_code, duration=duration)
    return result


def run_pytorch_profiler(
    example: Example,
    session_dir: Path,
    context: argparse.Namespace,
    modes: Sequence[str],
    timeout: int,
) -> List[RunResult]:
    results: List[RunResult] = []
    runner = Path(__file__).resolve().with_name("pytorch_profiler_runner.py")

    for mode in modes:
        out_dir = profiler_output_dir(session_dir, f"pytorch_{mode}", example)
        command = [
            PYTHON,
            str(runner),
            str(example.resolved_path(REPO_ROOT)),
            "--output-dir",
            str(out_dir),
            "--profile-mode",
            mode,
        ]
        if example.default_args:
            command.append("--script-args")
            command.extend(example.default_args)

        stdout_path = out_dir / "stdout.log"
        stderr_path = out_dir / "stderr.log"

        log_progress(
            "profile",
            f"pytorch_{mode}",
            example.name,
            "start",
            format_command(command),
        )
        exit_code, duration = run_command(
            command,
            cwd=example.resolved_workdir(REPO_ROOT),
            env=base_env(example),
            timeout=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            dry_run=context.dry_run,
        )
        (out_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))

        if context.dry_run:
            log_progress("profile", f"pytorch_{mode}", example.name, "dry-run")
        else:
            status = "ok" if exit_code == 0 else f"fail(exit={exit_code})"
            log_progress(
                "profile",
                f"pytorch_{mode}",
                example.name,
                status,
                f"duration={duration:.2f}s",
            )

        results.append(
            RunResult(
                profiler=f"pytorch_{mode}",
                example=example,
                command=command,
                output_dir=out_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                duration=duration,
                exit_code=exit_code,
                skipped=context.dry_run,
                skip_reason="dry-run" if context.dry_run else None,
                metrics_path=None,  # PyTorch profiler metrics extraction can be added later if needed
            )
        )
        if not context.dry_run:
            write_status(out_dir, exit_code=exit_code, duration=duration)
    return results


def _find_deep_profile_json(session_dir: Path, example_name: str) -> Optional[Path]:
    """Find deep profiling JSON report for an example in the session directory.
    
    Searches in ncu/ subdirectory for deep_profile_*.json files.
    """
    # Look in ncu subdirectory first
    ncu_dir = session_dir / "ncu" / example_name
    if ncu_dir.exists():
        for json_file in ncu_dir.glob("deep_profile_*.json"):
            return json_file
    
    # Also check for files directly named after the example
    for json_file in session_dir.rglob(f"*{example_name}*deep_profile*.json"):
        return json_file
    
    return None


def _run_differential_analysis(
    baseline_json: Path,
    optimized_json: Path,
    session_dir: Path,
    base_name: str,
) -> Optional[str]:
    """Run differential analysis between baseline and optimized deep profiles.
    
    Returns markdown string with differential analysis, or None if analysis fails.
    """
    try:
        from core.analysis.differential_profile_analyzer import (
            analyze_differential,
            generate_markdown_report,
        )
        
        report = analyze_differential(baseline_json, optimized_json)
        
        # Save JSON report
        diff_json_path = session_dir / f"differential_analysis_{base_name}.json"
        diff_json_path.write_text(json.dumps(report.to_dict(), indent=2))
        
        # Generate markdown
        markdown = generate_markdown_report(report)
        
        # Also save markdown
        diff_md_path = session_dir / f"differential_analysis_{base_name}.md"
        diff_md_path.write_text(markdown)
        
        log_progress("differential", base_name, "analysis-complete", f"speedup={report.overall_speedup:.2f}x")
        
        return markdown
    except ImportError as e:
        log_progress("differential", base_name, "import-error", str(e))
        return None
    except Exception as e:
        log_progress("differential", base_name, "analysis-error", str(e))
        return None


def generate_baseline_optimized_comparison(results: List[RunResult], session_dir: Path) -> Optional[str]:
    """Generate markdown comparison table for baseline vs optimized examples.
    
    Returns markdown string if comparisons found, None otherwise.
    """
    import csv
    from collections import defaultdict
    
    # Group results by example name (extract base name)
    baseline_results = {}
    optimized_results = {}
    
    for result in results:
        if result.skipped or result.exit_code != 0:
            continue
        
        example_name = result.example.name
        example_path = str(result.example.path).lower() if result.example.path else ""
        
        # Check if it's a baseline or optimized example (check both name and path)
        is_baseline = (
            example_name.startswith("baseline_") 
            or "_baseline_" in example_name.lower()
            or "/baseline_" in example_path
            or "baseline_" in example_path.split("/")[-1] if "/" in example_path else False
        )
        is_optimized = (
            example_name.startswith("optimized_") 
            or "_optimized_" in example_name.lower()
            or "/optimized_" in example_path
            or "optimized_" in example_path.split("/")[-1] if "/" in example_path else False
        )
        
        if is_baseline:
            # Extract base name from name or path
            base_name = None
            if example_name.startswith("baseline_") or "_baseline_" in example_name.lower():
                base_name = example_name.replace("baseline_", "").replace("_baseline", "").replace("baseline_", "")
            elif "/baseline_" in example_path:
                # Extract from path: ch04/baseline_dataparallel.py -> dataparallel
                file_part = example_path.split("/baseline_")[-1].replace(".py", "")
                base_name = file_part.split("_")[0] if "_" in file_part else file_part
            
            if base_name:
                # Remove chapter prefix if present (ch04_dataparallel -> dataparallel)
                if base_name.startswith("ch") and "_" in base_name:
                    base_name = "_".join(base_name.split("_")[1:])
                baseline_results[base_name] = result
        
        if is_optimized:
            # Extract base name from name or path
            base_name = None
            if example_name.startswith("optimized_") or "_optimized_" in example_name.lower():
                base_name = example_name.replace("optimized_", "").replace("_optimized", "").replace("optimized_", "")
            elif "/optimized_" in example_path:
                # Extract from path: ch04/optimized_dataparallel_ddp.py -> dataparallel
                file_part = example_path.split("/optimized_")[-1].replace(".py", "")
                # Take first part before underscore (or whole if no underscore)
                base_name = file_part.split("_")[0] if "_" in file_part and not file_part.startswith("ch") else file_part
            
            if base_name:
                # Remove chapter prefix if present
                if base_name.startswith("ch") and "_" in base_name:
                    base_name = "_".join(base_name.split("_")[1:])
                optimized_results[base_name] = result
    
    # Find matching pairs
    all_base_names = set(baseline_results.keys()) & set(optimized_results.keys())
    
    if not all_base_names:
        return None
    
    # Load metrics from nsys_metrics.csv (tagged by example name)
    nsys_metrics_csv = session_dir / "nsys_metrics.csv"
    metrics_by_example = defaultdict(dict)
    
    if nsys_metrics_csv.exists():
        with nsys_metrics_csv.open("r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tag = row.get("tag", "")
                section = row.get("section", "")
                metric = row.get("metric", "")
                value = row.get("value", "")
                if tag and metric and value:
                    # Extract example name from tag (e.g., "nsys_baseline_dataparallel" -> "baseline_dataparallel")
                    tag_clean = tag.replace("nsys_", "")
                    key = f"{section}::{metric}" if section else metric
                    metrics_by_example[tag_clean][key] = value
    
    # Also try to load from metrics_summary.json flattened metrics
    metrics_summary_path = session_dir / "metrics_summary.json"
    if metrics_summary_path.exists():
        try:
            metrics_data = json.loads(metrics_summary_path.read_text())
            # Map metrics from by_source to examples
            for source_type, sources in metrics_data.get("by_source", {}).items():
                for source_name, metrics in sources.items():
                    # Try to extract example name from source_name
                    # e.g., "nsys_baseline_dataparallel.nsys-rep" -> "baseline_dataparallel"
                    example_match = None
                    for base_name in all_base_names:
                        if f"baseline_{base_name}" in source_name.lower():
                            example_match = f"baseline_{base_name}"
                            break
                        elif f"optimized_{base_name}" in source_name.lower():
                            example_match = f"optimized_{base_name}"
                            break
                    
                    if example_match and metrics:
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                metrics_by_example[example_match][metric_name] = str(metric_value)
        except Exception:
            pass  # Fallback to CSV if JSON parsing fails
    
    # Build comparison table
    markdown_lines = []
    markdown_lines.append("\n" + "=" * 100)
    markdown_lines.append("# Baseline vs Optimized Comparison")
    markdown_lines.append("=" * 100)
    markdown_lines.append("")
    
    for base_name in sorted(all_base_names):
        baseline_result = baseline_results[base_name]
        optimized_result = optimized_results[base_name]
        
        baseline_example_name = baseline_result.example.name
        optimized_example_name = optimized_result.example.name
        
        baseline_metrics = metrics_by_example.get(baseline_example_name, {})
        optimized_metrics = metrics_by_example.get(optimized_example_name, {})
        
        # Collect all unique metrics
        all_metrics = set(baseline_metrics.keys()) | set(optimized_metrics.keys())
        
        # Always include duration from RunResult
        baseline_duration = baseline_result.duration
        optimized_duration = optimized_result.duration
        duration_speedup = baseline_duration / optimized_duration if optimized_duration > 0 else 1.0
        duration_improvement = (1.0 - (optimized_duration / baseline_duration)) * 100 if baseline_duration > 0 else 0.0
        
        markdown_lines.append(f"## {base_name}")
        markdown_lines.append("")
        markdown_lines.append(f"**Baseline:** `{baseline_example_name}`  |  **Optimized:** `{optimized_example_name}`")
        markdown_lines.append("")
        markdown_lines.append("| Metric | Baseline | Optimized | Improvement |")
        markdown_lines.append("|--------|----------|-----------|-------------|")
        
        # Add duration as first row
        markdown_lines.append(f"| Duration (s) | {baseline_duration:.3f} | {optimized_duration:.3f} | {duration_improvement:+.1f}% ({duration_speedup:.2f}x faster) |")
        
        if not all_metrics:
            markdown_lines.append("")
            markdown_lines.append("*No detailed metrics found. Duration comparison only.*")
            markdown_lines.append("")
            continue
        
        # Compare each metric
        for metric_key in sorted(all_metrics):
            baseline_val_str = baseline_metrics.get(metric_key, "N/A")
            optimized_val_str = optimized_metrics.get(metric_key, "N/A")
            
            if baseline_val_str == "N/A" and optimized_val_str == "N/A":
                continue
            
            # Try to parse numeric values
            try:
                # Clean the string values
                baseline_clean = str(baseline_val_str).replace(",", "").replace(" ms", "").replace(" GB", "").replace("%", "").replace(" TB", "").strip()
                optimized_clean = str(optimized_val_str).replace(",", "").replace(" ms", "").replace(" GB", "").replace("%", "").replace(" TB", "").strip()
                
                baseline_val = float(baseline_clean)
                optimized_val = float(optimized_clean)
                
                # Calculate improvement
                # For time/duration/latency metrics: lower is better
                # For throughput/bandwidth/utilization metrics: higher is better
                metric_lower = metric_key.lower()
                is_time_metric = any(term in metric_lower for term in ["time", "duration", "latency", "ms", "us", "ns"])
                
                if is_time_metric:
                    # Lower is better
                    improvement = ((baseline_val - optimized_val) / baseline_val * 100) if baseline_val > 0 else 0.0
                    speedup = baseline_val / optimized_val if optimized_val > 0 else 1.0
                    improvement_str = f"{improvement:+.1f}% ({speedup:.2f}x faster)"
                else:
                    # Higher is better (throughput, bandwidth, utilization, etc.)
                    improvement = ((optimized_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0.0
                    speedup = optimized_val / baseline_val if baseline_val > 0 else 1.0
                    improvement_str = f"{improvement:+.1f}% ({speedup:.2f}x higher)"
                
                markdown_lines.append(f"| {metric_key} | {baseline_val_str} | {optimized_val_str} | {improvement_str} |")
            except (ValueError, AttributeError):
                # Non-numeric comparison
                if baseline_val_str == optimized_val_str:
                    markdown_lines.append(f"| {metric_key} | {baseline_val_str} | {optimized_val_str} | Same |")
                else:
                    markdown_lines.append(f"| {metric_key} | {baseline_val_str} | {optimized_val_str} | Changed |")
        
        # Look for deep profiling JSON reports and run differential analysis
        baseline_deep_profile = _find_deep_profile_json(session_dir, baseline_example_name)
        optimized_deep_profile = _find_deep_profile_json(session_dir, optimized_example_name)
        
        if baseline_deep_profile and optimized_deep_profile:
            differential_md = _run_differential_analysis(
                baseline_deep_profile, 
                optimized_deep_profile,
                session_dir,
                base_name
            )
            if differential_md:
                markdown_lines.append("")
                markdown_lines.append(differential_md)
        
        markdown_lines.append("")
    
    markdown_lines.append("=" * 100)
    markdown_lines.append("")
    
    return "\n".join(markdown_lines)


def summarize(results: List[RunResult], session_dir: Path) -> None:
    summary = []
    failures: List[RunResult] = []
    for result in results:
        entry = {
            "profiler": result.profiler,
            "example": result.example.name,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration,
            "command": result.command,
            "stdout": str(result.stdout_path),
            "stderr": str(result.stderr_path),
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
        }
        # Include metrics path if available
        if result.metrics_path:
            entry["metrics_path"] = str(result.metrics_path)
            # For nsys, include relative path to .nsys-rep file for metric_extractor discovery
            if result.profiler == "nsys":
                nsys_rep = result.output_dir / f"nsys_{result.example.name}.nsys-rep"
                if nsys_rep.exists():
                    entry["nsys_rep_path"] = str(nsys_rep)
        summary.append(entry)
        if not result.skipped and result.exit_code != 0:
            failures.append(result)
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    
    # Aggregate all metrics using unified metric extractor
    try:
        # REPO_ROOT is core/scripts/, need code/ for tools import
        code_root = REPO_ROOT.parent if REPO_ROOT.name == "scripts" else REPO_ROOT
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        from core.analysis.metric_extractor import discover_and_extract_all, flatten_metrics
        
        all_metrics = discover_and_extract_all(session_dir)
        flattened = flatten_metrics(all_metrics)
        
        if flattened:
            metrics_summary_path = session_dir / "metrics_summary.json"
            metrics_summary_path.write_text(json.dumps({
                "session_dir": str(session_dir),
                "by_source": all_metrics,
                "flattened": flattened,
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2))
            log_progress("session", str(session_dir), "metrics-aggregated", f"{len(flattened)} total metrics from all profilers")
        
        # Log nsys-specific metrics CSV status
        nsys_metrics_csv = session_dir / "nsys_metrics.csv"
        if nsys_metrics_csv.exists():
            import csv
            with nsys_metrics_csv.open("r") as fh:
                reader = csv.DictReader(fh)
                metric_count = sum(1 for _ in reader)
            log_progress("session", str(session_dir), "nsys-metrics", f"{metric_count} nsys metrics in CSV")
        
        # Generate baseline vs optimized comparison
        comparison_md = generate_baseline_optimized_comparison(results, session_dir)
        if comparison_md:
            comparison_path = session_dir / "baseline_optimized_comparison.md"
            comparison_path.write_text(comparison_md)
            log_progress("session", str(session_dir), "comparison-generated", comparison_path.name)
            # Also print to stdout for immediate visibility
            print("\n" + comparison_md)
    except Exception as exc:
        # Don't fail the whole run if metrics aggregation fails
        log_progress("session", str(session_dir), "metrics-aggregation-warning", str(exc))
        latest_summary = REPO_ROOT / "profile_runs" / "harness" / "latest_summary.json"
        latest_failures = REPO_ROOT / "profile_runs" / "harness" / "latest_failures.txt"
        latest_summary.write_text(json.dumps(summary, indent=2))
        if failures:
            latest_failures.write_text(
                "\n".join(
                    f"{item.example.name} [{item.profiler}] (exit={item.exit_code})"
                    for item in failures
                )
                + "\n"
            )
        else:
            latest_failures.write_text("All tasks succeeded.\n")


def maybe_skip_output(out_dir: Path, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    summary = out_dir / "command.json"
    if not out_dir.exists() or not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text())
    except Exception:
        return False
    status_path = out_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return False
    return status.get("exit_code", 1) == 0


def write_status(out_dir: Path, *, exit_code: int, duration: float) -> None:
    status_path = out_dir / "status.json"
    payload = {
        "exit_code": exit_code,
        "duration_seconds": duration,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    status_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()

    if args.list:
        for example in EXAMPLES:
            print(f"{example.name:35s} :: tags={','.join(example.tags)} :: path={example.path}")
        return

    profilers = resolve_profilers(args.profile)
    cli_pytorch_modes = args.profile_mode or ["full"]

    selected = select_examples(args.examples, args.tags)
    if args.max_examples is not None:
        selected = selected[: args.max_examples]

    session_dir = session_directory(args.output_root)
    all_results: List[RunResult] = []

    import torch  # local import so command listing works without CUDA

    total_examples = len(selected)

    for index, example in enumerate(selected, start=1):
        log_progress("example", f"{index}/{total_examples}", example.name, "start")
        timeout = example.timeout_seconds or DEFAULT_TIMEOUT
        overrides = resolve_overrides(example)

        build_results, build_ok = prepare_example(example, session_dir, REPO_ROOT, args)
        all_results.extend(build_results)
        if not build_ok:
            print(f"[skip] {example.name} -> build failed")
            log_progress("example", f"{index}/{total_examples}", example.name, "build-failed")
            continue

        for profiler in profilers:
            if profiler == "pytorch":
                if example.kind is not ExampleKind.PYTHON:
                    continue
                requested_modes: List[str] = list(cli_pytorch_modes)
                for mode in overrides.pytorch_modes:
                    if mode not in requested_modes:
                        requested_modes.append(mode)

                modes_to_run: List[str] = []
                for mode in requested_modes:
                    out_dir = profiler_output_dir(session_dir, f"pytorch_{mode}", example)
                    if maybe_skip_output(out_dir, args.skip_existing):
                        log_progress(
                            "profile",
                            f"pytorch_{mode}",
                            example.name,
                            "skip",
                            "existing-output",
                        )
                        all_results.append(
                            RunResult(
                                profiler=f"pytorch_{mode}",
                                example=example,
                                command=[],
                                output_dir=out_dir,
                                stdout_path=out_dir / "stdout.log",
                                stderr_path=out_dir / "stderr.log",
                                duration=0.0,
                                exit_code=0,
                                skipped=True,
                                skip_reason="existing-output",
                            )
                        )
                    else:
                        modes_to_run.append(mode)
                if not modes_to_run:
                    continue

                reason = check_preconditions(example, "pytorch")
                if reason:
                    print(f"[skip] {example.name} (pytorch) -> {reason}")
                    log_progress("profile", "pytorch", example.name, "skip", reason)
                    for mode in modes_to_run:
                        out_dir = profiler_output_dir(session_dir, f"pytorch_{mode}", example)
                        all_results.append(
                            RunResult(
                                profiler=f"pytorch_{mode}",
                                example=example,
                                command=[],
                                output_dir=out_dir,
                                stdout_path=out_dir / "stdout.log",
                                stderr_path=out_dir / "stderr.log",
                                duration=0.0,
                                exit_code=0,
                                skipped=True,
                                skip_reason=reason,
                            )
                        )
                    continue

                results = run_pytorch_profiler(example, session_dir, args, modes_to_run, timeout)
                all_results.extend(results)
                continue

            out_dir = profiler_output_dir(session_dir, profiler, example)
            if maybe_skip_output(out_dir, args.skip_existing):
                log_progress("profile", profiler, example.name, "skip", "existing-output")
                all_results.append(
                    RunResult(
                        profiler=profiler,
                        example=example,
                        command=[],
                        output_dir=out_dir,
                        stdout_path=out_dir / "stdout.log",
                        stderr_path=out_dir / "stderr.log",
                        duration=0.0,
                        exit_code=0,
                        skipped=True,
                        skip_reason="existing-output",
                    )
                )
                continue

            reason = check_preconditions(example, profiler)
            if reason:
                print(f"[skip] {example.name} ({profiler}) -> {reason}")
                log_progress("profile", profiler, example.name, "skip", reason)
                all_results.append(
                    RunResult(
                        profiler=profiler,
                        example=example,
                        command=[],
                        output_dir=out_dir,
                        stdout_path=out_dir / "stdout.log",
                        stderr_path=out_dir / "stderr.log",
                        duration=0.0,
                        exit_code=0,
                        skipped=True,
                        skip_reason=reason,
                    )
                )
                continue

            if profiler == "nsys":
                result = run_nsys(example, session_dir, args, timeout, overrides)
                all_results.append(result)
            elif profiler == "ncu":
                result = run_ncu(example, session_dir, args, timeout, overrides)
                all_results.append(result)
            else:
                raise AssertionError(f"Unknown profiler {profiler}")

        log_progress("example", f"{index}/{total_examples}", example.name, "complete")

    summarize(all_results, session_dir)
    failures = [r for r in all_results if not r.skipped and r.exit_code != 0]
    failure_list = REPO_ROOT / "profile_runs" / "harness" / "latest_failures.txt"
    failure_list.parent.mkdir(parents=True, exist_ok=True)
    if failures:
            failure_list.write_text(
                "\n".join(
                    f"{item.example.name} [{item.profiler}] (exit={item.exit_code})"
                    for item in failures
                )
                + "\n"
            )
    else:
        failure_list.write_text("All tasks succeeded.\n")
    log_progress("session", str(session_dir), "summary-written")

    failed = [r for r in all_results if not r.skipped and r.exit_code != 0]
    if failed:
        print("\nFailures detected:")
        for item in failed:
            print(f" - {item.example.name} [{item.profiler}] (exit={item.exit_code})")
        sys.exit(1)


if __name__ == "__main__":
    import shutil
    import torch

    main()
