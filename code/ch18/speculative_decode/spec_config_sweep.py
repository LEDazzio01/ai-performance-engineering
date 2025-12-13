"""Sweep speculator configs and emit acceptance/chunk metrics to artifacts.

This is a chapter tool (not a comparable baseline/optimized benchmark).
Run via `python -m cli.aisp tools spec-config-sweep -- --config-dir ...`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.artifact_manager import ArtifactManager  # noqa: E402
from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402
from ch18.run_vllm_decoder import GraphMode, VLLMMoEInferenceBenchmark  # noqa: E402


def _discover_config_paths(config_dir: Path, explicit: Optional[str]) -> List[Path]:
    if explicit:
        paths = [Path(p).expanduser() for p in explicit.split(",") if p.strip()]
    else:
        paths = sorted(config_dir.glob("*.json")) + sorted(config_dir.glob("*.yml")) + sorted(config_dir.glob("*.yaml"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep speculator configs and emit summary JSON.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "ch18" / "spec_configs",
        help="Directory containing speculator config files.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=os.getenv("SPEC_SWEEP_CONFIGS"),
        help="Comma-separated list of explicit config paths (overrides --config-dir).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.getenv("SPEC_SWEEP_ITER", "2")),
        help="Iterations per config.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.getenv("SPEC_SWEEP_WARMUP", "1")),
        help="Warmup iterations per config.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base directory for writing artifacts.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.getenv("SPEC_SWEEP_RUN_ID", f"spec_config_sweep_{int(time.time())}"),
        help="Run identifier (subdirectory under --artifacts-dir).",
    )
    args = parser.parse_args()

    config_paths = _discover_config_paths(args.config_dir, args.configs)
    if not config_paths:
        raise FileNotFoundError(f"No speculator configs found under {args.config_dir}")

    artifacts = ArtifactManager(base_dir=args.artifacts_dir, run_id=args.run_id)
    results: Dict[str, Dict[str, float]] = {}

    for cfg_path in config_paths:
        bench = VLLMMoEInferenceBenchmark()
        bench.spec_config_path = cfg_path
        bench.graph_mode = GraphMode.EAGER
        bench.enable_graphs = False

        cfg = bench.get_config()
        cfg.iterations = int(args.iterations)
        cfg.warmup = int(args.warmup)

        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=cfg)
        run_result = harness.benchmark(bench)
        metrics = run_result.custom_metrics or {}
        results[cfg_path.name] = {
            "accept_rate": float(metrics.get("optimized_moe.spec_accept_rate", 0.0)),
            "chunk_size": float(metrics.get("optimized_moe.spec_chunk_size", 0.0)),
            "throughput_tok_s": float(metrics.get("optimized_moe.throughput_tok_s", 0.0)),
            "ttft_mean_ms": float(metrics.get("optimized_moe.ttft_mean_ms", 0.0)),
            "tpot_mean_ms": float(metrics.get("optimized_moe.tpot_mean_ms", 0.0)),
        }

    payload = {
        "run_id": artifacts.run_id,
        "config_paths": [str(p) for p in config_paths],
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "results": results,
    }
    out_path = artifacts.get_result_path("spec_config_sweep.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote sweep results to {out_path}")


if __name__ == "__main__":
    main()
