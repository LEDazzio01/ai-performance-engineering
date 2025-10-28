"""Simple harness to orchestrate recurring benchmark suites."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Continuous benchmark runner")
    parser.add_argument("config", type=Path, help="JSON config describing benchmarks to execute")
    parser.add_argument("--artifact-dir", type=Path, default=Path("benchmark_runs"), help="Directory to store aggregated results")
    parser.add_argument("--tag", help="Optional tag to annotate the run (e.g. commit hash)")
    parser.add_argument("--stop-on-fail", action="store_true", help="Abort remaining benchmarks on first failure")
    return parser.parse_args()


def load_config(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit("Benchmark config must be a list of job definitions")
    return data


def run_command(job: Dict) -> Dict:
    command = job.get("command")
    if not isinstance(command, list):
        raise SystemExit(f"Job missing 'command' list: {job}")
    workdir = Path(job.get("workdir", "."))
    output_path = job.get("output_json")

    start = time.time()
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
    duration = time.time() - start

    payload = {
        "name": job.get("name", "benchmark"),
        "command": command,
        "workdir": str(workdir),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_sec": duration,
    }

    if output_path:
        try:
            payload["benchmark_output"] = json.loads(Path(output_path).read_text())
        except FileNotFoundError:
            payload["benchmark_output_error"] = f"Output file not found: {output_path}"
        except json.JSONDecodeError as exc:
            payload["benchmark_output_error"] = f"Failed to parse JSON ({exc})"

    return payload


def main() -> None:
    args = parse_args()
    jobs = load_config(args.config)
    results: List[Dict] = []

    for job in jobs:
        print(f"Running benchmark: {job.get('name', job.get('command'))}")
        payload = run_command(job)
        results.append(payload)
        if payload["returncode"] != 0:
            print(f"  ❌ Benchmark failed (rc={payload['returncode']})")
            if args.stop_on_fail:
                break
        else:
            print(f"  ✅ Completed in {payload['duration_sec']:.2f}s")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outfile = args.artifact_dir / f"benchmark_run_{timestamp}.json"
    summary = {
        "tag": args.tag,
        "timestamp": timestamp,
        "benchmarks": results,
    }
    outfile.write_text(json.dumps(summary, indent=2))
    print(f"Saved run summary to {outfile}")


if __name__ == "__main__":
    main()
