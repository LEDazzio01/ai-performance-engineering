"""Sample GPU power consumption while running a command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

try:
    import pynvml
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("pynvml is required for power monitoring. Install with `pip install nvidia-ml-py`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("GPU power monitor")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute under monitoring")
    parser.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds")
    parser.add_argument("--gpus", type=str, help="Comma-separated GPU indices to monitor (default: all)")
    parser.add_argument("--output-json", type=Path, help="Optional path to write metrics")
    return parser.parse_args()


def _select_devices(arg: str | None) -> List[int]:
    pynvml.nvmlInit()
    total = pynvml.nvmlDeviceGetCount()
    if arg is None:
        return list(range(total))
    indices = [int(idx.strip()) for idx in arg.split(",") if idx.strip()]
    for idx in indices:
        if idx < 0 or idx >= total:
            raise SystemExit(f"Invalid GPU index {idx}; available 0..{total-1}")
    return indices


def monitor_power(command: List[str], interval: float, devices: List[int]) -> Dict:
    if not command:
        raise SystemExit("Provide a command to execute, e.g. -- python my_script.py")

    handles = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in devices]
    proc = subprocess.Popen(command)

    samples: List[Dict] = []
    energy_joules = 0.0
    prev_timestamp: float | None = None
    prev_total_power = 0.0

    try:
        while True:
            timestamp = time.time()
            per_device = []
            total_power = 0.0
            for handle in handles:
                mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                watts = mw / 1000.0
                per_device.append(watts)
                total_power += watts

            samples.append({"timestamp": timestamp, "per_device_watts": per_device, "total_watts": total_power})
            if prev_timestamp is not None:
                dt = timestamp - prev_timestamp
                energy_joules += prev_total_power * dt
            prev_timestamp = timestamp
            prev_total_power = total_power

            if proc.poll() is not None:
                break
            time.sleep(interval)
    finally:
        proc.wait()
        pynvml.nvmlShutdown()

    totals = [sample["total_watts"] for sample in samples]
    device_series = list(zip(*[sample["per_device_watts"] for sample in samples])) if samples else []
    per_device_stats = [
        {
            "gpu_index": devices[idx],
            "min_watts": min(series),
            "max_watts": max(series),
            "avg_watts": mean(series),
        }
        for idx, series in enumerate(device_series)
    ]

    metrics = {
        "command": command,
        "samples": len(samples),
        "duration": samples[-1]["timestamp"] - samples[0]["timestamp"] if samples else 0.0,
        "total_power": {
            "min_watts": min(totals) if totals else 0.0,
            "max_watts": max(totals) if totals else 0.0,
            "avg_watts": mean(totals) if totals else 0.0,
        },
        "energy_joules": energy_joules,
        "per_device": per_device_stats,
    }

    return metrics


def main() -> None:
    args = parse_args()
    devices = _select_devices(args.gpus)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = sys.argv[1:]
        if command and command[0] == "--":
            command = command[1:]

    metrics = monitor_power(command, args.interval, devices)

    print("=== GPU Power Metrics ===")
    print(f"Duration: {metrics['duration']:.2f}s")
    total = metrics["total_power"]
    print(
        "Total power (W): min={min:.2f} avg={avg:.2f} max={max:.2f}".format(
            min=total["min_watts"], avg=total["avg_watts"], max=total["max_watts"]
        )
    )
    print(f"Energy: {metrics['energy_joules'] / 1000:.2f} kJ")
    for entry in metrics["per_device"]:
        print(
            f"GPU{entry['gpu_index']}: min={entry['min_watts']:.2f}"
            f" avg={entry['avg_watts']:.2f} max={entry['max_watts']:.2f} W"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics written to {args.output_json}")


if __name__ == "__main__":
    main()
