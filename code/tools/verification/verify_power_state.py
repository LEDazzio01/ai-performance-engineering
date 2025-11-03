#!/usr/bin/env python3
"""Verify GPU power and clock states via nvidia-smi."""

from __future__ import annotations

import subprocess
import sys

QUERY = [
    "nvidia-smi",
    "--query-gpu=name,pstate,clocks.sm,clocks.mem,power.draw,power.limit",
    "--format=csv,noheader,nounits",
]


def run_nvidia_smi() -> list[str]:
    try:
        output = subprocess.check_output(QUERY, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to execute {' '.join(QUERY)}: {exc}") from exc
    return [line.strip() for line in output.strip().splitlines() if line.strip()]


def main() -> int:
    try:
        rows = run_nvidia_smi()
    except RuntimeError as exc:
        print(f"[verify_power_state] {exc}", file=sys.stderr)
        return 1

    issues = 0
    def normalize(field: str) -> tuple[str, bool]:
        if field.upper() in {"[N/A]", "N/A", ""}:
            return "not reported", False
        return field, True

    for idx, row in enumerate(rows):
        name, pstate, sm_clock, mem_clock, power_draw, power_limit = [part.strip() for part in row.split(',')]
        print(f"Device {idx}: {name}")
        print(f"  Performance state: P{pstate[-1] if pstate.startswith('P') else pstate}")
        sm_clock_str, sm_valid = normalize(sm_clock)
        mem_clock_str, mem_valid = normalize(mem_clock)
        power_limit_str, limit_valid = normalize(power_limit)
        power_draw_str, _ = normalize(power_draw)
        print(f"  SM clock: {sm_clock_str}{' MHz' if sm_valid else ''}")
        print(f"  Memory clock: {mem_clock_str}{' MHz' if mem_valid else ''}")
        if limit_valid:
            print(f"  Power draw: {power_draw_str} W (limit {power_limit_str} W)")
        else:
            print(f"  Power draw: {power_draw_str} W (power limit not reported)")

        if not pstate.startswith("P0"):
            print("  ⚠️  GPU not in maximum performance state (P0). Consider persisting clocks or enabling persistence mode.")
            issues += 1
        print()

    if issues:
        print("[verify_power_state] One or more GPUs are not in P0.")
        return 1

    print("[verify_power_state] All GPUs are in P0 with reported clocks/power above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
