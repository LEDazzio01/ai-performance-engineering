#!/usr/bin/env python3
"""Report ECC and MIG configuration via nvidia-smi."""

from __future__ import annotations

import subprocess
import sys

QUERY = [
    "nvidia-smi",
    "--query-gpu=name,ecc.mode.current,mig.mode.current",
    "--format=csv,noheader"
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
        print(f"[verify_ecc_mig] {exc}", file=sys.stderr)
        return 1

    for idx, row in enumerate(rows):
        parts = [part.strip() for part in row.split(',')]
        if len(parts) != 3:
            continue
        name, ecc_mode, mig_mode = parts
        print(f"Device {idx}: {name}")

        def normalize(value: str, label: str) -> tuple[str, bool]:
            lower = value.lower()
            if lower in {"[n/a]", "n/a", "unknown", ""}:
                return f"{label} not reported", False
            return value, True

        ecc_display, ecc_valid = normalize(ecc_mode, "ECC")
        mig_display, mig_valid = normalize(mig_mode, "MIG")
        print(f"  ECC mode: {ecc_display}")
        print(f"  MIG mode: {mig_display}")

        if ecc_valid and ecc_mode.lower() != "off":
            print("  ⚠️  ECC enabled – can reduce peak memory bandwidth. Disable if not required.")
        if mig_valid and mig_mode.lower() != "disabled":
            print("  ⚠️  MIG partitions active – total SMs/memory are reduced.")
        print()

    print("[verify_ecc_mig] Review warnings above for ECC/MIG impact on performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
