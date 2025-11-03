#!/usr/bin/env python3
"""Extract summary metrics from Nsight Systems reports (CSV or .nsys-rep)."""

import argparse
import csv
import io
import pathlib
import subprocess
import sys
from typing import Iterable, List, Dict


def _read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def _run_nsys_stats(rep_path: pathlib.Path) -> List[Dict[str, str]]:
    command = [
        "nsys",
        "stats",
        "--report",
        "summary",
        "--format",
        "csv",
        str(rep_path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("nsys binary not found on PATH; install Nsight Systems to extract summaries") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"nsys stats failed for {rep_path}: {exc.stderr.strip()}") from exc

    reader = csv.DictReader(io.StringIO(proc.stdout))
    return [row for row in reader]


def harvest(path: pathlib.Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".csv":
        rows = _read_csv(path)
    else:
        rows = _run_nsys_stats(path)

    extracted: List[Dict[str, str]] = []
    for row in rows:
        section = row.get("Section", "")
        metric = row.get("Metric Name") or row.get("Name")
        value = row.get("Metric Value") or row.get("Value")
        if metric and value:
            extracted.append({"section": section, "metric": metric, "value": value})
    return extracted


def process(patterns: Iterable[str]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for pattern in patterns:
        for candidate in pathlib.Path().glob(pattern):
            if not candidate.exists():
                continue
            metrics = harvest(candidate)
            tag = candidate.stem
            for entry in metrics:
                record = {"tag": tag}
                record.update(entry)
                output.append(record)
    return output


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Extract Nsight Systems summary metrics to CSV")
    parser.add_argument("patterns", nargs="+", help="Glob pattern(s) to .nsys-rep or CSV files")
    parser.add_argument(
        "--output",
        default="output/nsys_summary.csv",
        help="Destination CSV file (default: output/nsys_summary.csv)",
    )
    args = parser.parse_args(argv)

    rows = process(args.patterns)
    if not rows:
        print("No Nsight Systems metrics found", file=sys.stderr)
        return 1

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tag", "section", "metric", "value"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
