"""Audit baseline/optimized benchmark alignment.

This script statically inspects each baseline/optimized pair discovered via
``common.python.discovery`` and emits a quick report showing whether the pair
shares the same optimizer, NVTX label, and key workload knobs (batch/seq/hidden).

It is intentionally lightweight (no benchmark execution) so we can run it on
any workstation before firing off expensive smoke tests.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.discovery import discover_all_chapters, discover_benchmarks
from common.python.nvtx_helper import canonicalize_nvtx_name

ATTR_PATTERNS = {
    "batch_size": re.compile(r"self\.batch_size\s*=\s*([^\n#]+)"),
    "seq_len": re.compile(r"self\.seq_len\s*=\s*([^\n#]+)"),
    "hidden_dim": re.compile(r"self\.hidden_dim\s*=\s*([^\n#]+)"),
}

OPTIMIZER_PATTERN = re.compile(r"torch\.optim\.([A-Za-z0-9_]+)")
NVTX_PATTERN = re.compile(r"nvtx_range\(\s*['\"]([^'\"]+)['\"]")


@dataclass
class BenchmarkMetadata:
    path: str
    optimizer: str | None
    nvtx: List[str]
    batch_size: str | None
    seq_len: str | None
    hidden_dim: str | None


def extract_metadata(path: Path) -> BenchmarkMetadata:
    text = path.read_text()
    optimizer = None
    match = OPTIMIZER_PATTERN.search(text)
    if match:
        optimizer = match.group(1)

    raw_nvtx = NVTX_PATTERN.findall(text)
    nvtx = [canonicalize_nvtx_name(label) for label in raw_nvtx]

    attrs: Dict[str, str | None] = {key: None for key in ATTR_PATTERNS}
    for key, pattern in ATTR_PATTERNS.items():
        attr_match = pattern.search(text)
        if attr_match:
            attrs[key] = attr_match.group(1).strip()

    return BenchmarkMetadata(
        path=str(path.relative_to(repo_root)),
        optimizer=optimizer,
        nvtx=nvtx,
        batch_size=attrs["batch_size"],
        seq_len=attrs["seq_len"],
        hidden_dim=attrs["hidden_dim"],
    )


def compare_pair(baseline: BenchmarkMetadata, optimized: List[BenchmarkMetadata]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "baseline": asdict(baseline),
        "optimized": [asdict(meta) for meta in optimized],
        "mismatches": [],
    }

    def check(field: str, pretty_name: str) -> None:
        base_value = getattr(baseline, field)
        for meta in optimized:
            opt_value = getattr(meta, field)
            if base_value != opt_value:
                result["mismatches"].append(
                    f"{pretty_name} mismatch: baseline={base_value} vs {meta.path}={opt_value}"
                )

    check("optimizer", "optimizer")
    check("batch_size", "batch_size")
    check("seq_len", "seq_len")
    check("hidden_dim", "hidden_dim")

    baseline_nvtx = baseline.nvtx[0] if baseline.nvtx else None
    for meta in optimized:
        opt_nvtx = meta.nvtx[0] if meta.nvtx else None
        if baseline_nvtx != opt_nvtx:
            result["mismatches"].append(
                f"NVTX label mismatch: baseline={baseline_nvtx} vs {meta.path}={opt_nvtx}"
            )

    result["severity"] = severity_label(len(result["mismatches"]))
    return result


def severity_label(mismatch_count: int) -> str:
    if mismatch_count == 0:
        return "clean"
    if mismatch_count == 1:
        return "low"
    if mismatch_count <= 3:
        return "medium"
    return "high"


def write_csv(report: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "chapter",
        "example",
        "baseline_path",
        "optimized_paths",
        "severity",
        "mismatch_count",
        "mismatches",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report:
            writer.writerow(
                {
                    "chapter": entry["chapter"],
                    "example": entry["example"],
                    "baseline_path": entry["baseline"]["path"],
                    "optimized_paths": ";".join(
                        opt["path"] for opt in entry["optimized"]
                    ),
                    "severity": entry["severity"],
                    "mismatch_count": len(entry["mismatches"]),
                    "mismatches": " | ".join(entry["mismatches"]),
                }
            )


def write_markdown(report: List[Dict[str, object]], path: Path) -> None:
    total = len(report)
    mismatched = sum(1 for entry in report if entry["mismatches"])
    ok = total - mismatched
    buckets = {
        "clean": sum(1 for entry in report if entry["severity"] == "clean"),
        "low": sum(1 for entry in report if entry["severity"] == "low"),
        "medium": sum(1 for entry in report if entry["severity"] == "medium"),
        "high": sum(1 for entry in report if entry["severity"] == "high"),
    }
    lines = [
        "# Benchmark Alignment Report",
        "",
        f"- Total pairs: {total}",
        f"- Clean pairs: {ok}",
        f"- Pairs with mismatches: {mismatched}",
        f"- Severity buckets: clean={buckets['clean']}, low={buckets['low']}, medium={buckets['medium']}, high={buckets['high']}",
        "",
        "| Chapter | Example | Severity | Mismatches |",
        "| --- | --- | --- | --- |",
    ]
    for entry in report:
        mismatches = entry["mismatches"]
        if not mismatches:
            continue
        bullet = "<br>".join(mismatches)
        lines.append(f"| {entry['chapter']} | {entry['example']} | {entry['severity']} | {bullet} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check baseline/optimized alignment")
    parser.add_argument("--json", type=Path, help="Path to write JSON report")
    parser.add_argument("--csv", type=Path, help="Path to write CSV summary")
    parser.add_argument("--markdown", type=Path, help="Path to write Markdown summary")
    parser.add_argument(
        "--chapter",
        action="append",
        default=[],
        help="Filter to specific chapter(s) (e.g., ch13 or 13). Can be repeated.",
    )
    args = parser.parse_args()

    chapter_filters = normalize_chapter_filters(args.chapter)
    chapters = [
        ch for ch in discover_all_chapters(repo_root)
        if not chapter_filters or ch.name in chapter_filters
    ]
    report: List[Dict[str, object]] = []

    for chapter_dir in chapters:
        pairs = discover_benchmarks(chapter_dir)
        if not pairs:
            continue
        for baseline_path, optimized_paths, example_name in pairs:
            baseline_meta = extract_metadata(baseline_path)
            optimized_meta = [extract_metadata(p) for p in optimized_paths]
            comparison = compare_pair(baseline_meta, optimized_meta)
            comparison["chapter"] = chapter_dir.name
            comparison["example"] = example_name
            report.append(comparison)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(report, args.csv)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.markdown)


def normalize_chapter_filters(raw_filters: List[str]) -> Set[str]:
    normalized: Set[str] = set()
    for entry in raw_filters:
        if not entry:
            continue
        tokens = entry.replace(",", " ").split()
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if not token.startswith("ch"):
                token = f"ch{token}"
            normalized.add(token)
    return normalized


if __name__ == "__main__":
    main()
