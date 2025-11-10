#!/usr/bin/env python3
"""Generate proof-of-benefit dashboards from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class IssueEntry:
    chapter: str
    example: str
    severity: str
    best_speedup: float
    baseline_ms: Optional[float]
    artifact_dir: str
    notes: str
    optimizations: str


def list_result_files(artifacts_dir: Path) -> List[Path]:
    """Return every benchmark_test_results.json underneath artifacts/.

    Supports both legacy layouts that stored results at artifacts/<timestamp>/results/*
    and the newer structure that keeps per-example results at
    artifacts/<timestamp>/<example>/results/*.
    """
    result_files: List[Path] = []
    for timestamp_dir in artifacts_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        # Direct timestamp/results/ paths (legacy)
        direct_result = timestamp_dir / "results" / "benchmark_test_results.json"
        if direct_result.exists():
            result_files.append(direct_result)

        # Nested timestamp/<run>/results/ layouts (current)
        for run_dir in timestamp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            nested_result = run_dir / "results" / "benchmark_test_results.json"
            if nested_result.exists():
                result_files.append(nested_result)

    result_files.sort(key=lambda path: (path.stat().st_mtime, str(path)))
    return result_files


REPO_ROOT = Path(__file__).resolve().parents[2]


def _file_exists(repo_root: Path, chapter_name: str, filename: Optional[str]) -> bool:
    if not filename:
        return False

    path = Path(filename)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        chapter_dir = repo_root / chapter_name
        candidates.append(chapter_dir / path)
        candidates.append(repo_root / path)

    return any(candidate.exists() for candidate in candidates)


def _should_treat_failure_as_skip(benchmark: Dict, notes: str) -> bool:
    """Return True when a reported failure really means the test was skipped."""
    status = (benchmark.get("status") or "").lower()
    if status != "failed":
        return False
    notes_lower = notes.lower()
    if "failed to load baseline" in notes_lower:
        return True
    if notes_lower.startswith("skipped"):
        return True
    if benchmark.get("baseline_time_ms") is None and not benchmark.get("optimizations"):
        return True
    return False


def _derive_best_speedup(benchmark: Dict) -> float:
    """Derive best_speedup from optimization entries when metadata is missing."""
    best_speedup = benchmark.get("best_speedup")
    optimizations = benchmark.get("optimizations") or []
    if (best_speedup is None or best_speedup == 1.0) and optimizations:
        derived_speedups = [
            float(opt.get("speedup") or 0.0)
            for opt in optimizations
            if opt.get("speedup") is not None
        ]
        if derived_speedups:
            derived_best = max(derived_speedups)
            # Preserve sub-1.0 speedups so PoB reflects regressions accurately
            if derived_best or best_speedup is None:
                best_speedup = derived_best
    if best_speedup is None:
        best_speedup = 1.0
    return float(best_speedup)


def classify_entry(
    chapter_name: str,
    benchmark: Dict,
    artifact_dir: str,
    threshold: float,
    repo_root: Path,
) -> Optional[IssueEntry]:
    example = benchmark.get("example", "unknown")
    entry_type = benchmark.get("type", "python")
    example_name = f"{example}_cuda" if entry_type == "cuda" else example
    severity = None
    notes = (benchmark.get("error") or benchmark.get("skip_reason") or "").strip()
    baseline_ms = benchmark.get("baseline_time_ms")
    best_speedup = _derive_best_speedup(benchmark)

    status = (benchmark.get("status") or "").lower()
    if _should_treat_failure_as_skip(benchmark, notes):
        status = "skipped"
    baseline_file = benchmark.get("baseline_file")

    if status != "failed":
        if baseline_file and not _file_exists(repo_root, chapter_name, baseline_file):
            return None

        opt_files = [
            opt.get("file") for opt in benchmark.get("optimizations", [])
            if opt.get("file")
        ]
        if not opt_files:
            return None
        if not any(_file_exists(repo_root, chapter_name, opt_file) for opt_file in opt_files):
            return None
    if status == "failed":
        severity = "fail"
    elif status == "skipped":
        severity = "skipped"
    elif best_speedup < threshold:
        severity = "needs-work"

    if severity is None:
        return None

    opt_names = []
    for opt in benchmark.get("optimizations", []) or []:
        name = opt.get("file") or opt.get("technique")
        if name:
            opt_names.append(name)
    optimizations = ";".join(opt_names)

    return IssueEntry(
        chapter=chapter_name,
        example=example_name,
        severity=severity,
        best_speedup=best_speedup,
        baseline_ms=baseline_ms,
        artifact_dir=artifact_dir,
        notes=notes,
        optimizations=optimizations,
    )


def aggregate_issues(
    result_files: Iterable[Path],
    artifacts_dir: Path,
    threshold: float,
) -> Tuple[List[IssueEntry], Dict[str, Dict[str, int]]]:
    latest_issues: Dict[Tuple[str, str], IssueEntry] = {}
    chapter_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "total": 0,
        "latest_artifact": "",
    })

    for result_path in result_files:
        artifact_dir = str(result_path.parent.parent.relative_to(artifacts_dir))
        data = json.loads(result_path.read_text())
        for chapter in data.get("results", []):
            chapter_name = chapter.get("chapter", "unknown").lower()
            chapter_total_entry = chapter_totals[chapter_name]
            summary = chapter.get("summary", {})
            total_benchmarks = summary.get("total_benchmarks", len(chapter.get("benchmarks", [])))
            if total_benchmarks >= chapter_total_entry["total"]:
                chapter_total_entry["total"] = total_benchmarks
                chapter_total_entry["latest_artifact"] = artifact_dir

            for benchmark in chapter.get("benchmarks", []):
                issue = classify_entry(
                    chapter_name,
                    benchmark,
                    artifact_dir,
                    threshold,
                    REPO_ROOT,
                )
                entry_type = benchmark.get("type", "python")
                baseline_file = benchmark.get("baseline_file")
                example_name = benchmark.get("example", "unknown")
                example_key = f"{example_name}_cuda" if entry_type == "cuda" else example_name
                if not baseline_file:
                    baseline_file = example_key
                key = (chapter_name, baseline_file)
                if issue:
                    latest_issues[key] = issue
                else:
                    latest_issues.pop(key, None)

    # Recompute chapter severity counts based on filtered issues
    chapter_stats = defaultdict(lambda: {
        "total": 0,
        "fail": 0,
        "needs-work": 0,
        "skipped": 0,
        "latest_artifact": "",
    })

    for issue in latest_issues.values():
        stats = chapter_stats[issue.chapter]
        stats[issue.severity] += 1

    # Ensure every chapter observed in artifacts appears in summary
    for chapter_name, totals in chapter_totals.items():
        stats = chapter_stats[chapter_name]  # defaultdict ensures entry
        stats["total"] = totals.get("total", stats["total"])
        stats["latest_artifact"] = totals.get("latest_artifact", stats["latest_artifact"])

    issues = sorted(latest_issues.values(), key=lambda x: (x.chapter, x.example))
    return issues, chapter_stats


def write_csv(issues: List[IssueEntry], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "chapter",
            "example",
            "severity",
            "best_speedup",
            "baseline_ms",
            "artifact_dir",
            "notes",
            "optimizations",
        ])
        for entry in issues:
            notes = entry.notes.replace("\n", " ")
            writer.writerow([
                entry.chapter,
                entry.example,
                entry.severity,
                f"{entry.best_speedup:.2f}",
                f"{entry.baseline_ms:.3f}" if entry.baseline_ms is not None else "",
                entry.artifact_dir,
                notes,
                entry.optimizations,
            ])


def write_markdown(
    chapter_stats: Dict[str, Dict[str, int]],
    output_md: Path,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Proof-of-Benefit Gaps",
        "",
        "| Chapter | Fail | Needs Work | Skipped | Total | Latest Artifact |",
        "|---------|------|------------|---------|-------|-----------------|",
    ]
    for chapter in sorted(chapter_stats):
        stats = chapter_stats[chapter]
        lines.append(
            f"| {chapter.upper()} | "
            f"{stats.get('fail', 0)} | "
            f"{stats.get('needs-work', 0)} | "
            f"{stats.get('skipped', 0)} | "
            f"{stats.get('total', 0)} | "
            f"{stats.get('latest_artifact', '')} |"
        )

    lines.append("")
    lines.append("Generated automatically from latest artifacts.")
    output_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate proof-of-benefit reports.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing benchmark artifacts (default: artifacts/)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/proof_of_benefit_issues.csv"),
        help="Path to write CSV issue list",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/proof_of_benefit_issues.md"),
        help="Path to write Markdown summary",
    )
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        default=1.05,
        help="Minimum speedup required to consider optimized result healthy (default: 1.05x)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_files = list_result_files(args.artifacts_dir)
    if not result_files:
        raise SystemExit(f"No benchmark results found under {args.artifacts_dir}")

    issues, chapter_stats = aggregate_issues(
        result_files,
        args.artifacts_dir,
        args.speedup_threshold,
    )
    write_csv(issues, args.output_csv)
    write_markdown(chapter_stats, args.output_md)


if __name__ == "__main__":
    main()
