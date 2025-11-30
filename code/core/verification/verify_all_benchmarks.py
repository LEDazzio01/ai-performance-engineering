"""
Benchmark target resolution + lightweight verification entrypoints.

This module used to host the full verification harness; the logic was moved
into the discovery utilities and benchmark runner.  We keep a thin wrapper
here so existing imports keep working:
  - `chapter_slug` / `resolve_target_chapters` are reused by CLI + runners
  - `run_verification` drives a fast smoke pass (used by `aisp bench verify`)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from core.discovery import (
    chapter_slug as _chapter_slug,
    discover_all_chapters,
    discover_benchmarks,
    get_bench_roots,
    normalize_chapter_token,
)

# Repository root (…/code)
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOTS = get_bench_roots(repo_root=REPO_ROOT)
BENCH_ROOT = BENCH_ROOTS[0]


def chapter_slug(chapter_dir: Path, repo_root: Path = REPO_ROOT, bench_root: Optional[Path] = None) -> str:
    """Return a stable chapter identifier relative to the repo root."""
    return _chapter_slug(chapter_dir, repo_root, bench_root=bench_root or BENCH_ROOT)


def _parse_examples(examples: str) -> List[str]:
    """Split example payloads like 'a,b c' into distinct names."""
    tokens = examples.replace(",", " ").split()
    return [tok.strip() for tok in tokens if tok.strip()]


def resolve_target_chapters(
    targets: Optional[List[str]],
    bench_root: Optional[Path] = None,
) -> Tuple[List[Path], Dict[str, Set[str]]]:
    """
    Translate CLI target tokens into chapter directories + per-chapter filters.

    Args:
        targets: List like ["ch07", "ch07:memory_access"] (None/"all" -> every chapter)

    Returns:
        (chapter_dirs, chapter_filters)
          chapter_dirs: ordered list of chapter paths to run
          chapter_filters: map of chapter slug -> set of example names to include
    """
    chapter_filters: Dict[str, Set[str]] = {}

    roots = [Path(bench_root).resolve()] if bench_root else BENCH_ROOTS
    primary_root = roots[0]

    # Default: run everything
    if not targets or any(str(t).lower() == "all" for t in targets):
        return discover_all_chapters(primary_root, bench_roots=roots), chapter_filters

    chapter_dirs: List[Path] = []
    for raw_target in targets:
        if not raw_target:
            continue

        target = str(raw_target).strip()
        if not target:
            continue

        chapter_token, sep, examples = target.partition(":")
        normalized = normalize_chapter_token(chapter_token, repo_root=REPO_ROOT, bench_root=primary_root)
        chapter_dir = Path(normalized)
        if not chapter_dir.is_absolute():
            chapter_dir = (primary_root / normalized).resolve()
        if not chapter_dir.is_dir():
            raise FileNotFoundError(f"Chapter '{normalized}' not found at {chapter_dir}")

        if chapter_dir not in chapter_dirs:
            chapter_dirs.append(chapter_dir)

        # Collect per-chapter example filters when provided
        if sep:
            slug = chapter_slug(chapter_dir, REPO_ROOT, bench_root=primary_root)
            allowed = {example for _, _, example in discover_benchmarks(chapter_dir)}
            for example in _parse_examples(examples):
                if allowed and example not in allowed:
                    raise ValueError(
                        f"Example '{example}' not found in {slug}. "
                        f"Available: {', '.join(sorted(allowed))}"
                    )
                chapter_filters.setdefault(slug, set()).add(example)

    if not chapter_dirs:
        raise ValueError("No valid chapters resolved from targets.")

    return chapter_dirs, chapter_filters


def run_verification(
    targets: Optional[List[str]] = None,
    bench_root: Optional[Path] = None,
    timeout_seconds: Optional[int] = None,
) -> int:
    """
    Run a quick verification pass (smoke mode, no profiling) over selected targets.

    This mirrors the old verify_all_benchmarks entry point but delegates to the
    modern runner to avoid duplicate logic.
    """
    from core.harness import run_all_benchmarks

    argv = [sys.argv[0], "--format", "json", "--profile", "none", "--smoke-test"]
    if targets:
        argv.append("--targets")
        argv.extend(targets)
    if bench_root:
        argv.extend(["--bench-root", str(bench_root)])
    if timeout_seconds is not None:
        argv.extend(["--suite-timeout", str(timeout_seconds)])

    old_argv = sys.argv
    sys.argv = argv
    try:
        run_all_benchmarks.main()
        return 0
    except SystemExit as exc:  # propagate exit codes without bubbling exceptions
        return int(exc.code or 0)
    finally:
        sys.argv = old_argv
