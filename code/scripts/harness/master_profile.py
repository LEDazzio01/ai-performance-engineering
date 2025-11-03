#!/usr/bin/env python3
"""Entry point for the master profiling workflow (wrapper around profile_harness)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HARNESS = SCRIPT_DIR / "profile_harness.py"
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)


def _dedupe(items: Iterable[str]) -> List[str]:
    ordered = OrderedDict.fromkeys(items)
    return list(ordered.keys())


def _resolve_target(target: str) -> Path:
    path = Path(target)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _match_example(target: Path) -> Optional[str]:
    from example_registry import EXAMPLES

    resolved = target
    candidates: List[tuple[str, Path]] = []
    for example in EXAMPLES:
        run_path = (REPO_ROOT / example.path).resolve()
        candidates.append((example.name, run_path))
        if example.source is not None:
            source_path = (REPO_ROOT / example.source).resolve()
            candidates.append((example.name, source_path))
    for name, path in candidates:
        if resolved == path:
            return name
    return None


def _merge_examples(arguments: List[str], extra_examples: List[str]) -> List[str]:
    if not extra_examples:
        return list(arguments)

    args = list(arguments)
    try:
        idx = args.index("--examples")
    except ValueError:
        return args + ["--examples", *_dedupe(extra_examples)]

    args.pop(idx)
    existing: List[str] = []
    while idx < len(args) and not args[idx].startswith("--"):
        existing.append(args.pop(idx))

    combined = _dedupe(existing + extra_examples)
    if combined:
        args[idx:idx] = ["--examples", *combined]
    return args


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    raw_args = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--list", action="store_true")
    namespace, _ = parser.parse_known_args(raw_args)
    return namespace, raw_args


def _split_targets(
    raw_args: List[str],
    candidate_targets: List[str],
    *,
    example_lookup: Dict[str, object],
) -> tuple[List[str], List[str]]:
    """Separate positional targets from harness arguments.

    Tokens that resolve to registered examples or known paths become
    resolved example names; everything else is forwarded to the harness.
    """

    forward_args: List[str] = []
    resolved: List[str] = []
    pending = list(candidate_targets)

    # Option classification mirrors the harness interface so we can recognise
    # when a "target" token is really an argument value.
    multi_value_options = {"--examples", "--tags", "--profile", "--profile-mode"}
    single_value_options = {"--output-root", "--max-examples"}
    flag_options = {"--dry-run", "--skip-existing", "--force-build", "--skip-smoke", "--list", "--help", "-h"}

    active_option: Optional[str] = None
    single_value_budget = 0

    for index, token in enumerate(raw_args):
        if pending and token == pending[0]:
            treat_as_value = False
            if active_option in multi_value_options:
                treat_as_value = True
            elif single_value_budget > 0:
                treat_as_value = True
                single_value_budget -= 1

            if treat_as_value:
                forward_args.append(token)
                pending.pop(0)
                continue

            pending.pop(0)

            if token in example_lookup:
                resolved.append(token)
                active_option = None
                single_value_budget = 0
                continue

            resolved_path = _resolve_target(token)
            if not resolved_path.exists():
                raise SystemExit(f"Unable to map target '{token}' to a registered example")

            match = _match_example(resolved_path)
            if match is None:
                raise SystemExit(f"Unable to map target '{token}' to a registered example")

            resolved.append(match)
            active_option = None
            single_value_budget = 0
        else:
            forward_args.append(token)

            if token in flag_options:
                active_option = None
                single_value_budget = 0
            elif token in multi_value_options:
                active_option = token
            elif token in single_value_options:
                active_option = token
                single_value_budget = 1
            elif token.startswith("-"):
                active_option = token
                single_value_budget = 0
            elif active_option not in multi_value_options:
                active_option = None
                single_value_budget = 0

    forward_args.extend(pending)
    return forward_args, resolved


def main() -> int:
    args, raw_args = parse_args()

    from example_registry import EXAMPLE_BY_NAME

    try:
        forward_args, resolved_examples = _split_targets(
            raw_args,
            args.targets,
            example_lookup=EXAMPLE_BY_NAME,
        )
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1

    harness_args = _merge_examples(forward_args, resolved_examples)
    command = [PYTHON_BIN, str(HARNESS), *harness_args]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
