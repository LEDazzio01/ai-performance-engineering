"""Cheap eval stack runner (tool).

This is a lab tool (NOT a comparable baseline/optimized benchmark target).
It runs the "full" cheap-eval stack and emits artifacts under:
`artifacts/dynamic_router/cheap_eval/full_<timestamp>/`.
"""

from __future__ import annotations

import json
import sys

from labs.dynamic_router.eval_stack import EvalConfig, run_eval_stack


def main(argv: list[str]) -> int:
    cfg = EvalConfig.from_flags(argv, seed=42)
    summary = run_eval_stack(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
