"""Baseline V1 engine polling loop (stops on an idle step and leaks KV cache)."""

from __future__ import annotations

import sys
from typing import Any, Iterator, List

from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from ch18.v1_engine_loop_common import MockRequestOutput, build_demo_stack


def baseline_engine_loop(
    engine_core: Any, core_client: Any
) -> Iterator[MockRequestOutput]:
    """
    Naive polling loop that assumes an idle step means the engine is done.

    This is a pre-V1 style loop: it stops when EngineCore reports executed=False
    and yields no outputs, which can strand queued work and leave KV pages alive.
    """
    while True:
        outputs, executed = engine_core.step()
        for ro in outputs:
            yield ro

        finished_ids: List[str] = [ro.request_id for ro in outputs if getattr(ro, "finished", False)]
        if finished_ids:
            core_client.report_finished_ids(finished_ids)

        if not executed and not outputs:
            break


def _demo() -> None:
    engine_core, core_client = build_demo_stack()
    outputs = list(baseline_engine_loop(engine_core, core_client))
    summary = {
        "steps": engine_core.calls,
        "tokens": "".join(ro.delta_text for ro in outputs),
        "reported_finished": list(core_client.finished_reported),
        "all_done": core_client.is_all_done(),
    }
    print("Baseline loop demo:", summary)


from typing import Optional


class BaselineV1EngineLoopBenchmark(BaseBenchmark):
    """Benchmark for baseline V1 engine polling loop."""
    
    def __init__(self) -> None:
        super().__init__()
        self.engine_core = None
        self.core_client = None
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def setup(self) -> None:
        """Set up the mock engine stack."""
        self.engine_core, self.core_client = build_demo_stack()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 64),
            accepted_tokens=getattr(self, '_accepted_tokens', 48),
            draft_time_ms=getattr(self, '_draft_ms', 5.0),
            verify_time_ms=getattr(self, '_verify_ms', 10.0),
            num_rounds=getattr(self, '_num_rounds', 8),
        )

    def benchmark_fn(self) -> Optional[dict]:
        """Run the baseline engine loop and measure it."""
        # Reset the engine state for each iteration
        self.engine_core, self.core_client = build_demo_stack()
        outputs = list(baseline_engine_loop(self.engine_core, self.core_client))
        return {
            "steps": self.engine_core.calls,
            "tokens_generated": len(outputs),
        }


def get_benchmark() -> BaseBenchmark:
    return BaselineV1EngineLoopBenchmark()


if __name__ == "__main__":
    _demo()
