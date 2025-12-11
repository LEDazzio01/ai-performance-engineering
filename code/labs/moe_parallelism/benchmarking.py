"""Shared benchmarking helpers for the MoE parallelism lab."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

# Try to import plan module - may not be available in all builds
_PLAN_AVAILABLE = False
_PLAN_ERROR: Optional[str] = None

try:
    from .plan import (  # noqa: E402
        DEFAULT_CLUSTER,
        DEFAULT_MODEL,
        get_default_cluster_spec,
        get_default_model_spec,
        ParallelismPlan,
        PlanEvaluator,
        PlanReport,
        format_report,
    )
    _PLAN_AVAILABLE = True
except ModuleNotFoundError as exc:
    _PLAN_ERROR = f"moe_parallelism plan module unavailable: {exc}"
    # Define stubs so the module can import
    DEFAULT_CLUSTER = None
    DEFAULT_MODEL = None
    def get_default_cluster_spec(): return None
    def get_default_model_spec(): return None
    ParallelismPlan = None  # type: ignore
    PlanEvaluator = None  # type: ignore
    PlanReport = None  # type: ignore
    def format_report(*args, **kwargs): return ""


def is_plan_available() -> bool:
    """Check if the plan module is available."""
    return _PLAN_AVAILABLE


def get_plan_error() -> Optional[str]:
    """Get the error message if plan is not available."""
    return _PLAN_ERROR


class _SkipPlanBenchmark(BaseBenchmark):
    """Skip benchmark when plan module is not available."""
    
    def __init__(self) -> None:
        super().__init__()
        self.metrics = torch.randn((1, 1), dtype=torch.float32)
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5)
    
    def benchmark_fn(self) -> None:
        raise RuntimeError(f"SKIPPED: {_PLAN_ERROR or 'moe_parallelism plan module unavailable'}")

    def get_verify_output(self) -> torch.Tensor:
        raise RuntimeError("benchmark_fn() must be called before verification")

    def get_input_signature(self) -> dict:
        return {"type": "skip", "shapes": {"metrics": (1, 1)}}

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)


def get_skip_benchmark() -> BaseBenchmark:
    """Return a skip benchmark for use when plan is unavailable."""
    return _SkipPlanBenchmark()


class PlanBenchmark(BaseBenchmark):
    """CPU-only benchmark that emits a plan analysis report."""

    def __init__(
        self,
        plan: ParallelismPlan,
        cluster: Optional[ClusterSpec] = None,
        model: Optional[ModelSpec] = None,
    ) -> None:
        self._override_device = torch.device("cpu")
        super().__init__()
        resolved_cluster = cluster
        resolved_model = model
        if resolved_cluster is None or resolved_model is None:
            default_cluster = get_default_cluster_spec()
            default_model = get_default_model_spec()
            resolved_cluster = resolved_cluster or default_cluster
            resolved_model = resolved_model or default_model
        self.plan = plan
        self.cluster = resolved_cluster
        self.model = resolved_model
        self.evaluator = PlanEvaluator(resolved_cluster, resolved_model)
        self.report: Optional[PlanReport] = None
        self._summary: Optional[str] = None
        self._config: Optional[BenchmarkConfig] = None
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        return self._override_device

    def setup(self) -> None:
        self.report = None
        self._summary = None

    def benchmark_fn(self) -> None:
        report = self.evaluator.analyze(self.plan)
        self.report = report
        self._summary = format_report(report)
        self._finalize_output(
            [
                float(report.estimated_step_ms),
                float(report.throughput_tokens_per_s),
                float(getattr(report, "memory_per_device_gb", 0.0)),
            ]
        )

    def teardown(self) -> None:
        self.report = None
        self._summary = None
        self.metrics = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        if self._config is None:
            config = BenchmarkConfig(iterations=1, warmup=5)
            config.use_subprocess = False
            self._config = config
        return self._config

    def validate_result(self) -> Optional[str]:
        if self.report is None:
            return "Plan analysis missing"
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        if self.report is None:
            return None
        throughput = max(self.report.throughput_tokens_per_s, 0.0)
        return WorkloadMetadata(
            requests_per_iteration=throughput,
            tokens_per_iteration=throughput,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return MoE parallelism plan metrics."""
        if self.report is None:
            return None
        return {
            "moe.estimated_step_ms": self.report.estimated_step_ms,
            "moe.throughput_tokens_per_s": self.report.throughput_tokens_per_s,
            "moe.memory_per_device_gb": getattr(self.report, "memory_per_device_gb", 0.0),
        }

    def print_summary(self) -> None:
        if self._summary:
            print(self._summary)

    def _finalize_output(self, metric_values: List[float]) -> None:
        expected_shape = (1, len(metric_values))
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.randn(expected_shape, dtype=torch.float32)
        summary_tensor = torch.tensor([metric_values], dtype=torch.float32)
        self.output = (summary_tensor + self.metrics).detach()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        shape = tuple(self.metrics.shape) if self.metrics is not None else (1, 3)
        return {"plan": str(self.plan), "cluster": str(self.cluster), "shapes": {"metrics": shape}}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def run_benchmark(benchmark: PlanBenchmark) -> None:
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    # Re-run the analysis in-process so we can show the report even if the harness
    # executed the benchmark in a worker thread/subprocess.
    report = benchmark.evaluator.analyze(benchmark.plan)
    print(format_report(report))
    timing_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"Synthetic step time (from analysis): {report.estimated_step_ms:.1f} ms")
    print(f"Projected throughput: {report.throughput_tokens_per_s:,.0f} tokens/s")
    print(f"Harness runtime to generate report: {timing_ms:.4f} ms")
