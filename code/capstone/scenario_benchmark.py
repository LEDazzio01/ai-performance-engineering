"""Full-stack capstone scenarios composed from chapter benchmarks.

This module provides a reusable Benchmark implementation that stitches
baseline/optimized examples from every chapter into a single narrative run.
Each scenario defines a list of chapter phases (baseline + optimized), and the
CapstoneScenarioBenchmark executes them sequentially using the shared
BenchmarkHarness.  Every phase runs in smoke-test scale by default so the
entire storyline can execute on a single developer workstation while still
exercising the real benchmark code paths.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from common.python.logger import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fallback when logger unavailable
    import logging

    logger = logging.getLogger(__name__)

from common.python.benchmark_harness import (  # noqa: E402  (after sys.path)
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


SMOKE_FLAG = "BENCHMARK_SMOKE_TEST"
DEFAULT_PHASE_ITERATIONS = 16
DEFAULT_PHASE_TIMEOUT_S = 120


class ScenarioVariant(Enum):
    """Whether we are running the baseline or optimized path."""

    BASELINE = "baseline"
    OPTIMIZED = "optimized"


@dataclass(frozen=True)
class PhaseSpec:
    """Description of a single chapter benchmark phase."""

    name: str
    module: str
    chapter: str
    description: str
    max_iterations: int = DEFAULT_PHASE_ITERATIONS


@dataclass(frozen=True)
class ScenarioSpec:
    """Capstone scenario definition with baseline/optimized phases."""

    key: str
    title: str
    summary: str
    chapters: Tuple[str, ...]
    baseline_phases: Tuple[PhaseSpec, ...]
    optimized_phases: Tuple[PhaseSpec, ...]


class PhaseRunStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"


@dataclass
class PhaseRunResult:
    spec: PhaseSpec
    status: PhaseRunStatus
    message: Optional[str] = None
    timing_ms: Optional[float] = None


def _phase(
    name: str,
    module: str,
    chapter: str,
    description: str,
    *,
    max_iterations: int = DEFAULT_PHASE_ITERATIONS,
) -> PhaseSpec:
    return PhaseSpec(
        name=name,
        module=module,
        chapter=chapter,
        description=description,
        max_iterations=max(1, max_iterations),
    )


SCENARIO_SPECS: Dict[str, ScenarioSpec] = {
    "01_system_foundations": ScenarioSpec(
        key="01_system_foundations",
        title="System Foundations",
        summary=(
            "Ch1–Ch3 instrumentation, hardware introspection, and NUMA tuning "
            "before touching kernels."
        ),
        chapters=("ch1", "ch2", "ch3"),
        baseline_phases=(
            _phase(
                "ch1_baseline_performance",
                "ch1.baseline_performance",
                "ch1",
                "Raw throughput measurement without NVTX ranges",
            ),
            _phase(
                "ch2_baseline_cutlass",
                "ch2.baseline_cutlass",
                "ch2",
                "Baseline hardware-aware CUTLASS GEMM",
            ),
            _phase(
                "ch3_baseline_numa",
                "ch3.baseline_numa_unaware",
                "ch3",
                "Untuned NUMA / IRQ placement",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch1_optimized_performance",
                "ch1.optimized_performance",
                "ch1",
                "NVTX + profiler-enabled performance pass",
            ),
            _phase(
                "ch2_optimized_cutlass",
                "ch2.optimized_cutlass",
                "ch2",
                "Optimized CUTLASS GEMM with architecture-specific tuning",
            ),
            _phase(
                "ch3_optimized_numa",
                "ch3.optimized_numa_unaware",
                "ch3",
                "NUMA-aware placement + IRQ pinning",
            ),
        ),
    ),
    "02_cluster_parallelism": ScenarioSpec(
        key="02_cluster_parallelism",
        title="Cluster Parallelism",
        summary="Ch4 tensor/data parallelism and NVLink validation",
        chapters=("ch4",),
        baseline_phases=(
            _phase(
                "ch4_baseline_dataparallel",
                "ch4.baseline_dataparallel",
                "ch4",
                "Single-host data parallel baseline",
            ),
            _phase(
                "ch4_baseline_tensor_parallel",
                "ch4.baseline_tensor_parallelism",
                "ch4",
                "Tensor parallel math without overlap",
            ),
            _phase(
                "ch4_baseline_nvlink",
                "ch4.baseline_nvlink",
                "ch4",
                "NVLink link-check baseline",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch4_optimized_dataparallel",
                "ch4.optimized_dataparallel",
                "ch4",
                "Data parallel run with overlap and tuned buckets",
            ),
            _phase(
                "ch4_optimized_tensor_parallel",
                "ch4.optimized_tensor_parallelism",
                "ch4",
                "Tensor parallel pipeline with symmetric memory",
            ),
            _phase(
                "ch4_optimized_nvlink",
                "ch4.optimized_nvlink",
                "ch4",
                "NVLink saturated warm-path",
            ),
        ),
    ),
    "03_io_pipeline": ScenarioSpec(
        key="03_io_pipeline",
        title="Storage & IO Pipeline",
        summary="Ch5 GPUDirect Storage and loader optimizations",
        chapters=("ch5",),
        baseline_phases=(
            _phase(
                "ch5_baseline_storage",
                "ch5.baseline_storage_cpu",
                "ch5",
                "CPU-bound staging path",
            ),
            _phase(
                "ch5_baseline_vectorization",
                "ch5.baseline_vectorization",
                "ch5",
                "Naive IO vectorization",
            ),
            _phase(
                "ch5_baseline_roofline",
                "ch5.baseline_roofline",
                "ch5",
                "Baseline IO roofline plot",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch5_optimized_storage",
                "ch5.optimized_storage_cpu",
                "ch5",
                "Pinned + async storage staging",
            ),
            _phase(
                "ch5_optimized_vectorization",
                "ch5.optimized_vectorization",
                "ch5",
                "Vectorized IO + overlap",
            ),
            _phase(
                "ch5_optimized_roofline",
                "ch5.optimized_roofline",
                "ch5",
                "Optimized IO roofline pipeline",
            ),
        ),
    ),
    "04_kernel_optimization": ScenarioSpec(
        key="04_kernel_optimization",
        title="Kernel Optimization",
        summary="Ch6–Ch10 progression from ILP to clustered TMA",
        chapters=("ch6", "ch7", "ch8", "ch9", "ch10"),
        baseline_phases=(
            _phase(
                "ch6_baseline_warp_divergence",
                "ch6.baseline_warp_divergence_ilp",
                "ch6",
                "Scalar ILP with warp divergence",
            ),
            _phase(
                "ch7_baseline_matmul",
                "ch7.baseline_matmul_cuda",
                "ch7",
                "Naive memory-bound matmul",
            ),
            _phase(
                "ch8_baseline_occupancy",
                "ch8.baseline_occupancy",
                "ch8",
                "Launch-bounds limited occupancy",
            ),
            _phase(
                "ch9_baseline_micro_tiling",
                "ch9.baseline_micro_tiling_matmul",
                "ch9",
                "Baseline micro-tiling matmul",
            ),
            _phase(
                "ch10_baseline_cluster_group",
                "ch10.baseline_cluster_group",
                "ch10",
                "Thread-block cluster warm-up",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch6_optimized_warp_divergence",
                "ch6.optimized_warp_divergence_ilp",
                "ch6",
                "Vectorized ILP with warp specialization",
            ),
            _phase(
                "ch7_optimized_matmul",
                "ch7.optimized_matmul_cuda",
                "ch7",
                "Tiled + coalesced matmul",
            ),
            _phase(
                "ch8_optimized_occupancy",
                "ch8.optimized_occupancy",
                "ch8",
                "Occupancy-tuned launch bounds",
            ),
            _phase(
                "ch9_optimized_micro_tiling",
                "ch9.optimized_micro_tiling_matmul",
                "ch9",
                "Optimized micro-tiling with fused epilogue",
            ),
            _phase(
                "ch10_optimized_cluster_group",
                "ch10.optimized_cluster_group",
                "ch10",
                "Clustered matmul with DSMEM",
            ),
        ),
    ),
    "05_streams_and_graphs": ScenarioSpec(
        key="05_streams_and_graphs",
        title="Streams & CUDA Graphs",
        summary="Ch11–Ch12 multi-stream overlap and graph capture",
        chapters=("ch11", "ch12"),
        baseline_phases=(
            _phase(
                "ch11_baseline_streams",
                "ch11.baseline_streams",
                "ch11",
                "Single-stream kernels",
            ),
            _phase(
                "ch11_baseline_stream_ordered",
                "ch11.baseline_stream_ordered",
                "ch11",
                "Stream-ordered allocator baseline",
            ),
            _phase(
                "ch12_baseline_cuda_graphs",
                "ch12.baseline_cuda_graphs",
                "ch12",
                "Naive CUDA Graph capture",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch11_optimized_streams",
                "ch11.optimized_streams",
                "ch11",
                "Warp-specialized multistream pipeline",
            ),
            _phase(
                "ch11_optimized_stream_ordered",
                "ch11.optimized_stream_ordered",
                "ch11",
                "Stream-ordered pipeline with overlap",
            ),
            _phase(
                "ch12_optimized_cuda_graphs",
                "ch12.optimized_cuda_graphs",
                "ch12",
                "Graph replay with device-side launches",
            ),
        ),
    ),
    "06_compiler_stack": ScenarioSpec(
        key="06_compiler_stack",
        title="PyTorch + Compiler Stack",
        summary="Ch13–Ch14 torch.compile, Triton, and quantization",
        chapters=("ch13", "ch14"),
        baseline_phases=(
            _phase(
                "ch13_baseline_matmul",
                "ch13.baseline_matmul_pytorch",
                "ch13",
                "Eager PyTorch matmul",
            ),
            _phase(
                "ch13_baseline_quantization",
                "ch13.baseline_quantization",
                "ch13",
                "Reference quantization path",
            ),
            _phase(
                "ch14_baseline_model_eager",
                "ch14.baseline_model_eager",
                "ch14",
                "Uncompiled transformer",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch13_optimized_matmul",
                "ch13.optimized_matmul_pytorch",
                "ch13",
                "torch.compile accelerated matmul",
            ),
            _phase(
                "ch13_optimized_quantization",
                "ch13.optimized_quantization",
                "ch13",
                "Quantization + custom allocator",
            ),
            _phase(
                "ch14_optimized_model_eager",
                "ch14.optimized_model_eager",
                "ch14",
                "Compiled transformer with Triton kernels",
            ),
        ),
    ),
    "07_inference_attention": ScenarioSpec(
        key="07_inference_attention",
        title="Inference & Attention Systems",
        summary="Ch15–Ch18 decoding, routing, and attention optimizations",
        chapters=("ch15", "ch16", "ch17", "ch18"),
        baseline_phases=(
            _phase(
                "ch15_baseline_inference",
                "ch15.baseline_inference_monolithic",
                "ch15",
                "Single-stage decoder inference",
            ),
            _phase(
                "ch16_baseline_moe",
                "ch16.baseline_moe",
                "ch16",
                "Baseline MoE routing",
            ),
            _phase(
                "ch17_baseline_routing",
                "ch17.baseline_routing_static",
                "ch17",
                "Static routing without adaptivity",
            ),
            _phase(
                "ch18_baseline_attention",
                "ch18.baseline_attention",
                "ch18",
                "Classic attention kernel",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch15_optimized_inference",
                "ch15.optimized_inference_monolithic",
                "ch15",
                "Disaggregated inference path",
            ),
            _phase(
                "ch16_optimized_moe",
                "ch16.optimized_moe",
                "ch16",
                "Optimized MoE with overlap",
            ),
            _phase(
                "ch17_optimized_routing",
                "ch17.optimized_routing_static",
                "ch17",
                "Adaptive routing + early exit",
            ),
            _phase(
                "ch18_optimized_attention",
                "ch18.optimized_attention",
                "ch18",
                "Flash/Flex attention fast path",
            ),
        ),
    ),
    "08_low_precision": ScenarioSpec(
        key="08_low_precision",
        title="Low Precision & Memory",
        summary="Ch19 FP4/FP8 workflows with advanced allocators",
        chapters=("ch19",),
        baseline_phases=(
            _phase(
                "ch19_baseline_double_buffering",
                "ch19.baseline_memory_double_buffering",
                "ch19",
                "Baseline memory double buffering",
            ),
            _phase(
                "ch19_baseline_vectorization",
                "ch19.baseline_vectorization_memory",
                "ch19",
                "Vectorization without quantization",
            ),
            _phase(
                "ch19_baseline_cutlass_memory",
                "ch19.baseline_cutlass_memory",
                "ch19",
                "Baseline CUTLASS memory path",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch19_optimized_double_buffering",
                "ch19.optimized_memory_double_buffering",
                "ch19",
                "Optimized overlapping double buffering",
            ),
            _phase(
                "ch19_optimized_vectorization",
                "ch19.optimized_vectorization_memory",
                "ch19",
                "Vectorized FP8 kernels",
            ),
            _phase(
                "ch19_optimized_cutlass_memory",
                "ch19.optimized_cutlass_memory",
                "ch19",
                "CUTLASS pipeline with precision switching",
            ),
        ),
    ),
    "09_end_to_end": ScenarioSpec(
        key="09_end_to_end",
        title="End-to-End Production",
        summary="Ch20 orchestration of training, pipeline, and KV cache",
        chapters=("ch20",),
        baseline_phases=(
            _phase(
                "ch20_baseline_training",
                "ch20.baseline_training_single",
                "ch20",
                "Single-stage training baseline",
            ),
            _phase(
                "ch20_baseline_pipeline",
                "ch20.baseline_pipeline_sequential",
                "ch20",
                "Sequential pipeline",
            ),
            _phase(
                "ch20_baseline_bandwidth",
                "ch20.baseline_end_to_end_bandwidth",
                "ch20",
                "Unoptimized bandwidth validation",
            ),
            _phase(
                "ch20_baseline_kv_cache",
                "ch20.baseline_integrated_kv_cache",
                "ch20",
                "Untuned KV cache integration",
            ),
        ),
        optimized_phases=(
            _phase(
                "ch20_optimized_training",
                "ch20.optimized_training_single",
                "ch20",
                "Graph-captured training step",
            ),
            _phase(
                "ch20_optimized_pipeline",
                "ch20.optimized_pipeline_sequential",
                "ch20",
                "Overlapped sequential pipeline",
            ),
            _phase(
                "ch20_optimized_bandwidth",
                "ch20.optimized_end_to_end_bandwidth",
                "ch20",
                "Optimized system bandwidth validation",
            ),
            _phase(
                "ch20_optimized_kv_cache",
                "ch20.optimized_integrated_kv_cache",
                "ch20",
                "Production KV cache orchestration",
            ),
        ),
    ),
}


def list_available_scenarios() -> Sequence[str]:
    return sorted(SCENARIO_SPECS.keys())


def get_scenario_spec(key: str) -> ScenarioSpec:
    try:
        return SCENARIO_SPECS[key]
    except KeyError as exc:  # pragma: no cover - guarded by tests
        raise KeyError(f"Unknown capstone scenario '{key}'") from exc


def _load_benchmark(module_path: str):
    module = importlib.import_module(module_path)
    if not hasattr(module, "get_benchmark"):
        raise RuntimeError(f"Module {module_path} is missing get_benchmark()")
    benchmark = module.get_benchmark()
    return benchmark


@contextmanager
def _smoke_mode(enabled: bool = True):
    previous = os.environ.get(SMOKE_FLAG)
    if enabled:
        os.environ[SMOKE_FLAG] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SMOKE_FLAG, None)
        else:
            os.environ[SMOKE_FLAG] = previous


class CapstoneScenarioBenchmark(BaseBenchmark):
    """Benchmark wrapper that executes a sequence of chapter phases."""

    def __init__(
        self,
        scenario_key: str,
        variant: ScenarioVariant,
        *,
        max_phase_iterations: int = DEFAULT_PHASE_ITERATIONS,
        phase_iteration_floor: int = 12,
    ) -> None:
        super().__init__()
        self.scenario = get_scenario_spec(scenario_key)
        self.variant = variant
        self.max_phase_iterations = max(1, max_phase_iterations)
        self._phase_iteration_floor = max(1, phase_iteration_floor)
        self._phase_specs = (
            self.scenario.baseline_phases
            if variant == ScenarioVariant.BASELINE
            else self.scenario.optimized_phases
        )
        self._phase_results: List[PhaseRunResult] = []
        self._scenario_total_ms: float = 0.0
        self.register_workload_metadata(
            custom_units_per_iteration=float(len(self._phase_specs)),
            custom_unit_name="phases",
        )

    def setup(self) -> None:
        self._phase_results.clear()
        self._scenario_total_ms = 0.0
        # Preload modules so heavyweight CUDA extensions build during setup,
        # keeping proof-of-benefit measurements focused on steady-state timing.
        for spec in self._phase_specs:
            try:
                _load_benchmark(spec.module)
            except RuntimeError:
                # Defer detailed error handling to the timed phase execution.
                pass

    def benchmark_fn(self) -> None:
        with _smoke_mode(True):
            for spec in self._phase_specs:
                with self._nvtx_range(f"capstone_{spec.name}"):
                    self._run_phase(spec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._scenario_total_ms = sum(
            result.timing_ms
            for result in self._phase_results
            if result.status == PhaseRunStatus.SUCCESS and result.timing_ms is not None
        )

    def _run_phase(self, spec: PhaseSpec) -> None:
        logger.info(
            "[capstone] Starting %s (%s/%s)",
            spec.name,
            spec.chapter,
            self.variant.value,
        )
        benchmark = _load_benchmark(spec.module)
        config = self._prepare_config(benchmark, spec)
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        try:
            result = harness.benchmark(benchmark)
        except RuntimeError as err:
            message = str(err)
            if "SKIPPED" in message or "requires" in message:
                logger.warning("[capstone] Phase %s skipped: %s", spec.name, message)
                self._phase_results.append(
                    PhaseRunResult(spec=spec, status=PhaseRunStatus.SKIPPED, message=message)
                )
                return
            raise

        timing_ms = result.timing.mean_ms if result.timing else None
        self._phase_results.append(
            PhaseRunResult(
                spec=spec,
                status=PhaseRunStatus.SUCCESS,
                timing_ms=timing_ms,
            )
        )
        if timing_ms is not None:
            logger.info(
                "[capstone] Completed %s in %.3f ms", spec.name, timing_ms
            )
        else:
            logger.info("[capstone] Completed %s", spec.name)

    def _prepare_config(self, benchmark, spec: PhaseSpec) -> BenchmarkConfig:
        original = benchmark.get_config()
        if original is None:
            config = BenchmarkConfig()
        else:
            config = copy.deepcopy(original)

        max_allowed = max(1, min(spec.max_iterations, self.max_phase_iterations))
        config.iterations = min(
            max_allowed,
            max(config.iterations, self._phase_iteration_floor),
        )
        if config.warmup is None:
            config.warmup = 0
        else:
            config.warmup = min(config.warmup, 1)
        config.enable_profiling = False
        config.enable_nsys = False
        config.enable_ncu = False
        config.enable_nvtx = True if self.variant == ScenarioVariant.OPTIMIZED else False
        config.min_run_time_ms = 0.0
        config.use_subprocess = False
        measurement_timeout = (
            config.measurement_timeout_seconds
            if config.measurement_timeout_seconds is not None
            else DEFAULT_PHASE_TIMEOUT_S
        )
        config.measurement_timeout_seconds = max(
            measurement_timeout, DEFAULT_PHASE_TIMEOUT_S
        )
        if config.setup_timeout_seconds is not None:
            config.setup_timeout_seconds = max(
                config.setup_timeout_seconds, DEFAULT_PHASE_TIMEOUT_S // 2
            )
        return config

    def teardown(self) -> None:
        self._log_summary()
        super().teardown()

    def _log_summary(self) -> None:
        if not self._phase_results:
            logger.warning("[capstone] No phases executed for %s", self.scenario.key)
            return
        lines = [
            f"[capstone] Scenario '{self.scenario.title}' ({self.variant.value}) summary:"
        ]
        for result in self._phase_results:
            status = result.status.value
            timing = f"{result.timing_ms:.3f} ms" if result.timing_ms is not None else "n/a"
            lines.append(
                f"  - {result.spec.name}: {status} ({timing})"
                + (f" – {result.message}" if result.message else "")
            )
        if self._scenario_total_ms > 0:
            lines.append(
                f"  • Aggregated phase time: {self._scenario_total_ms:.3f} ms"
            )
        logger.info("\n".join(lines))

    def get_config(self) -> BenchmarkConfig:
        scenario_timeout = max(len(self._phase_specs), 1) * DEFAULT_PHASE_TIMEOUT_S
        return BenchmarkConfig(
            iterations=1,
            warmup=1,
            enable_profiling=False,
            enable_nvtx=self.variant == ScenarioVariant.OPTIMIZED,
            min_run_time_ms=0.0,
            measurement_timeout_seconds=scenario_timeout,
            use_subprocess=False,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self._scenario_total_ms <= 0:
            return None
        successful_phases = sum(
            1 for result in self._phase_results if result.status == PhaseRunStatus.SUCCESS
        )
        return {
            "scenario_total_phase_ms": self._scenario_total_ms,
            "scenario_successful_phases": float(successful_phases),
        }

    def validate_result(self) -> Optional[str]:
        if not self._phase_results:
            return "Scenario did not execute any phases"
        succeeded = [r for r in self._phase_results if r.status == PhaseRunStatus.SUCCESS]
        if not succeeded:
            return "All scenario phases were skipped; see logs for details"
        return None


__all__ = [
    "CapstoneScenarioBenchmark",
    "ScenarioVariant",
    "ScenarioSpec",
    "PhaseSpec",
    "list_available_scenarios",
]
