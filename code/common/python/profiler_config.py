"""Centralized profiler configuration.

Single source of truth for profiler command generation and metric sets.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from common.python.nvtx_helper import canonicalize_nvtx_name


# Metric sets for different profiling scenarios
ROOFLINE_METRICS = [
    # Kernel timing
    "gpu__time_duration.avg",
    # Compute throughput (SM)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory throughput - DRAM (HBM)
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory throughput - L2 cache
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    # Occupancy - active warps
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

DEEP_DIVE_METRICS = [
    # All roofline metrics
    *ROOFLINE_METRICS,
    # Memory throughput counters (sectors + bytes) for bandwidth + reuse analysis
    "dram__sectors_read.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    # Instruction mix (separated adds/muls/ffma + dtype split)
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
    # Tensor Core utilization (if applicable)
    "sm__inst_executed_pipe_tensor.sum",
    # Occupancy proxy already in roofline (sm__warps_active...), keep explicit time
    "gpu__time_duration.sum",
    # Shared memory bank conflicts
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
]

MINIMAL_METRICS = [
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
]

# =============================================================================
# Chapter-Specific Metric Sets
# =============================================================================
# Use these for targeted profiling of specific optimization techniques.
# Each set focuses on the metrics most relevant to that chapter's topic.

# Ch6: Kernel Fundamentals - Bank conflicts, warp divergence, occupancy
CH6_KERNEL_METRICS = [
    *ROOFLINE_METRICS,
    # Bank conflicts (critical for Ch6)
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Warp divergence
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    # Occupancy details
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_per_block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
]

# Ch7: Memory Access - Coalescing, vectorization, cache behavior
CH7_MEMORY_METRICS = [
    *ROOFLINE_METRICS,
    # Coalescing efficiency
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
    # L1 cache behavior
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit_rate.pct",
    # L2 cache behavior
    "lts__t_sectors_op_read_hit_rate.pct",
    "lts__t_sectors_op_write_hit_rate.pct",
    # Transaction counts
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
]

# Ch8: Optimization Techniques - Stall analysis, ILP, register pressure
CH8_OPTIMIZATION_METRICS = [
    *DEEP_DIVE_METRICS,
    # Stall reasons (THE KEY to understanding "why faster")
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    # ILP metrics
    "smsp__inst_executed_per_warp.ratio",
    # Register pressure
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
]

# Ch9: Compute-Bound - Tensor Core utilization, arithmetic intensity
CH9_COMPUTE_METRICS = [
    *ROOFLINE_METRICS,
    # Tensor Core utilization
    "sm__inst_executed_pipe_tensor.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # FMA operations
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dmma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    # Instruction throughput
    "sm__inst_executed.avg.per_cycle_elapsed",
]

# Ch10: Pipelining - Pipeline utilization, async copy
CH10_PIPELINE_METRICS = [
    *ROOFLINE_METRICS,
    # Async copy metrics
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
    # Pipeline stalls
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    # SM utilization
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__ctas_active.avg.pct_of_peak_sustained_active",
]

# Ch11: Streams - Overlap metrics (use nsys primarily)
CH11_STREAM_METRICS = [
    *MINIMAL_METRICS,
    # Basic timing for overlap analysis
    "gpu__time_duration.sum",
    "gpu__time_duration.avg",
]

# Ch12: CUDA Graphs - Launch overhead metrics
CH12_GRAPH_METRICS = [
    *MINIMAL_METRICS,
    # Launch timing
    "gpu__time_duration.avg",
    "gpu__time_duration.sum",
]

# Ch13-14: PyTorch/Triton - General performance
CH13_PYTORCH_METRICS = [
    *ROOFLINE_METRICS,
    # Tensor Core for FP8/FP16
    "sm__inst_executed_pipe_tensor.sum",
    # Memory efficiency
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
]

# Mapping from chapter number to metric set
CHAPTER_METRICS = {
    6: CH6_KERNEL_METRICS,
    7: CH7_MEMORY_METRICS,
    8: CH8_OPTIMIZATION_METRICS,
    9: CH9_COMPUTE_METRICS,
    10: CH10_PIPELINE_METRICS,
    11: CH11_STREAM_METRICS,
    12: CH12_GRAPH_METRICS,
    13: CH13_PYTORCH_METRICS,
    14: CH13_PYTORCH_METRICS,  # Same as Ch13
}


def get_chapter_metrics(chapter: int) -> List[str]:
    """Get the recommended ncu metrics for a specific chapter.
    
    Args:
        chapter: Chapter number (1-20)
    
    Returns:
        List of ncu metric names appropriate for that chapter's topic
    """
    return CHAPTER_METRICS.get(chapter, ROOFLINE_METRICS)


NCU_SET_BY_METRIC = {
    "deep_dive": "full",
    "roofline": "roofline",
    "minimal": "speed-of-light",
}


@dataclass
class ProfilerConfig:
    """Configuration for profiler execution."""
    
    metric_set: str = "minimal"  # "roofline", "deep_dive", or "minimal"
    preset: str = "minimal"  # "minimal" (low overhead) or "deep_dive"
    nvtx_includes: Optional[List[str]] = None
    nsys_trace_types: str = "cuda,nvtx,osrt,cublas,cudnn"
    nsys_trace_types_minimal: str = "cuda,nvtx,osrt"
    nsys_sample_cpu: bool = True
    nsys_python_sampling: bool = True
    nsys_python_sampling_frequency: int = 1000
    nsys_cudabacktrace: bool = True
    nsys_stats: bool = True
    nsys_backtrace: str = "none"
    ncu_replay_mode: str = "kernel"  # "kernel" or "application"
    ncu_sampling_interval: int = 75000
    
    def get_nsys_command(
        self, 
        output_path: str, 
        python_script: str,
        python_executable: Optional[str] = None,
        nvtx_includes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generate nsys command with configured flags.
        
        Args:
            output_path: Path for nsys output (without .nsys-rep extension)
            python_script: Path to Python script to profile
            python_executable: Python executable to use (defaults to sys.executable)
            nvtx_includes: Optional NVTX range filters to include
        
        Returns:
            List of command arguments for subprocess.run
        """
        import sys
        
        preset = (self.preset or "minimal").lower()
        nvtx_filters = list(dict.fromkeys(nvtx_includes or self.nvtx_includes or []))
        trace_types = (
            self.nsys_trace_types_minimal
            if preset == "minimal"
            else self.nsys_trace_types
        )

        cmd = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "-o",
            output_path,
            "-t", trace_types,
        ]
        
        if self.nsys_sample_cpu:
            cmd.extend(["-s", "cpu"])
        
        if preset != "minimal" and self.nsys_python_sampling:
            cmd.extend([
                "--python-sampling=true",
                f"--python-sampling-frequency={self.nsys_python_sampling_frequency}",
            ])
        
        if preset != "minimal" and self.nsys_cudabacktrace:
            cmd.append("--cudabacktrace=true")
        
        if preset == "minimal":
            cmd.extend(["--backtrace", self.nsys_backtrace])
        if self.nsys_stats:
            cmd.append("--stats=true")

        for tag in nvtx_filters:
            cmd.extend(["--nvtx-include", tag])
        
        cmd.append(python_executable or sys.executable)
        cmd.append(python_script)
        
        return cmd
    
    def get_ncu_command(
        self,
        output_path: str,
        python_script: str,
        python_executable: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        nvtx_includes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generate ncu command with configured metrics.
        
        Args:
            output_path: Path for ncu output (without .ncu-rep extension)
            python_script: Path to Python script to profile
            python_executable: Python executable to use (defaults to sys.executable)
            metrics: Optional list of metrics (defaults to metric_set)
            nvtx_includes: Optional NVTX range filters to include
        
        Returns:
            List of command arguments for subprocess.run
        """
        import sys
        
        preset = (self.preset or "minimal").lower()
        metric_set = self.metric_set or "deep_dive"
        if metrics is None:
            if metric_set == "roofline":
                metrics = ROOFLINE_METRICS
            elif metric_set == "deep_dive":
                metrics = DEEP_DIVE_METRICS
            elif metric_set == "minimal":
                metrics = MINIMAL_METRICS
            else:
                metrics = DEEP_DIVE_METRICS
        ncu_set = NCU_SET_BY_METRIC.get(metric_set, "full")

        cmd = [
            "ncu",
            "--set", ncu_set,
            "--metrics", ",".join(metrics),
        ]

        if preset == "minimal":
            cmd.extend([
                "--replay-mode", "application",
            ])
            if self.ncu_sampling_interval:
                cmd.extend(["--pm-sampling-interval", str(self.ncu_sampling_interval)])
            cmd.extend([
                "--target-processes", "all",
            ])
        else:
            cmd.extend(["--replay-mode", self.ncu_replay_mode])

        nvtx_filters = list(dict.fromkeys(nvtx_includes or self.nvtx_includes or []))
        for tag in nvtx_filters:
            cmd.extend(["--nvtx-include", tag])

        cmd.extend([
            "-o", output_path,
            python_executable or sys.executable,
            python_script,
        ])
        
        return cmd
    
    def get_proton_command(
        self,
        output_path: str,
        python_script: str,
        python_executable: Optional[str] = None,
    ) -> List[str]:
        """Generate Proton CLI invocation.
        
        Falls back to python -m torch._inductor.tools.proton if the standalone
        `proton` binary is not on PATH.
        """
        import sys
        
        output_arg = output_path if Path(output_path).suffix else f"{output_path}.json"
        if shutil.which("proton"):
            return [
                "proton",
                "profile",
                "--output", output_arg,
                "--python-script", python_script,
            ]
        return [
            python_executable or sys.executable,
            "-m",
            "torch._inductor.tools.proton",
            "profile",
            "--output", output_arg,
            "--python-script", python_script,
        ]
    
    def get_torch_profiler_config(self) -> dict:
        """Get PyTorch profiler configuration.
        
        Returns:
            Dictionary of kwargs for torch.profiler.profile()
        """
        try:
            import torch
            from torch.profiler import ProfilerActivity
        except Exception:
            return {}

        preset = (self.preset or "minimal").lower()
        if preset == "minimal":
            activities = [ProfilerActivity.CUDA]
            record_shapes = False
            profile_memory = False
            with_stack = False
            with_flops = False
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1)
            experimental_config = None
            if hasattr(torch._C, "_profiler"):
                try:
                    experimental_config = torch._C._profiler._ExperimentalConfig(
                        verbose=False,
                        profiler_measure_per_kernel=False,
                        use_cuda=True,
                    )
                except Exception:
                    experimental_config = None
        else:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            record_shapes = True
            profile_memory = True
            with_stack = True
            with_flops = True
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=1)
            experimental_config = None

        cfg = {
            "activities": activities,
            "record_shapes": record_shapes,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            "with_flops": with_flops,
            "schedule": schedule,
        }
        if experimental_config is not None:
            cfg["experimental_config"] = experimental_config
        return cfg


# Default profiler config instance
_DEFAULT_METRIC_SET = "minimal"
DEFAULT_PROFILER_CONFIG = ProfilerConfig(metric_set=_DEFAULT_METRIC_SET)


def set_default_profiler_metric_set(metric_set: str) -> None:
    """Update the global profiler configuration with a new metric set."""
    normalized = metric_set.strip().lower()
    if normalized not in {"deep_dive", "roofline", "minimal"}:
        raise ValueError(
            f"Unsupported metric set '{metric_set}'. "
            "Choose from 'deep_dive', 'roofline', or 'minimal'."
        )
    global _DEFAULT_METRIC_SET, DEFAULT_PROFILER_CONFIG
    _DEFAULT_METRIC_SET = normalized
    DEFAULT_PROFILER_CONFIG = ProfilerConfig(metric_set=_DEFAULT_METRIC_SET)


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _discover_nvtx_includes_from_source(module: Any) -> List[str]:
    if module is None:
        return []
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return []
    try:
        text = Path(module_file).read_text()
    except Exception:
        return []
    matches = re.findall(r'(?:nvtx_range|_nvtx_range)\(\s*[\'"]([^\'"]+)[\'"]', text)
    return [canonicalize_nvtx_name(m) for m in matches if m]


def discover_nvtx_includes(
    benchmark_module: Any,
    benchmark_class: Optional[str] = None,
    explicit: Optional[Sequence[str]] = None,
    limit: int = 6,
) -> List[str]:
    """Suggest NVTX filters for nsys/ncu based on benchmark source."""
    includes: List[str] = []
    if benchmark_class:
        includes.append(canonicalize_nvtx_name(benchmark_class))
    includes.extend(explicit or [])
    includes.extend(_discover_nvtx_includes_from_source(benchmark_module))
    
    priority_keywords = (
        "prefill",
        "decode",
        "router",
        "moe",
        "expert",
        "kv",
        "warp",
        "graph",
        "nvlink",
        "nccl",
        "p2p",
    )
    prioritized = [
        name for name in includes
        if any(key in name for key in priority_keywords)
    ]
    ordered = _dedupe_preserve_order(prioritized or includes)
    return ordered[:limit] if limit else ordered


def build_profiler_config_from_benchmark(
    config: Any,
    benchmark_module: Any = None,
    benchmark_class: Optional[str] = None,
) -> ProfilerConfig:
    """Create a ProfilerConfig tuned to the benchmark + harness defaults."""
    preset = str(getattr(config, "profile_type", None) or "minimal").lower()
    metric_set = getattr(config, "ncu_metric_set", None)
    if metric_set is None or str(metric_set).lower() == "auto":
        metric_set = preset if preset in {"minimal", "roofline", "deep_dive"} else "deep_dive"
    sampling_interval = getattr(config, "ncu_sampling_interval", 75000)
    explicit_includes = getattr(config, "nsys_nvtx_include", None)
    nvtx_includes = discover_nvtx_includes(
        benchmark_module,
        benchmark_class,
        explicit=explicit_includes,
    )
    return ProfilerConfig(
        metric_set=str(metric_set),
        preset=preset,
        nvtx_includes=nvtx_includes or None,
        ncu_sampling_interval=sampling_interval,
    )
