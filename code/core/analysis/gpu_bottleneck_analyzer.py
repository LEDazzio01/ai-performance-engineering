"""
GPU Bottleneck Analyzer - Advanced Performance Diagnosis

Provides automatic bottleneck classification, roofline analysis, and
optimization recommendations based on collected GPU metrics.

Designed for power users and AI assistants to quickly diagnose and
prioritize optimization efforts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# GPU HARDWARE SPECIFICATIONS DATABASE
# =============================================================================

@dataclass
class GPUSpecs:
    """Hardware specifications for a GPU architecture."""
    name: str
    architecture: str
    sm_count: int
    fp32_tflops: float  # Peak FP32 TFLOPS
    fp16_tflops: float  # Peak FP16 TFLOPS (Tensor Core)
    fp8_tflops: float   # Peak FP8 TFLOPS (Tensor Core, if available)
    int8_tops: float    # Peak INT8 TOPS (Tensor Core)
    memory_bandwidth_gbps: float  # HBM/GDDR bandwidth in GB/s
    memory_size_gb: float
    l2_cache_mb: float
    shared_mem_per_sm_kb: float
    registers_per_sm: int
    max_warps_per_sm: int
    nvlink_bandwidth_gbps: float  # Per-direction, per-link
    nvlink_links: int
    pcie_bandwidth_gbps: float
    tdp_watts: float
    
    @property
    def peak_memory_bandwidth_tbps(self) -> float:
        return self.memory_bandwidth_gbps / 1000.0
    
    @property
    def total_nvlink_bandwidth_gbps(self) -> float:
        return self.nvlink_bandwidth_gbps * self.nvlink_links * 2  # Bidirectional


# GPU Specifications Database
GPU_SPECS: Dict[str, GPUSpecs] = {
    # Blackwell
    "B200": GPUSpecs(
        name="NVIDIA B200",
        architecture="Blackwell",
        sm_count=148,
        fp32_tflops=80.0,
        fp16_tflops=2250.0,  # With sparsity: 4500
        fp8_tflops=4500.0,   # With sparsity: 9000
        int8_tops=4500.0,
        memory_bandwidth_gbps=8000.0,  # HBM3e
        memory_size_gb=192.0,
        l2_cache_mb=128.0,
        shared_mem_per_sm_kb=228.0,
        registers_per_sm=65536,
        max_warps_per_sm=64,
        nvlink_bandwidth_gbps=900.0,
        nvlink_links=18,
        pcie_bandwidth_gbps=128.0,  # PCIe 5.0 x16
        tdp_watts=1000.0,
    ),
    "B100": GPUSpecs(
        name="NVIDIA B100",
        architecture="Blackwell",
        sm_count=132,
        fp32_tflops=60.0,
        fp16_tflops=1800.0,
        fp8_tflops=3600.0,
        int8_tops=3600.0,
        memory_bandwidth_gbps=8000.0,
        memory_size_gb=192.0,
        l2_cache_mb=128.0,
        shared_mem_per_sm_kb=228.0,
        registers_per_sm=65536,
        max_warps_per_sm=64,
        nvlink_bandwidth_gbps=900.0,
        nvlink_links=18,
        pcie_bandwidth_gbps=128.0,
        tdp_watts=700.0,
    ),
    # Hopper
    "H100_SXM": GPUSpecs(
        name="NVIDIA H100 SXM",
        architecture="Hopper",
        sm_count=132,
        fp32_tflops=67.0,
        fp16_tflops=1979.0,  # With sparsity
        fp8_tflops=3958.0,   # With sparsity
        int8_tops=3958.0,
        memory_bandwidth_gbps=3350.0,  # HBM3
        memory_size_gb=80.0,
        l2_cache_mb=50.0,
        shared_mem_per_sm_kb=228.0,
        registers_per_sm=65536,
        max_warps_per_sm=64,
        nvlink_bandwidth_gbps=450.0,
        nvlink_links=18,
        pcie_bandwidth_gbps=128.0,
        tdp_watts=700.0,
    ),
    "H100_PCIe": GPUSpecs(
        name="NVIDIA H100 PCIe",
        architecture="Hopper",
        sm_count=114,
        fp32_tflops=51.0,
        fp16_tflops=1513.0,
        fp8_tflops=3026.0,
        int8_tops=3026.0,
        memory_bandwidth_gbps=2000.0,  # HBM2e
        memory_size_gb=80.0,
        l2_cache_mb=50.0,
        shared_mem_per_sm_kb=228.0,
        registers_per_sm=65536,
        max_warps_per_sm=64,
        nvlink_bandwidth_gbps=450.0,
        nvlink_links=4,
        pcie_bandwidth_gbps=128.0,
        tdp_watts=350.0,
    ),
    # Ada Lovelace
    "L40S": GPUSpecs(
        name="NVIDIA L40S",
        architecture="Ada Lovelace",
        sm_count=142,
        fp32_tflops=91.6,
        fp16_tflops=733.0,  # Tensor Core
        fp8_tflops=733.0,
        int8_tops=733.0,
        memory_bandwidth_gbps=864.0,  # GDDR6
        memory_size_gb=48.0,
        l2_cache_mb=96.0,
        shared_mem_per_sm_kb=100.0,
        registers_per_sm=65536,
        max_warps_per_sm=48,
        nvlink_bandwidth_gbps=0.0,  # No NVLink
        nvlink_links=0,
        pcie_bandwidth_gbps=64.0,  # PCIe 4.0 x16
        tdp_watts=350.0,
    ),
    # Ampere
    "A100_SXM": GPUSpecs(
        name="NVIDIA A100 SXM",
        architecture="Ampere",
        sm_count=108,
        fp32_tflops=19.5,
        fp16_tflops=312.0,  # Tensor Core
        fp8_tflops=0.0,     # No FP8
        int8_tops=624.0,
        memory_bandwidth_gbps=2039.0,  # HBM2e
        memory_size_gb=80.0,
        l2_cache_mb=40.0,
        shared_mem_per_sm_kb=164.0,
        registers_per_sm=65536,
        max_warps_per_sm=64,
        nvlink_bandwidth_gbps=300.0,
        nvlink_links=12,
        pcie_bandwidth_gbps=64.0,
        tdp_watts=400.0,
    ),
    "A100_PCIe": GPUSpecs(
        name="NVIDIA A100 PCIe",
        architecture="Ampere",
        sm_count=108,
        fp32_tflops=19.5,
        fp16_tflops=312.0,
        fp8_tflops=0.0,
        int8_tops=624.0,
        memory_bandwidth_gbps=1935.0,
        memory_size_gb=80.0,
        l2_cache_mb=40.0,
        shared_mem_per_sm_kb=164.0,
        registers_per_sm=65536,
        max_warps_per_sm=64,
        nvlink_bandwidth_gbps=300.0,
        nvlink_links=4,
        pcie_bandwidth_gbps=64.0,
        tdp_watts=300.0,
    ),
}


def get_gpu_specs(gpu_name: str) -> Optional[GPUSpecs]:
    """Look up GPU specs by name (fuzzy match)."""
    gpu_name_upper = gpu_name.upper()
    
    # Direct match
    for key, specs in GPU_SPECS.items():
        if key in gpu_name_upper or specs.name.upper() in gpu_name_upper:
            return specs
    
    # Fuzzy match
    if "B200" in gpu_name_upper or "BLACKWELL" in gpu_name_upper:
        return GPU_SPECS["B200"]
    if "B100" in gpu_name_upper:
        return GPU_SPECS["B100"]
    if "H100" in gpu_name_upper:
        if "SXM" in gpu_name_upper:
            return GPU_SPECS["H100_SXM"]
        return GPU_SPECS["H100_PCIe"]
    if "L40" in gpu_name_upper:
        return GPU_SPECS["L40S"]
    if "A100" in gpu_name_upper:
        if "SXM" in gpu_name_upper:
            return GPU_SPECS["A100_SXM"]
        return GPU_SPECS["A100_PCIe"]
    
    return None


# =============================================================================
# BOTTLENECK CLASSIFICATION
# =============================================================================

class BottleneckType(Enum):
    """Primary bottleneck categories."""
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    LATENCY_BOUND = "latency_bound"
    OCCUPANCY_LIMITED = "occupancy_limited"
    INSTRUCTION_FETCH = "instruction_fetch"
    SYNCHRONIZATION = "synchronization"
    REGISTER_SPILL = "register_spill"
    BANK_CONFLICT = "bank_conflict"
    UNCOALESCED_ACCESS = "uncoalesced_access"
    WARP_DIVERGENCE = "warp_divergence"
    TENSOR_CORE_UNDERUTILIZED = "tensor_core_underutilized"
    BALANCED = "balanced"  # No clear bottleneck
    UNKNOWN = "unknown"


class StallCategory(Enum):
    """Aggregated stall categories for easier diagnosis."""
    MEMORY_LATENCY = "memory_latency"      # Long scoreboard, memory throttle
    COMPUTE_THROUGHPUT = "compute_throughput"  # Math pipe throttle
    SYNCHRONIZATION = "synchronization"     # Barrier, membar, drain
    DATA_DEPENDENCY = "data_dependency"     # Short scoreboard, dependency
    INSTRUCTION_ISSUE = "instruction_issue" # No instruction, dispatch, not selected
    RESOURCE_CONTENTION = "resource_contention"  # MIO, tex, lg throttle
    OTHER = "other"


@dataclass
class StallAnalysis:
    """Aggregated stall analysis results."""
    dominant_category: StallCategory
    category_percentages: Dict[StallCategory, float]
    top_stall_reasons: List[Tuple[str, float]]  # (metric_name, percentage)
    actionable_insights: List[str]


@dataclass
class BottleneckDiagnosis:
    """Complete bottleneck diagnosis for a kernel or workload."""
    primary_bottleneck: BottleneckType
    secondary_bottlenecks: List[BottleneckType]
    confidence: float  # 0-1, how confident we are in the diagnosis
    
    # Roofline analysis
    arithmetic_intensity: Optional[float]  # FLOP/byte
    compute_utilization_pct: float
    memory_utilization_pct: float
    distance_from_roofline_pct: float  # How far from peak (0 = at peak)
    
    # Stall analysis
    stall_analysis: Optional[StallAnalysis]
    
    # Key metrics that led to diagnosis
    key_metrics: Dict[str, float]
    
    # Prioritized recommendations
    recommendations: List[str]
    
    # Estimated improvement potential
    improvement_potential: str  # "low", "medium", "high"


# Stall metric to category mapping
STALL_CATEGORY_MAP: Dict[str, StallCategory] = {
    # Memory latency
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": StallCategory.MEMORY_LATENCY,
    "smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct": StallCategory.MEMORY_LATENCY,
    "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct": StallCategory.MEMORY_LATENCY,
    
    # Compute throughput
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct": StallCategory.COMPUTE_THROUGHPUT,
    
    # Synchronization
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": StallCategory.SYNCHRONIZATION,
    "smsp__warp_issue_stalled_membar_per_warp_active.pct": StallCategory.SYNCHRONIZATION,
    "smsp__warp_issue_stalled_drain_per_warp_active.pct": StallCategory.SYNCHRONIZATION,
    "smsp__warp_issue_stalled_wait_per_warp_active.pct": StallCategory.SYNCHRONIZATION,
    
    # Data dependency
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": StallCategory.DATA_DEPENDENCY,
    "smsp__warp_issue_stalled_dependency_per_warp_active.pct": StallCategory.DATA_DEPENDENCY,
    
    # Instruction issue
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct": StallCategory.INSTRUCTION_ISSUE,
    "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct": StallCategory.INSTRUCTION_ISSUE,
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct": StallCategory.INSTRUCTION_ISSUE,
    
    # Resource contention
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct": StallCategory.RESOURCE_CONTENTION,
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct": StallCategory.RESOURCE_CONTENTION,
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct": StallCategory.RESOURCE_CONTENTION,
    
    # Other
    "smsp__warp_issue_stalled_sleeping_per_warp_active.pct": StallCategory.OTHER,
    "smsp__warp_issue_stalled_misc_per_warp_active.pct": StallCategory.OTHER,
}


def analyze_stalls(metrics: Dict[str, float]) -> StallAnalysis:
    """Aggregate stall metrics into categories and generate insights."""
    category_totals: Dict[StallCategory, float] = {cat: 0.0 for cat in StallCategory}
    stall_values: List[Tuple[str, float]] = []
    
    for metric_name, category in STALL_CATEGORY_MAP.items():
        value = metrics.get(metric_name, 0.0)
        if value > 0:
            category_totals[category] += value
            stall_values.append((metric_name, value))
    
    # Sort stalls by value
    stall_values.sort(key=lambda x: x[1], reverse=True)
    top_stalls = stall_values[:5]
    
    # Find dominant category
    total_stall = sum(category_totals.values())
    if total_stall > 0:
        category_pcts = {cat: (val / total_stall) * 100 for cat, val in category_totals.items()}
    else:
        category_pcts = {cat: 0.0 for cat in StallCategory}
    
    dominant_cat = max(category_totals, key=category_totals.get)  # type: ignore
    
    # Generate insights
    insights = []
    if category_totals[StallCategory.MEMORY_LATENCY] > 30:
        insights.append("High memory latency stalls - consider prefetching, better locality, or async copies")
    if category_totals[StallCategory.SYNCHRONIZATION] > 20:
        insights.append("Significant sync stalls - reduce barrier frequency or use warp-level sync")
    if category_totals[StallCategory.DATA_DEPENDENCY] > 25:
        insights.append("Data dependency stalls - increase ILP, unroll loops, or reorder instructions")
    if category_totals[StallCategory.RESOURCE_CONTENTION] > 15:
        insights.append("Resource contention detected - check shared memory/texture/MIO balance")
    if category_totals[StallCategory.INSTRUCTION_ISSUE] > 10:
        insights.append("Instruction issue stalls - kernel may be too large or have poor code locality")
    
    return StallAnalysis(
        dominant_category=dominant_cat,
        category_percentages=category_pcts,
        top_stall_reasons=top_stalls,
        actionable_insights=insights,
    )


def calculate_arithmetic_intensity(
    flops: float,
    bytes_transferred: float,
) -> float:
    """Calculate arithmetic intensity (FLOP/byte)."""
    if bytes_transferred <= 0:
        return float('inf')
    return flops / bytes_transferred


def calculate_roofline_position(
    arithmetic_intensity: float,
    achieved_tflops: float,
    gpu_specs: GPUSpecs,
    precision: str = "fp16",
) -> Tuple[float, float, str]:
    """
    Calculate position on roofline model.
    
    Returns:
        (compute_utilization_pct, memory_utilization_pct, bound_type)
    """
    # Get peak compute based on precision
    if precision == "fp8":
        peak_tflops = gpu_specs.fp8_tflops
    elif precision in ("fp16", "bf16"):
        peak_tflops = gpu_specs.fp16_tflops
    elif precision == "int8":
        peak_tflops = gpu_specs.int8_tops
    else:  # fp32
        peak_tflops = gpu_specs.fp32_tflops
    
    peak_bandwidth_tbps = gpu_specs.peak_memory_bandwidth_tbps
    
    # Ridge point (where compute meets memory)
    ridge_point = peak_tflops / peak_bandwidth_tbps  # FLOP/byte
    
    # Theoretical peak at this arithmetic intensity
    if arithmetic_intensity < ridge_point:
        # Memory bound region
        theoretical_peak = arithmetic_intensity * peak_bandwidth_tbps
        bound_type = "memory"
    else:
        # Compute bound region
        theoretical_peak = peak_tflops
        bound_type = "compute"
    
    compute_util = (achieved_tflops / peak_tflops) * 100 if peak_tflops > 0 else 0
    memory_util = (achieved_tflops / (arithmetic_intensity * peak_bandwidth_tbps)) * 100 if arithmetic_intensity > 0 else 0
    memory_util = min(memory_util, 100.0)
    
    return compute_util, memory_util, bound_type


def diagnose_bottleneck(
    metrics: Dict[str, float],
    gpu_name: str = "H100",
    kernel_time_us: Optional[float] = None,
    flops: Optional[float] = None,
    bytes_transferred: Optional[float] = None,
) -> BottleneckDiagnosis:
    """
    Comprehensive bottleneck diagnosis from collected metrics.
    
    Args:
        metrics: Dict of NCU/profiler metrics
        gpu_name: GPU model name for specs lookup
        kernel_time_us: Kernel execution time in microseconds
        flops: Total FLOPs executed (if known)
        bytes_transferred: Total bytes moved (if known)
    
    Returns:
        BottleneckDiagnosis with classification and recommendations
    """
    gpu_specs = get_gpu_specs(gpu_name)
    
    # Extract key metrics with defaults
    sm_throughput = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    dram_throughput = metrics.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 
                                   metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0))
    occupancy = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
    tensor_util = metrics.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0)
    
    # Register spills
    local_ld = metrics.get("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", 0)
    local_st = metrics.get("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", 0)
    has_spills = (local_ld + local_st) > 0
    
    # Bank conflicts
    bank_conflicts_ld = metrics.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 0)
    bank_conflicts_st = metrics.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 0)
    has_bank_conflicts = (bank_conflicts_ld + bank_conflicts_st) > 1000
    
    # Coalescing efficiency
    ld_efficiency = metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 100)
    st_efficiency = metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", 100)
    poor_coalescing = ld_efficiency < 50 or st_efficiency < 50
    
    # Warp divergence
    branch_uniformity = metrics.get("smsp__sass_average_branch_targets_threads_uniform.pct", 100)
    has_divergence = branch_uniformity < 80
    
    # Stall analysis
    stall_analysis = analyze_stalls(metrics)
    
    # Determine primary bottleneck
    bottlenecks: List[Tuple[BottleneckType, float]] = []
    
    # Compute vs Memory bound
    if sm_throughput > 70 and dram_throughput < 50:
        bottlenecks.append((BottleneckType.COMPUTE_BOUND, sm_throughput))
    elif dram_throughput > 70 and sm_throughput < 50:
        bottlenecks.append((BottleneckType.MEMORY_BOUND, dram_throughput))
    
    # Occupancy
    if occupancy < 30:
        bottlenecks.append((BottleneckType.OCCUPANCY_LIMITED, 100 - occupancy))
    
    # Register spills
    if has_spills:
        bottlenecks.append((BottleneckType.REGISTER_SPILL, 80))
    
    # Bank conflicts
    if has_bank_conflicts:
        bottlenecks.append((BottleneckType.BANK_CONFLICT, 70))
    
    # Coalescing
    if poor_coalescing:
        bottlenecks.append((BottleneckType.UNCOALESCED_ACCESS, 100 - min(ld_efficiency, st_efficiency)))
    
    # Divergence
    if has_divergence:
        bottlenecks.append((BottleneckType.WARP_DIVERGENCE, 100 - branch_uniformity))
    
    # Tensor core underutilization (if doing matmul-heavy workload)
    if tensor_util < 30 and sm_throughput > 50:
        bottlenecks.append((BottleneckType.TENSOR_CORE_UNDERUTILIZED, 100 - tensor_util))
    
    # Synchronization
    if stall_analysis.category_percentages.get(StallCategory.SYNCHRONIZATION, 0) > 30:
        bottlenecks.append((BottleneckType.SYNCHRONIZATION, 
                          stall_analysis.category_percentages[StallCategory.SYNCHRONIZATION]))
    
    # Instruction fetch
    if stall_analysis.category_percentages.get(StallCategory.INSTRUCTION_ISSUE, 0) > 15:
        bottlenecks.append((BottleneckType.INSTRUCTION_FETCH,
                          stall_analysis.category_percentages[StallCategory.INSTRUCTION_ISSUE]))
    
    # Sort by severity
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    if not bottlenecks:
        if sm_throughput > 60 and dram_throughput > 60:
            primary = BottleneckType.BALANCED
        else:
            primary = BottleneckType.UNKNOWN
        secondary = []
        confidence = 0.5
    else:
        primary = bottlenecks[0][0]
        secondary = [b[0] for b in bottlenecks[1:4]]
        confidence = min(bottlenecks[0][1] / 100, 0.95)
    
    # Roofline analysis
    arithmetic_intensity = None
    distance_from_roofline = 100.0
    
    if flops is not None and bytes_transferred is not None and gpu_specs:
        arithmetic_intensity = calculate_arithmetic_intensity(flops, bytes_transferred)
        if kernel_time_us:
            achieved_tflops = (flops / 1e12) / (kernel_time_us / 1e6)
            compute_util, memory_util, _ = calculate_roofline_position(
                arithmetic_intensity, achieved_tflops, gpu_specs
            )
            distance_from_roofline = 100 - max(compute_util, memory_util)
    
    # Generate recommendations
    recommendations = _generate_recommendations(
        primary, secondary, stall_analysis, metrics, gpu_specs
    )
    
    # Estimate improvement potential
    if distance_from_roofline > 50 or occupancy < 30 or has_spills:
        improvement_potential = "high"
    elif distance_from_roofline > 20 or has_bank_conflicts or poor_coalescing:
        improvement_potential = "medium"
    else:
        improvement_potential = "low"
    
    return BottleneckDiagnosis(
        primary_bottleneck=primary,
        secondary_bottlenecks=secondary,
        confidence=confidence,
        arithmetic_intensity=arithmetic_intensity,
        compute_utilization_pct=sm_throughput,
        memory_utilization_pct=dram_throughput,
        distance_from_roofline_pct=distance_from_roofline,
        stall_analysis=stall_analysis,
        key_metrics={
            "sm_throughput_pct": sm_throughput,
            "dram_throughput_pct": dram_throughput,
            "occupancy_pct": occupancy,
            "tensor_util_pct": tensor_util,
            "ld_efficiency_pct": ld_efficiency,
            "st_efficiency_pct": st_efficiency,
            "branch_uniformity_pct": branch_uniformity,
        },
        recommendations=recommendations,
        improvement_potential=improvement_potential,
    )


def _generate_recommendations(
    primary: BottleneckType,
    secondary: List[BottleneckType],
    stall_analysis: StallAnalysis,
    metrics: Dict[str, float],
    gpu_specs: Optional[GPUSpecs],
) -> List[str]:
    """Generate prioritized optimization recommendations."""
    recs = []
    
    # Primary bottleneck recommendations
    if primary == BottleneckType.MEMORY_BOUND:
        recs.extend([
            "🔴 MEMORY BOUND: Reduce memory traffic or improve locality",
            "   → Use shared memory for data reuse",
            "   → Implement tiling/blocking to fit in cache",
            "   → Use vectorized loads (float4/int4)",
            "   → Consider data compression or lower precision",
        ])
    elif primary == BottleneckType.COMPUTE_BOUND:
        recs.extend([
            "🟢 COMPUTE BOUND: Already efficient! Focus on algorithmic improvements",
            "   → Use Tensor Cores if not already (FP16/FP8/INT8)",
            "   → Check if torch.compile() or custom CUDA kernels help",
            "   → Consider model quantization for inference",
        ])
    elif primary == BottleneckType.OCCUPANCY_LIMITED:
        occupancy = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
        regs = metrics.get("launch__registers_per_thread", 0)
        smem = metrics.get("launch__shared_mem_per_block", 0)
        recs.extend([
            f"🟡 LOW OCCUPANCY ({occupancy:.0f}%): Increase parallelism",
            f"   → Current: {regs:.0f} regs/thread, {smem:.0f} bytes shared/block",
            "   → Reduce register usage (smaller tiles, fewer local vars)",
            "   → Reduce shared memory if possible",
            "   → Try different block sizes (128, 256, 512)",
        ])
    elif primary == BottleneckType.REGISTER_SPILL:
        recs.extend([
            "🔴 REGISTER SPILLS: Local memory access detected!",
            "   → Reduce register pressure (smaller arrays, fewer temps)",
            "   → Use #pragma unroll judiciously",
            "   → Consider __launch_bounds__ to control register allocation",
            "   → Split kernel into smaller kernels if needed",
        ])
    elif primary == BottleneckType.BANK_CONFLICT:
        recs.extend([
            "🟡 SHARED MEMORY BANK CONFLICTS detected",
            "   → Add padding to shared memory arrays (e.g., [N][N+1])",
            "   → Reorganize access patterns for conflict-free access",
            "   → Use swizzling for 2D array accesses",
        ])
    elif primary == BottleneckType.UNCOALESCED_ACCESS:
        ld_eff = metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 0)
        recs.extend([
            f"🔴 UNCOALESCED MEMORY ACCESS (load eff: {ld_eff:.0f}%)",
            "   → Ensure threads access consecutive memory locations",
            "   → Transpose data layout if accessing columns",
            "   → Use shared memory for irregular access patterns",
            "   → Align data to 128-byte boundaries",
        ])
    elif primary == BottleneckType.WARP_DIVERGENCE:
        recs.extend([
            "🟡 WARP DIVERGENCE detected",
            "   → Reorganize data to group similar operations",
            "   → Use predication instead of branches where possible",
            "   → Sort input data by operation type",
        ])
    elif primary == BottleneckType.SYNCHRONIZATION:
        recs.extend([
            "🟡 HIGH SYNCHRONIZATION OVERHEAD",
            "   → Reduce __syncthreads() frequency",
            "   → Use warp-level primitives (__syncwarp, shuffle)",
            "   → Consider async copy with cp.async",
        ])
    elif primary == BottleneckType.TENSOR_CORE_UNDERUTILIZED:
        recs.extend([
            "🟡 TENSOR CORES UNDERUTILIZED",
            "   → Ensure matrix dimensions are multiples of 16 (FP16) or 32 (FP8)",
            "   → Use torch.matmul() or CUTLASS instead of manual loops",
            "   → Enable Tensor Core math: torch.backends.cuda.matmul.allow_tf32=True",
        ])
    
    # Add stall-specific recommendations
    recs.extend(stall_analysis.actionable_insights)
    
    return recs


def format_diagnosis_report(diagnosis: BottleneckDiagnosis) -> str:
    """Format diagnosis as a human-readable report."""
    lines = [
        "=" * 70,
        "GPU BOTTLENECK ANALYSIS REPORT",
        "=" * 70,
        "",
        f"PRIMARY BOTTLENECK: {diagnosis.primary_bottleneck.value.upper()}",
        f"Confidence: {diagnosis.confidence:.0%}",
        f"Improvement Potential: {diagnosis.improvement_potential.upper()}",
        "",
    ]
    
    if diagnosis.secondary_bottlenecks:
        secondary = ", ".join(b.value for b in diagnosis.secondary_bottlenecks)
        lines.append(f"Secondary issues: {secondary}")
        lines.append("")
    
    lines.extend([
        "KEY METRICS:",
        f"  Compute utilization: {diagnosis.compute_utilization_pct:.1f}%",
        f"  Memory utilization:  {diagnosis.memory_utilization_pct:.1f}%",
    ])
    
    if diagnosis.arithmetic_intensity is not None:
        lines.append(f"  Arithmetic intensity: {diagnosis.arithmetic_intensity:.2f} FLOP/byte")
    
    lines.append(f"  Distance from roofline: {diagnosis.distance_from_roofline_pct:.1f}%")
    lines.append("")
    
    if diagnosis.stall_analysis:
        lines.extend([
            "STALL ANALYSIS:",
            f"  Dominant category: {diagnosis.stall_analysis.dominant_category.value}",
        ])
        for cat, pct in sorted(
            diagnosis.stall_analysis.category_percentages.items(),
            key=lambda x: x[1],
            reverse=True
        )[:4]:
            if pct > 1:
                lines.append(f"    {cat.value}: {pct:.1f}%")
        lines.append("")
    
    lines.extend([
        "RECOMMENDATIONS (prioritized):",
    ])
    for i, rec in enumerate(diagnosis.recommendations[:10], 1):
        lines.append(f"  {rec}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# PROFILE COMPARISON UTILITY
# =============================================================================

@dataclass
class MetricDelta:
    """Change in a metric between baseline and optimized."""
    metric_name: str
    baseline_value: float
    optimized_value: float
    absolute_delta: float
    relative_delta_pct: float
    direction: str  # "improved", "regressed", "unchanged"
    significance: str  # "high", "medium", "low"


@dataclass 
class ProfileComparison:
    """Comparison between baseline and optimized profiles."""
    speedup: float
    baseline_time_us: float
    optimized_time_us: float
    
    improved_metrics: List[MetricDelta]
    regressed_metrics: List[MetricDelta]
    unchanged_metrics: List[MetricDelta]
    
    bottleneck_shift: Optional[str]  # e.g., "memory_bound -> compute_bound"
    key_improvements: List[str]
    remaining_issues: List[str]


def _diff_metrics(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    baseline_time_us: float,
    optimized_time_us: float,
    improvement_threshold_pct: float = 5.0,
) -> ProfileComparison:
    """
    Internal helper: Compare baseline and optimized profile metrics.
    
    Identifies what improved, what regressed, and what bottlenecks shifted.
    
    NOTE: This is an internal helper. Use the unified `compare_profiles()` from
    `core.perf_core_base` or `core.engine.ProfileEngine.compare()` which
    integrates this analysis with file-level comparison.
    """
    speedup = baseline_time_us / optimized_time_us if optimized_time_us > 0 else 1.0
    
    improved = []
    regressed = []
    unchanged = []
    
    # Metrics where higher is better
    higher_better = {
        "sm__throughput", "dram__throughput", "occupancy", "hit_rate",
        "utilization", "efficiency", "uniformity",
    }
    
    # Metrics where lower is better
    lower_better = {
        "stalled", "bank_conflict", "spill", "local_op",
    }
    
    all_metrics = set(baseline_metrics.keys()) | set(optimized_metrics.keys())
    
    for metric in all_metrics:
        base_val = baseline_metrics.get(metric, 0)
        opt_val = optimized_metrics.get(metric, 0)
        
        if base_val == 0 and opt_val == 0:
            continue
        
        abs_delta = opt_val - base_val
        rel_delta = (abs_delta / base_val * 100) if base_val != 0 else (100 if opt_val > 0 else 0)
        
        # Determine if improvement
        is_higher_better = any(k in metric.lower() for k in higher_better)
        is_lower_better = any(k in metric.lower() for k in lower_better)
        
        if is_higher_better:
            direction = "improved" if abs_delta > 0 else ("regressed" if abs_delta < 0 else "unchanged")
        elif is_lower_better:
            direction = "improved" if abs_delta < 0 else ("regressed" if abs_delta > 0 else "unchanged")
        else:
            direction = "unchanged"  # Unknown metric semantics
        
        # Significance
        if abs(rel_delta) > 20:
            significance = "high"
        elif abs(rel_delta) > improvement_threshold_pct:
            significance = "medium"
        else:
            significance = "low"
            if abs(rel_delta) < improvement_threshold_pct:
                direction = "unchanged"
        
        delta = MetricDelta(
            metric_name=metric,
            baseline_value=base_val,
            optimized_value=opt_val,
            absolute_delta=abs_delta,
            relative_delta_pct=rel_delta,
            direction=direction,
            significance=significance,
        )
        
        if direction == "improved":
            improved.append(delta)
        elif direction == "regressed":
            regressed.append(delta)
        else:
            unchanged.append(delta)
    
    # Sort by significance
    improved.sort(key=lambda x: abs(x.relative_delta_pct), reverse=True)
    regressed.sort(key=lambda x: abs(x.relative_delta_pct), reverse=True)
    
    # Diagnose bottleneck shift
    base_diagnosis = diagnose_bottleneck(baseline_metrics)
    opt_diagnosis = diagnose_bottleneck(optimized_metrics)
    
    if base_diagnosis.primary_bottleneck != opt_diagnosis.primary_bottleneck:
        bottleneck_shift = f"{base_diagnosis.primary_bottleneck.value} → {opt_diagnosis.primary_bottleneck.value}"
    else:
        bottleneck_shift = None
    
    # Key improvements narrative
    key_improvements = []
    if speedup > 1.1:
        key_improvements.append(f"Achieved {speedup:.2f}x speedup ({baseline_time_us:.1f}µs → {optimized_time_us:.1f}µs)")
    
    for delta in improved[:3]:
        if delta.significance in ("high", "medium"):
            key_improvements.append(
                f"{delta.metric_name}: {delta.baseline_value:.1f} → {delta.optimized_value:.1f} "
                f"({delta.relative_delta_pct:+.1f}%)"
            )
    
    # Remaining issues
    remaining_issues = opt_diagnosis.recommendations[:3]
    
    return ProfileComparison(
        speedup=speedup,
        baseline_time_us=baseline_time_us,
        optimized_time_us=optimized_time_us,
        improved_metrics=improved,
        regressed_metrics=regressed,
        unchanged_metrics=unchanged,
        bottleneck_shift=bottleneck_shift,
        key_improvements=key_improvements,
        remaining_issues=remaining_issues,
    )


def format_comparison_report(comparison: ProfileComparison) -> str:
    """Format profile comparison as a human-readable report."""
    lines = [
        "=" * 70,
        "PROFILE COMPARISON REPORT",
        "=" * 70,
        "",
        f"SPEEDUP: {comparison.speedup:.2f}x",
        f"  Baseline: {comparison.baseline_time_us:.1f} µs",
        f"  Optimized: {comparison.optimized_time_us:.1f} µs",
        "",
    ]
    
    if comparison.bottleneck_shift:
        lines.append(f"BOTTLENECK SHIFT: {comparison.bottleneck_shift}")
        lines.append("")
    
    if comparison.key_improvements:
        lines.append("KEY IMPROVEMENTS:")
        for imp in comparison.key_improvements:
            lines.append(f"  ✅ {imp}")
        lines.append("")
    
    if comparison.regressed_metrics:
        lines.append("REGRESSIONS (investigate):")
        for delta in comparison.regressed_metrics[:5]:
            lines.append(
                f"  ⚠️ {delta.metric_name}: {delta.baseline_value:.1f} → {delta.optimized_value:.1f} "
                f"({delta.relative_delta_pct:+.1f}%)"
            )
        lines.append("")
    
    if comparison.remaining_issues:
        lines.append("REMAINING OPTIMIZATION OPPORTUNITIES:")
        for issue in comparison.remaining_issues:
            lines.append(f"  {issue}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# QUICK DIAGNOSIS FUNCTIONS FOR AI ASSISTANTS
# =============================================================================

def quick_diagnosis(
    sm_throughput_pct: float,
    dram_throughput_pct: float,
    occupancy_pct: float,
    tensor_util_pct: float = 0,
    top_stall: str = "",
    top_stall_pct: float = 0,
    gpu_name: str = "H100",
) -> str:
    """
    Quick diagnosis from a few key metrics.
    
    Useful for AI assistants to quickly assess a kernel's performance.
    """
    issues = []
    
    # Utilization assessment
    if sm_throughput_pct < 30 and dram_throughput_pct < 30:
        issues.append("⚠️ LOW UTILIZATION: Both compute and memory underutilized")
        issues.append("   Check occupancy, launch config, and workload size")
    elif dram_throughput_pct > 70 and sm_throughput_pct < 40:
        issues.append("🔴 MEMORY BOUND: DRAM={:.0f}%, SM={:.0f}%".format(dram_throughput_pct, sm_throughput_pct))
        issues.append("   Focus on reducing memory traffic, improving locality")
    elif sm_throughput_pct > 70 and dram_throughput_pct < 40:
        issues.append("🟢 COMPUTE BOUND: SM={:.0f}%, DRAM={:.0f}%".format(sm_throughput_pct, dram_throughput_pct))
        issues.append("   Good efficiency! Consider Tensor Cores if not using")
    else:
        issues.append("🟡 BALANCED: SM={:.0f}%, DRAM={:.0f}%".format(sm_throughput_pct, dram_throughput_pct))
    
    # Occupancy
    if occupancy_pct < 25:
        issues.append(f"🔴 VERY LOW OCCUPANCY: {occupancy_pct:.0f}%")
        issues.append("   Reduce registers/shared mem or increase block size")
    elif occupancy_pct < 50:
        issues.append(f"🟡 LOW OCCUPANCY: {occupancy_pct:.0f}%")
    
    # Tensor cores
    if tensor_util_pct > 0 and tensor_util_pct < 30:
        issues.append(f"🟡 TENSOR CORES UNDERUTILIZED: {tensor_util_pct:.0f}%")
        issues.append("   Check matrix alignment (multiples of 16/32)")
    
    # Top stall
    if top_stall_pct > 20:
        issues.append(f"⚠️ TOP STALL: {top_stall} ({top_stall_pct:.0f}%)")
        if "scoreboard" in top_stall.lower() or "memory" in top_stall.lower():
            issues.append("   Memory latency issue - prefetch or hide latency")
        elif "barrier" in top_stall.lower():
            issues.append("   Reduce sync frequency or use warp-level sync")
    
    return "\n".join(issues)


# =============================================================================
# EXPORT FOR EASY ACCESS
# =============================================================================

__all__ = [
    # Specs
    "GPUSpecs",
    "GPU_SPECS",
    "get_gpu_specs",
    # Types
    "BottleneckType",
    "StallCategory",
    # Analysis
    "StallAnalysis",
    "BottleneckDiagnosis",
    "analyze_stalls",
    "diagnose_bottleneck",
    "format_diagnosis_report",
    # Comparison (data classes for integration)
    "ProfileComparison",
    "MetricDelta",
    "format_comparison_report",
    # Quick helpers
    "quick_diagnosis",
    # Internal helper for integration (prefixed with _)
    # _diff_metrics - used by core.perf_core_base.compare_profiles()
]

