"""
Kernel Fingerprinting & Pattern Recognition

Identifies kernel types and common patterns to suggest targeted optimizations.
Useful for AI assistants to quickly understand what kind of kernel they're analyzing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re


class KernelType(Enum):
    """High-level kernel classification."""
    GEMM = "gemm"                    # Matrix multiplication
    ATTENTION = "attention"          # Self/cross attention
    ELEMENTWISE = "elementwise"      # Point-wise operations
    REDUCTION = "reduction"          # Sum, mean, max, etc.
    SOFTMAX = "softmax"              # Softmax/normalization
    LAYERNORM = "layernorm"          # Layer normalization
    EMBEDDING = "embedding"          # Embedding lookup
    COPY = "copy"                    # Memory copy/transpose
    CONVOLUTION = "convolution"      # Conv2D, Conv3D
    POOLING = "pooling"              # MaxPool, AvgPool
    SCATTER_GATHER = "scatter_gather"  # Sparse operations
    FUSED = "fused"                  # Fused multi-op kernel
    UNKNOWN = "unknown"


class ComputePattern(Enum):
    """Compute access pattern."""
    MATMUL = "matmul"               # O(N³) compute, N² data
    MAP = "map"                     # 1:1 input:output
    REDUCE = "reduce"              # N:1 input:output
    BROADCAST = "broadcast"        # 1:N input:output
    STENCIL = "stencil"            # Neighbor access pattern
    SCATTER = "scatter"            # Irregular write
    GATHER = "gather"              # Irregular read
    TRANSPOSE = "transpose"        # Data reordering


class MemoryPattern(Enum):
    """Memory access pattern."""
    COALESCED_STREAMING = "coalesced_streaming"
    COALESCED_REUSE = "coalesced_reuse"
    STRIDED = "strided"
    RANDOM = "random"
    BLOCKED = "blocked"  # Tiled access with reuse


@dataclass
class KernelFingerprint:
    """Complete fingerprint of a kernel's characteristics."""
    kernel_name: str
    kernel_type: KernelType
    compute_pattern: ComputePattern
    memory_pattern: MemoryPattern
    
    # Derived characteristics
    arithmetic_intensity_class: str  # "low", "medium", "high"
    parallelism_type: str  # "data", "task", "tensor"
    reuse_opportunity: str  # "none", "temporal", "spatial", "both"
    
    # Optimization hints
    tensor_core_candidate: bool
    tiling_beneficial: bool
    fusion_candidate: bool
    async_copy_candidate: bool
    
    # Matched pattern info
    pattern_confidence: float
    pattern_details: str


# Kernel name patterns for recognition
KERNEL_PATTERNS: Dict[str, Tuple[KernelType, ComputePattern, float]] = {
    # GEMM patterns
    r"gemm|matmul|mm_|_mm|cublas.*gemm": (KernelType.GEMM, ComputePattern.MATMUL, 0.95),
    r"cutlass.*gemm|cutlass.*mm": (KernelType.GEMM, ComputePattern.MATMUL, 0.95),
    r"sm\d+_xmma|hmma|imma|dmma": (KernelType.GEMM, ComputePattern.MATMUL, 0.90),
    r"ampere_.*gemm|hopper_.*gemm|blackwell_.*gemm": (KernelType.GEMM, ComputePattern.MATMUL, 0.95),
    r"linear|dense|fc_|fully_connected": (KernelType.GEMM, ComputePattern.MATMUL, 0.85),
    
    # Attention patterns
    r"attention|attn|self_attn|cross_attn": (KernelType.ATTENTION, ComputePattern.MATMUL, 0.90),
    r"flash.*attn|fused.*attn|mha|multihead": (KernelType.ATTENTION, ComputePattern.MATMUL, 0.95),
    r"sdpa|scaled_dot_product": (KernelType.ATTENTION, ComputePattern.MATMUL, 0.95),
    r"rotary|rope|pos_emb": (KernelType.ATTENTION, ComputePattern.MAP, 0.80),
    
    # Elementwise patterns
    r"elementwise|pointwise|ewise|pw_": (KernelType.ELEMENTWISE, ComputePattern.MAP, 0.90),
    r"relu|gelu|silu|swish|sigmoid|tanh": (KernelType.ELEMENTWISE, ComputePattern.MAP, 0.85),
    r"add_|mul_|sub_|div_|_add|_mul": (KernelType.ELEMENTWISE, ComputePattern.MAP, 0.80),
    r"bias|residual": (KernelType.ELEMENTWISE, ComputePattern.MAP, 0.75),
    
    # Reduction patterns
    r"reduce|sum|mean|avg|max|min|argmax|argmin": (KernelType.REDUCTION, ComputePattern.REDUCE, 0.90),
    r"all_reduce|nccl.*reduce": (KernelType.REDUCTION, ComputePattern.REDUCE, 0.95),
    
    # Softmax
    r"softmax|log_softmax|safe_softmax": (KernelType.SOFTMAX, ComputePattern.REDUCE, 0.95),
    
    # LayerNorm / RMSNorm
    r"layernorm|layer_norm|ln_|rmsnorm|rms_norm": (KernelType.LAYERNORM, ComputePattern.REDUCE, 0.95),
    r"batch_norm|batchnorm|bn_|instance_norm": (KernelType.LAYERNORM, ComputePattern.REDUCE, 0.90),
    
    # Embedding
    r"embedding|embed|lookup|gather_embedding": (KernelType.EMBEDDING, ComputePattern.GATHER, 0.90),
    
    # Copy/Transpose
    r"copy|memcpy|transpose|permute|contiguous": (KernelType.COPY, ComputePattern.TRANSPOSE, 0.85),
    r"cat|concat|stack|split|chunk": (KernelType.COPY, ComputePattern.TRANSPOSE, 0.80),
    
    # Convolution
    r"conv|convolution|cudnn.*conv|implicit_gemm": (KernelType.CONVOLUTION, ComputePattern.STENCIL, 0.90),
    r"winograd|fft_conv": (KernelType.CONVOLUTION, ComputePattern.MATMUL, 0.90),
    
    # Pooling
    r"pool|maxpool|avgpool|adaptive_pool": (KernelType.POOLING, ComputePattern.REDUCE, 0.90),
    
    # Scatter/Gather
    r"scatter|gather|index_select|index_add": (KernelType.SCATTER_GATHER, ComputePattern.SCATTER, 0.90),
}


def identify_kernel_type(kernel_name: str) -> Tuple[KernelType, ComputePattern, float]:
    """
    Identify kernel type from its name using pattern matching.
    
    Returns:
        (KernelType, ComputePattern, confidence)
    """
    kernel_lower = kernel_name.lower()
    
    best_match = (KernelType.UNKNOWN, ComputePattern.MAP, 0.0)
    
    for pattern, (ktype, cpattern, confidence) in KERNEL_PATTERNS.items():
        if re.search(pattern, kernel_lower):
            if confidence > best_match[2]:
                best_match = (ktype, cpattern, confidence)
    
    return best_match


def infer_memory_pattern(
    ld_efficiency: float,
    st_efficiency: float,
    l1_hit_rate: float,
    l2_hit_rate: float,
    kernel_type: KernelType,
) -> MemoryPattern:
    """
    Infer memory access pattern from efficiency metrics.
    """
    # High efficiency = coalesced access
    avg_efficiency = (ld_efficiency + st_efficiency) / 2 if st_efficiency > 0 else ld_efficiency
    
    # High cache hit = reuse
    has_reuse = l1_hit_rate > 50 or l2_hit_rate > 70
    
    if avg_efficiency > 80:
        if has_reuse:
            return MemoryPattern.COALESCED_REUSE
        return MemoryPattern.COALESCED_STREAMING
    elif avg_efficiency > 40:
        if kernel_type in (KernelType.GEMM, KernelType.CONVOLUTION):
            return MemoryPattern.BLOCKED
        return MemoryPattern.STRIDED
    else:
        return MemoryPattern.RANDOM


def fingerprint_kernel(
    kernel_name: str,
    metrics: Dict[str, float],
) -> KernelFingerprint:
    """
    Generate a complete fingerprint for a kernel.
    
    Args:
        kernel_name: Name of the kernel (from profiler)
        metrics: Dict of NCU metrics
    
    Returns:
        KernelFingerprint with type classification and optimization hints
    """
    # Identify type from name
    kernel_type, compute_pattern, confidence = identify_kernel_type(kernel_name)
    
    # Get metrics
    ld_efficiency = metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 50)
    st_efficiency = metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", 50)
    l1_hit_rate = metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct", 0)
    l2_hit_rate = metrics.get("lts__t_sectors_op_read_hit_rate.pct", 0)
    
    sm_throughput = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    dram_throughput = metrics.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                                   metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0))
    tensor_util = metrics.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0)
    
    # Infer memory pattern
    memory_pattern = infer_memory_pattern(
        ld_efficiency, st_efficiency, l1_hit_rate, l2_hit_rate, kernel_type
    )
    
    # Arithmetic intensity class
    if sm_throughput > 60 and dram_throughput < 40:
        ai_class = "high"
    elif dram_throughput > 60 and sm_throughput < 40:
        ai_class = "low"
    else:
        ai_class = "medium"
    
    # Parallelism type
    if kernel_type == KernelType.GEMM or tensor_util > 30:
        parallelism = "tensor"
    elif compute_pattern == ComputePattern.REDUCE:
        parallelism = "task"
    else:
        parallelism = "data"
    
    # Reuse opportunity
    if kernel_type in (KernelType.GEMM, KernelType.ATTENTION, KernelType.CONVOLUTION):
        reuse = "both"  # Spatial and temporal
    elif l2_hit_rate > 50:
        reuse = "temporal"
    elif memory_pattern == MemoryPattern.BLOCKED:
        reuse = "spatial"
    else:
        reuse = "none"
    
    # Optimization hints
    tensor_core_candidate = (
        kernel_type in (KernelType.GEMM, KernelType.ATTENTION, KernelType.CONVOLUTION)
        and tensor_util < 50  # Underutilized = room to improve
    )
    
    tiling_beneficial = (
        kernel_type in (KernelType.GEMM, KernelType.ATTENTION, KernelType.CONVOLUTION)
        or (ai_class == "medium" and reuse != "none")
    )
    
    fusion_candidate = (
        kernel_type in (KernelType.ELEMENTWISE, KernelType.LAYERNORM, KernelType.SOFTMAX)
        or (ai_class == "low" and kernel_type != KernelType.REDUCTION)
    )
    
    async_copy_candidate = (
        memory_pattern in (MemoryPattern.COALESCED_STREAMING, MemoryPattern.BLOCKED)
        and tiling_beneficial
    )
    
    # Pattern details
    details = _generate_pattern_details(kernel_type, compute_pattern, memory_pattern, metrics)
    
    return KernelFingerprint(
        kernel_name=kernel_name,
        kernel_type=kernel_type,
        compute_pattern=compute_pattern,
        memory_pattern=memory_pattern,
        arithmetic_intensity_class=ai_class,
        parallelism_type=parallelism,
        reuse_opportunity=reuse,
        tensor_core_candidate=tensor_core_candidate,
        tiling_beneficial=tiling_beneficial,
        fusion_candidate=fusion_candidate,
        async_copy_candidate=async_copy_candidate,
        pattern_confidence=confidence,
        pattern_details=details,
    )


def _generate_pattern_details(
    kernel_type: KernelType,
    compute_pattern: ComputePattern,
    memory_pattern: MemoryPattern,
    metrics: Dict[str, float],
) -> str:
    """Generate human-readable pattern details."""
    details = []
    
    if kernel_type == KernelType.GEMM:
        details.append("Matrix multiplication - O(N³) compute, O(N²) data")
        details.append("Best optimized with Tensor Cores and tiling")
    elif kernel_type == KernelType.ATTENTION:
        details.append("Attention mechanism - QKV matmuls + softmax + output")
        details.append("Consider FlashAttention or fused attention kernels")
    elif kernel_type == KernelType.ELEMENTWISE:
        details.append("Elementwise operation - 1:1 mapping, memory bound")
        details.append("Candidate for kernel fusion to reduce memory traffic")
    elif kernel_type == KernelType.REDUCTION:
        details.append("Reduction operation - tree-based parallel reduction")
        details.append("Optimize with warp shuffle and shared memory")
    elif kernel_type == KernelType.SOFTMAX:
        details.append("Softmax - max + exp + sum + div")
        details.append("Best with online/fused implementation")
    elif kernel_type == KernelType.LAYERNORM:
        details.append("Layer normalization - mean, variance, normalize")
        details.append("Can be fused with adjacent operations")
    
    return " | ".join(details)


def format_fingerprint(fp: KernelFingerprint) -> str:
    """Format fingerprint as a human-readable summary."""
    lines = [
        f"KERNEL: {fp.kernel_name}",
        f"  Type: {fp.kernel_type.value.upper()} (confidence: {fp.pattern_confidence:.0%})",
        f"  Compute pattern: {fp.compute_pattern.value}",
        f"  Memory pattern: {fp.memory_pattern.value}",
        f"  Arithmetic intensity: {fp.arithmetic_intensity_class}",
        f"  Parallelism: {fp.parallelism_type}",
        f"  Reuse opportunity: {fp.reuse_opportunity}",
        "",
        "Optimization Opportunities:",
    ]
    
    if fp.tensor_core_candidate:
        lines.append("  ✓ Tensor Core candidate - ensure aligned dimensions")
    if fp.tiling_beneficial:
        lines.append("  ✓ Tiling beneficial - implement blocked algorithm")
    if fp.fusion_candidate:
        lines.append("  ✓ Fusion candidate - merge with adjacent kernels")
    if fp.async_copy_candidate:
        lines.append("  ✓ Async copy candidate - overlap compute/memory")
    
    if fp.pattern_details:
        lines.extend(["", f"Pattern: {fp.pattern_details}"])
    
    return "\n".join(lines)


# =============================================================================
# OPTIMIZATION CHECKLIST GENERATOR
# =============================================================================

@dataclass
class OptimizationItem:
    """Single optimization checklist item."""
    category: str
    description: str
    priority: str  # "critical", "high", "medium", "low"
    effort: str    # "trivial", "easy", "moderate", "hard"
    expected_impact: str
    how_to_check: str
    how_to_fix: str


def generate_optimization_checklist(
    fingerprint: KernelFingerprint,
    metrics: Dict[str, float],
) -> List[OptimizationItem]:
    """
    Generate a prioritized optimization checklist based on kernel fingerprint.
    """
    checklist = []
    
    # Memory efficiency checks
    ld_efficiency = metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 100)
    if ld_efficiency < 50:
        checklist.append(OptimizationItem(
            category="Memory Access",
            description=f"Poor load coalescing ({ld_efficiency:.0f}%)",
            priority="critical",
            effort="moderate",
            expected_impact="2-4x for memory-bound kernels",
            how_to_check="NCU: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
            how_to_fix="Ensure consecutive threads access consecutive memory; use AoS->SoA transforms",
        ))
    
    # Bank conflicts
    bank_conflicts = metrics.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 0)
    if bank_conflicts > 1000:
        checklist.append(OptimizationItem(
            category="Shared Memory",
            description=f"Shared memory bank conflicts ({bank_conflicts:.0f})",
            priority="high",
            effort="easy",
            expected_impact="10-30% for shared memory heavy kernels",
            how_to_check="NCU: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            how_to_fix="Add padding to shared memory arrays, e.g., smem[N][M+1]",
        ))
    
    # Register spills
    local_ld = metrics.get("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", 0)
    local_st = metrics.get("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", 0)
    if local_ld + local_st > 0:
        checklist.append(OptimizationItem(
            category="Register Pressure",
            description="Register spills to local memory detected",
            priority="critical",
            effort="hard",
            expected_impact="Can be 10x+ for spill-heavy kernels",
            how_to_check="NCU: l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum > 0",
            how_to_fix="Reduce register pressure: smaller tiles, fewer temps, use __launch_bounds__",
        ))
    
    # Occupancy
    occupancy = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 100)
    if occupancy < 30:
        regs = metrics.get("launch__registers_per_thread", 0)
        checklist.append(OptimizationItem(
            category="Occupancy",
            description=f"Low occupancy ({occupancy:.0f}%), {regs:.0f} regs/thread",
            priority="high",
            effort="moderate",
            expected_impact="Improves latency hiding, 20-50% for latency-bound",
            how_to_check="NCU: sm__warps_active.avg.pct_of_peak_sustained_active",
            how_to_fix="Reduce registers/shared mem, or adjust block size",
        ))
    
    # Tensor core utilization
    tensor_util = metrics.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0)
    if fingerprint.tensor_core_candidate and tensor_util < 30:
        checklist.append(OptimizationItem(
            category="Tensor Cores",
            description=f"Tensor Core underutilized ({tensor_util:.0f}%)",
            priority="high",
            effort="easy",
            expected_impact="5-15x for GEMM/attention kernels",
            how_to_check="NCU: sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            how_to_fix="Ensure M,N,K divisible by 16 (FP16) or 32 (FP8); use cuBLAS or CUTLASS",
        ))
    
    # Fusion opportunity
    if fingerprint.fusion_candidate:
        sm_throughput = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        if sm_throughput < 30:
            checklist.append(OptimizationItem(
                category="Kernel Fusion",
                description="Low compute utilization - fusion candidate",
                priority="medium",
                effort="moderate",
                expected_impact="2-5x by reducing kernel launch and memory traffic",
                how_to_check="Multiple small kernels in trace, low SM utilization",
                how_to_fix="Use torch.compile(), write fused kernel, or use xFormers/FlashAttention",
            ))
    
    # Async copy
    if fingerprint.async_copy_candidate:
        memory_stall = metrics.get("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", 0)
        if memory_stall > 20:
            checklist.append(OptimizationItem(
                category="Memory Pipelining",
                description=f"High memory latency stalls ({memory_stall:.0f}%)",
                priority="medium",
                effort="hard",
                expected_impact="20-50% by hiding memory latency",
                how_to_check="NCU: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                how_to_fix="Use cp.async for software pipelining, double buffering",
            ))
    
    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    checklist.sort(key=lambda x: priority_order.get(x.priority, 4))
    
    return checklist


def format_checklist(checklist: List[OptimizationItem]) -> str:
    """Format checklist as a human-readable report."""
    if not checklist:
        return "✅ No major optimization opportunities identified - kernel looks well-optimized!"
    
    lines = [
        "=" * 70,
        "OPTIMIZATION CHECKLIST",
        "=" * 70,
    ]
    
    for i, item in enumerate(checklist, 1):
        priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        emoji = priority_emoji.get(item.priority, "⚪")
        
        lines.extend([
            "",
            f"{i}. {emoji} [{item.priority.upper()}] {item.description}",
            f"   Category: {item.category}",
            f"   Effort: {item.effort} | Expected impact: {item.expected_impact}",
            f"   Check: {item.how_to_check}",
            f"   Fix: {item.how_to_fix}",
        ])
    
    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "KernelType",
    "ComputePattern", 
    "MemoryPattern",
    "KernelFingerprint",
    "identify_kernel_type",
    "fingerprint_kernel",
    "format_fingerprint",
    "OptimizationItem",
    "generate_optimization_checklist",
    "format_checklist",
]






