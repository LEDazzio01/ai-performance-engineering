"""
Core Analysis Module

Advanced GPU performance analysis tools including:
- Bottleneck classification and diagnosis
- Stall analysis and categorization
- Kernel fingerprinting and pattern recognition
- Optimization recommendation engine

NOTE: Profile comparison is handled by `core.perf_core_base.compare_profiles()`
which integrates the metric-level analysis from this module.
"""

from core.analysis.gpu_bottleneck_analyzer import (
    # GPU Specs
    GPUSpecs,
    GPU_SPECS,
    get_gpu_specs,
    # Bottleneck Analysis
    BottleneckType,
    StallCategory,
    StallAnalysis,
    BottleneckDiagnosis,
    analyze_stalls,
    diagnose_bottleneck,
    format_diagnosis_report,
    # Profile Comparison Data Classes (for integration)
    ProfileComparison,
    MetricDelta,
    format_comparison_report,
    # Quick Helpers
    quick_diagnosis,
    calculate_arithmetic_intensity,
    calculate_roofline_position,
    # Internal helper for perf_core_base integration
    _diff_metrics,
)

from core.analysis.kernel_fingerprint import (
    # Types
    KernelType,
    ComputePattern,
    MemoryPattern,
    KernelFingerprint,
    # Functions
    identify_kernel_type,
    fingerprint_kernel,
    format_fingerprint,
    # Checklist
    OptimizationItem,
    generate_optimization_checklist,
    format_checklist,
)

__all__ = [
    # GPU Specs
    "GPUSpecs",
    "GPU_SPECS", 
    "get_gpu_specs",
    # Bottleneck Analysis
    "BottleneckType",
    "StallCategory",
    "StallAnalysis",
    "BottleneckDiagnosis",
    "analyze_stalls",
    "diagnose_bottleneck",
    "format_diagnosis_report",
    # Profile Comparison Data Classes
    # NOTE: Use core.perf_core_base.compare_profiles() for full comparison
    "ProfileComparison",
    "MetricDelta",
    "format_comparison_report",
    # Quick Helpers
    "quick_diagnosis",
    "calculate_arithmetic_intensity",
    "calculate_roofline_position",
    # Kernel Fingerprinting
    "KernelType",
    "ComputePattern",
    "MemoryPattern",
    "KernelFingerprint",
    "identify_kernel_type",
    "fingerprint_kernel",
    "format_fingerprint",
    # Optimization Checklist
    "OptimizationItem",
    "generate_optimization_checklist",
    "format_checklist",
]

