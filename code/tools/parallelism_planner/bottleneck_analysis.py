"""
Bottleneck Analysis Module for Parallelism Planner

Identifies performance bottlenecks in distributed training:
- Compute-bound vs Memory-bound vs Communication-bound
- GPU utilization analysis
- Memory bandwidth analysis
- Scaling efficiency analysis
- What-if analysis for configuration changes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math


class BottleneckType(Enum):
    """Type of performance bottleneck."""
    COMPUTE = "compute"          # GPU compute is limiting factor
    MEMORY = "memory"            # GPU memory is limiting factor
    COMMUNICATION = "communication"  # Network/interconnect is limiting factor
    MEMORY_BANDWIDTH = "memory_bandwidth"  # HBM bandwidth is limiting factor
    CPU = "cpu"                  # CPU preprocessing is limiting factor
    IO = "io"                    # Data loading is limiting factor
    BALANCED = "balanced"        # No clear bottleneck


@dataclass
class BottleneckAnalysis:
    """Result of bottleneck analysis."""
    primary_bottleneck: BottleneckType
    secondary_bottleneck: Optional[BottleneckType]
    bottleneck_severity: float  # 0-1, how severe the bottleneck is
    
    # Utilization metrics (0-1)
    compute_utilization: float
    memory_utilization: float
    communication_overhead: float
    memory_bandwidth_utilization: float
    
    # Recommendations
    recommendations: List[str]
    
    # Detailed breakdown
    time_breakdown: Dict[str, float]  # Percentage of time in each phase
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_bottleneck": self.primary_bottleneck.value,
            "secondary_bottleneck": self.secondary_bottleneck.value if self.secondary_bottleneck else None,
            "bottleneck_severity": self.bottleneck_severity,
            "utilization": {
                "compute": self.compute_utilization,
                "memory": self.memory_utilization,
                "communication": self.communication_overhead,
                "memory_bandwidth": self.memory_bandwidth_utilization,
            },
            "recommendations": self.recommendations,
            "time_breakdown": self.time_breakdown,
        }


@dataclass
class ScalingAnalysis:
    """Analysis of scaling efficiency."""
    current_gpus: int
    current_throughput: float
    
    # Scaling projections
    scaling_projections: List[Dict[str, Any]]  # [{gpus, throughput, efficiency}]
    
    # Efficiency metrics
    strong_scaling_efficiency: float  # How well it scales with more GPUs (same problem)
    weak_scaling_efficiency: float    # How well it scales with proportionally larger problem
    
    # Optimal scale
    optimal_gpu_count: int
    optimal_efficiency: float
    
    # Recommendations
    scaling_recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_gpus": self.current_gpus,
            "current_throughput": self.current_throughput,
            "scaling_projections": self.scaling_projections,
            "strong_scaling_efficiency": self.strong_scaling_efficiency,
            "weak_scaling_efficiency": self.weak_scaling_efficiency,
            "optimal_gpu_count": self.optimal_gpu_count,
            "optimal_efficiency": self.optimal_efficiency,
            "scaling_recommendations": self.scaling_recommendations,
        }


@dataclass
class WhatIfResult:
    """Result of what-if analysis."""
    scenario: str
    changes: Dict[str, Any]
    
    # Projected metrics
    projected_throughput: float
    projected_memory_per_gpu: float
    projected_efficiency: float
    
    # Comparison with current
    throughput_change_pct: float
    memory_change_pct: float
    efficiency_change_pct: float
    
    # Feasibility
    is_feasible: bool
    feasibility_issues: List[str]
    
    # Recommendations
    trade_offs: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "changes": self.changes,
            "projected": {
                "throughput": self.projected_throughput,
                "memory_per_gpu": self.projected_memory_per_gpu,
                "efficiency": self.projected_efficiency,
            },
            "change_vs_current": {
                "throughput_pct": self.throughput_change_pct,
                "memory_pct": self.memory_change_pct,
                "efficiency_pct": self.efficiency_change_pct,
            },
            "is_feasible": self.is_feasible,
            "feasibility_issues": self.feasibility_issues,
            "trade_offs": self.trade_offs,
        }


class BottleneckDetector:
    """Detects performance bottlenecks in distributed training."""
    
    # Hardware specs for analysis
    GPU_SPECS = {
        "h100": {"tflops_bf16": 1979, "memory_gb": 80, "bandwidth_gbps": 3350, "nvlink_gbps": 900},
        "h200": {"tflops_bf16": 1979, "memory_gb": 141, "bandwidth_gbps": 4800, "nvlink_gbps": 900},
        "b100": {"tflops_bf16": 3500, "memory_gb": 192, "bandwidth_gbps": 8000, "nvlink_gbps": 1800},
        "b200": {"tflops_bf16": 4500, "memory_gb": 192, "bandwidth_gbps": 8000, "nvlink_gbps": 1800},
        "a100": {"tflops_bf16": 312, "memory_gb": 80, "bandwidth_gbps": 2039, "nvlink_gbps": 600},
    }
    
    def __init__(self, gpu_type: str = "h100"):
        self.gpu_type = gpu_type.lower()
        self.gpu_spec = self.GPU_SPECS.get(self.gpu_type, self.GPU_SPECS["h100"])
    
    def analyze(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        num_gpus: int,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        hidden_size: int = 8192,
        num_layers: int = 80,
        has_nvlink: bool = True,
        measured_throughput_tps: Optional[float] = None,
    ) -> BottleneckAnalysis:
        """
        Analyze performance bottlenecks.
        
        Args:
            model_params_b: Model parameters in billions
            batch_size: Global batch size
            seq_length: Sequence length
            num_gpus: Total number of GPUs
            tp/pp/dp: Parallelism configuration
            hidden_size: Model hidden size
            num_layers: Number of transformer layers
            has_nvlink: Whether NVLink is available
            measured_throughput_tps: Actual measured throughput (optional)
        """
        
        # Calculate theoretical peak
        flops_per_token = self._estimate_flops_per_token(model_params_b, seq_length)
        peak_tflops = self.gpu_spec["tflops_bf16"] * num_gpus
        theoretical_max_tps = (peak_tflops * 1e12) / flops_per_token
        
        # Estimate actual throughput if not provided
        if measured_throughput_tps is None:
            # Estimate based on typical efficiencies
            base_efficiency = 0.45  # ~45% MFU is typical
            tp_overhead = 0.95 ** (tp - 1)  # ~5% overhead per TP step
            pp_overhead = 1 - ((pp - 1) / (pp * 4))  # Pipeline bubble
            comm_overhead = 0.98 if has_nvlink else 0.85
            
            estimated_efficiency = base_efficiency * tp_overhead * pp_overhead * comm_overhead
            measured_throughput_tps = theoretical_max_tps * estimated_efficiency
        
        # Calculate utilization metrics
        compute_utilization = measured_throughput_tps / theoretical_max_tps
        
        # Memory analysis
        memory_per_gpu = self._estimate_memory_per_gpu(
            model_params_b, batch_size, seq_length, tp, pp, dp
        )
        memory_utilization = memory_per_gpu / self.gpu_spec["memory_gb"]
        
        # Communication overhead
        comm_volume_gb = self._estimate_communication_volume(
            model_params_b, batch_size, seq_length, tp, pp, dp
        )
        available_bandwidth = self.gpu_spec["nvlink_gbps"] if has_nvlink else 50  # PCIe
        comm_time_fraction = comm_volume_gb / (available_bandwidth * 1e-3)  # Rough estimate
        communication_overhead = min(1.0, comm_time_fraction * 0.1)  # Normalize
        
        # Memory bandwidth analysis
        bytes_per_token = model_params_b * 1e9 * 2 / tp  # BF16, divided by TP
        memory_bandwidth_needed_gbps = (measured_throughput_tps * bytes_per_token) / 1e9
        memory_bandwidth_utilization = memory_bandwidth_needed_gbps / self.gpu_spec["bandwidth_gbps"]
        
        # Determine bottleneck
        bottleneck_scores = {
            BottleneckType.COMPUTE: compute_utilization,
            BottleneckType.MEMORY: memory_utilization,
            BottleneckType.COMMUNICATION: communication_overhead,
            BottleneckType.MEMORY_BANDWIDTH: memory_bandwidth_utilization,
        }
        
        # Primary bottleneck is highest utilization
        sorted_bottlenecks = sorted(bottleneck_scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_bottlenecks[0][0]
        secondary = sorted_bottlenecks[1][0] if sorted_bottlenecks[1][1] > 0.5 else None
        severity = sorted_bottlenecks[0][1]
        
        # If all utilizations are low, check for balanced or other issues
        if max(bottleneck_scores.values()) < 0.4:
            primary = BottleneckType.BALANCED
            severity = 0.3
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            primary, secondary, bottleneck_scores, tp, pp, dp, has_nvlink
        )
        
        # Time breakdown
        compute_time = compute_utilization * 0.6
        comm_time = communication_overhead * 0.25
        other_time = 1 - compute_time - comm_time
        
        time_breakdown = {
            "compute": compute_time,
            "communication": comm_time,
            "memory_access": memory_bandwidth_utilization * 0.1,
            "other": max(0, other_time - memory_bandwidth_utilization * 0.1),
        }
        
        return BottleneckAnalysis(
            primary_bottleneck=primary,
            secondary_bottleneck=secondary,
            bottleneck_severity=severity,
            compute_utilization=compute_utilization,
            memory_utilization=memory_utilization,
            communication_overhead=communication_overhead,
            memory_bandwidth_utilization=memory_bandwidth_utilization,
            recommendations=recommendations,
            time_breakdown=time_breakdown,
        )
    
    def _estimate_flops_per_token(self, model_params_b: float, seq_length: int) -> float:
        """Estimate FLOPs per token for a transformer model."""
        # Rough estimate: 6 * params for forward, 12 * params for forward+backward
        # Plus attention: O(seq_length * hidden^2)
        return 6 * model_params_b * 1e9
    
    def _estimate_memory_per_gpu(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        tp: int,
        pp: int,
        dp: int,
    ) -> float:
        """Estimate memory per GPU in GB."""
        # Model parameters (BF16)
        param_memory = (model_params_b * 1e9 * 2) / (tp * pp) / 1e9  # GB
        
        # Optimizer states (8 bytes per param for AdamW)
        optimizer_memory = (model_params_b * 1e9 * 8) / (tp * pp * dp) / 1e9
        
        # Activations (rough estimate)
        activation_memory = (batch_size / dp) * seq_length * 8192 * 2 / 1e9  # Rough
        
        return param_memory + optimizer_memory + activation_memory
    
    def _estimate_communication_volume(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        tp: int,
        pp: int,
        dp: int,
    ) -> float:
        """Estimate communication volume in GB per step."""
        comm_volume = 0
        
        # TP: all-reduce for each layer
        if tp > 1:
            comm_volume += model_params_b * 1e9 * 2 / 1e9 * 2  # Forward + backward
        
        # PP: point-to-point activations
        if pp > 1:
            comm_volume += batch_size * seq_length * 8192 * 2 / 1e9
        
        # DP: gradient all-reduce
        if dp > 1:
            comm_volume += model_params_b * 1e9 * 2 / 1e9
        
        return comm_volume
    
    def _generate_recommendations(
        self,
        primary: BottleneckType,
        secondary: Optional[BottleneckType],
        scores: Dict[BottleneckType, float],
        tp: int,
        pp: int,
        dp: int,
        has_nvlink: bool,
    ) -> List[str]:
        """Generate recommendations based on bottleneck analysis."""
        recommendations = []
        
        if primary == BottleneckType.COMPUTE:
            recommendations.append("✓ Compute-bound is ideal - you're utilizing GPU compute well")
            if scores[BottleneckType.COMPUTE] < 0.5:
                recommendations.append("Consider increasing batch size to improve compute utilization")
        
        elif primary == BottleneckType.MEMORY:
            recommendations.append("Memory-bound: Consider these optimizations:")
            recommendations.append("  • Enable gradient checkpointing to reduce activation memory")
            recommendations.append("  • Use ZeRO-3 or FSDP to shard optimizer states")
            recommendations.append("  • Reduce batch size or sequence length")
            if tp == 1:
                recommendations.append("  • Increase tensor parallelism to distribute model")
        
        elif primary == BottleneckType.COMMUNICATION:
            recommendations.append("Communication-bound: Consider these optimizations:")
            if not has_nvlink:
                recommendations.append("  • Use NVLink-connected GPUs for lower latency")
            if tp > 4:
                recommendations.append("  • Reduce tensor parallelism (high all-reduce overhead)")
            if dp > 1:
                recommendations.append("  • Enable gradient compression for data parallel")
                recommendations.append("  • Increase gradient accumulation to reduce sync frequency")
            recommendations.append("  • Enable communication-computation overlap")
        
        elif primary == BottleneckType.MEMORY_BANDWIDTH:
            recommendations.append("Memory bandwidth-bound: Consider these optimizations:")
            recommendations.append("  • Enable Flash Attention to reduce memory access")
            recommendations.append("  • Use fused kernels (fused LayerNorm, fused GELU)")
            recommendations.append("  • Consider FP8 for reduced memory traffic on Hopper+")
        
        elif primary == BottleneckType.BALANCED:
            recommendations.append("No clear bottleneck - configuration appears balanced")
            recommendations.append("Consider these general optimizations:")
            recommendations.append("  • Profile with nsys to identify hidden bottlenecks")
            recommendations.append("  • Check for CPU data loading bottlenecks")
        
        return recommendations


class ScalingAnalyzer:
    """Analyzes scaling efficiency for distributed training."""
    
    def __init__(self, gpu_type: str = "h100"):
        self.gpu_type = gpu_type
        self.bottleneck_detector = BottleneckDetector(gpu_type)
    
    def analyze_scaling(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        current_gpus: int,
        current_throughput_tps: float,
        max_gpus: int = 512,
        has_nvlink: bool = True,
    ) -> ScalingAnalysis:
        """
        Analyze how training scales with more GPUs.
        """
        
        projections = []
        
        # Project for various GPU counts
        gpu_counts = [current_gpus]
        scale = current_gpus
        while scale * 2 <= max_gpus:
            scale *= 2
            gpu_counts.append(scale)
        
        base_efficiency = current_throughput_tps / current_gpus  # Per-GPU throughput
        
        for gpus in gpu_counts:
            scale_factor = gpus / current_gpus
            
            # Communication overhead increases with scale
            comm_overhead = 1 - (0.02 * math.log2(max(1, gpus / 8)))  # ~2% per doubling after 8
            
            # Pipeline bubble overhead for large scale
            pp_stages = max(1, gpus // 64)  # Assume PP increases at large scale
            bubble_overhead = 1 - ((pp_stages - 1) / (pp_stages * 4)) if pp_stages > 1 else 1
            
            # Projected efficiency
            efficiency = base_efficiency * comm_overhead * bubble_overhead
            projected_throughput = efficiency * gpus
            
            projections.append({
                "gpus": gpus,
                "throughput_tps": projected_throughput,
                "efficiency": efficiency / base_efficiency,  # Relative to baseline
                "per_gpu_throughput": efficiency,
            })
        
        # Calculate scaling efficiencies
        if len(projections) > 1:
            strong_scaling = projections[-1]["throughput_tps"] / (
                projections[0]["throughput_tps"] * (projections[-1]["gpus"] / projections[0]["gpus"])
            )
        else:
            strong_scaling = 1.0
        
        weak_scaling = projections[-1]["efficiency"] if projections else 1.0
        
        # Find optimal scale (where efficiency drops below 80%)
        optimal_idx = 0
        for i, proj in enumerate(projections):
            if proj["efficiency"] >= 0.8:
                optimal_idx = i
        
        optimal_gpus = projections[optimal_idx]["gpus"]
        optimal_efficiency = projections[optimal_idx]["efficiency"]
        
        # Recommendations
        recommendations = []
        if strong_scaling < 0.7:
            recommendations.append("Strong scaling is limited - communication overhead is significant")
            recommendations.append("Consider using pipeline parallelism for large scale")
        if optimal_gpus < max_gpus:
            recommendations.append(f"Optimal scale is ~{optimal_gpus} GPUs (80%+ efficiency)")
            recommendations.append(f"Beyond {optimal_gpus} GPUs, diminishing returns")
        if weak_scaling > 0.9:
            recommendations.append("Good weak scaling - consider increasing batch size with more GPUs")
        
        return ScalingAnalysis(
            current_gpus=current_gpus,
            current_throughput=current_throughput_tps,
            scaling_projections=projections,
            strong_scaling_efficiency=strong_scaling,
            weak_scaling_efficiency=weak_scaling,
            optimal_gpu_count=optimal_gpus,
            optimal_efficiency=optimal_efficiency,
            scaling_recommendations=recommendations,
        )


class WhatIfAnalyzer:
    """Performs what-if analysis for configuration changes."""
    
    def __init__(self, gpu_type: str = "h100", gpu_memory_gb: float = 80):
        self.gpu_type = gpu_type
        self.gpu_memory_gb = gpu_memory_gb
        self.bottleneck_detector = BottleneckDetector(gpu_type)
    
    def analyze(
        self,
        current_config: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
    ) -> List[WhatIfResult]:
        """
        Analyze what-if scenarios.
        
        Args:
            current_config: Current configuration
            scenarios: List of scenario changes to analyze
        
        Returns:
            List of WhatIfResult for each scenario
        """
        
        results = []
        
        for scenario in scenarios:
            result = self._analyze_scenario(current_config, scenario)
            results.append(result)
        
        return results
    
    def _analyze_scenario(
        self,
        current: Dict[str, Any],
        changes: Dict[str, Any],
    ) -> WhatIfResult:
        """Analyze a single what-if scenario."""
        
        # Apply changes to current config
        new_config = {**current, **changes}
        
        # Extract parameters
        model_params_b = new_config.get("model_params_b", 70)
        batch_size = new_config.get("batch_size", 8)
        seq_length = new_config.get("seq_length", 4096)
        num_gpus = new_config.get("num_gpus", 8)
        tp = new_config.get("tp", 1)
        pp = new_config.get("pp", 1)
        dp = new_config.get("dp", 8)
        
        # Calculate memory
        param_mem = (model_params_b * 1e9 * 2) / (tp * pp) / 1e9
        optimizer_mem = (model_params_b * 1e9 * 8) / (tp * pp * dp) / 1e9
        activation_mem = (batch_size / dp) * seq_length * 8192 * 2 / 1e9
        total_mem = param_mem + optimizer_mem + activation_mem
        
        # Check feasibility
        is_feasible = True
        issues = []
        
        if total_mem > self.gpu_memory_gb:
            is_feasible = False
            issues.append(f"Memory ({total_mem:.1f}GB) exceeds GPU capacity ({self.gpu_memory_gb}GB)")
        
        if tp * pp * dp != num_gpus:
            is_feasible = False
            issues.append(f"TP×PP×DP ({tp}×{pp}×{dp}={tp*pp*dp}) doesn't match GPU count ({num_gpus})")
        
        # Estimate throughput
        base_tflops = 1979 if "h100" in self.gpu_type else 4500  # H100 vs B200
        peak_throughput = (base_tflops * 1e12 * num_gpus) / (6 * model_params_b * 1e9)
        
        # Apply efficiency factors
        efficiency = 0.45  # Base MFU
        efficiency *= 0.95 ** (tp - 1)  # TP overhead
        efficiency *= (1 - (pp - 1) / (pp * 4)) if pp > 1 else 1  # PP bubble
        
        projected_throughput = peak_throughput * efficiency
        
        # Compare with current
        current_mem = self._estimate_memory(current)
        current_throughput = current.get("throughput_tps", projected_throughput * 0.9)
        
        throughput_change = (projected_throughput - current_throughput) / current_throughput * 100
        memory_change = (total_mem - current_mem) / current_mem * 100 if current_mem > 0 else 0
        efficiency_change = 0  # Would need more info
        
        # Trade-offs
        trade_offs = []
        if "tp" in changes and changes["tp"] > current.get("tp", 1):
            trade_offs.append("Higher TP increases communication but reduces memory per GPU")
        if "batch_size" in changes and changes["batch_size"] > current.get("batch_size", 1):
            trade_offs.append("Larger batch improves throughput but uses more memory")
        if "pp" in changes and changes["pp"] > current.get("pp", 1):
            trade_offs.append("Pipeline parallelism adds bubble overhead but enables larger models")
        
        # Generate scenario name
        scenario_parts = []
        for key, value in changes.items():
            scenario_parts.append(f"{key}={value}")
        scenario_name = ", ".join(scenario_parts)
        
        return WhatIfResult(
            scenario=scenario_name,
            changes=changes,
            projected_throughput=projected_throughput,
            projected_memory_per_gpu=total_mem,
            projected_efficiency=efficiency,
            throughput_change_pct=throughput_change,
            memory_change_pct=memory_change,
            efficiency_change_pct=efficiency_change,
            is_feasible=is_feasible,
            feasibility_issues=issues,
            trade_offs=trade_offs,
        )
    
    def _estimate_memory(self, config: Dict[str, Any]) -> float:
        """Estimate memory for a configuration."""
        model_params_b = config.get("model_params_b", 70)
        batch_size = config.get("batch_size", 8)
        seq_length = config.get("seq_length", 4096)
        tp = config.get("tp", 1)
        pp = config.get("pp", 1)
        dp = config.get("dp", 8)
        
        param_mem = (model_params_b * 1e9 * 2) / (tp * pp) / 1e9
        optimizer_mem = (model_params_b * 1e9 * 8) / (tp * pp * dp) / 1e9
        activation_mem = (batch_size / dp) * seq_length * 8192 * 2 / 1e9
        
        return param_mem + optimizer_mem + activation_mem
    
    def generate_common_scenarios(
        self,
        current_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate common what-if scenarios to analyze."""
        
        scenarios = []
        
        current_gpus = current_config.get("num_gpus", 8)
        current_tp = current_config.get("tp", 1)
        current_pp = current_config.get("pp", 1)
        current_batch = current_config.get("batch_size", 8)
        
        # Double the GPUs
        if current_gpus < 512:
            scenarios.append({
                "name": "Double GPUs",
                "num_gpus": current_gpus * 2,
                "dp": current_config.get("dp", 8) * 2,
            })
        
        # Increase TP
        if current_tp < 8:
            scenarios.append({
                "name": "Increase TP",
                "tp": min(8, current_tp * 2),
                "dp": max(1, current_config.get("dp", 8) // 2),
            })
        
        # Add pipeline parallelism
        if current_pp == 1:
            scenarios.append({
                "name": "Add PP=2",
                "pp": 2,
                "dp": max(1, current_config.get("dp", 8) // 2),
            })
        
        # Double batch size
        scenarios.append({
            "name": "Double batch size",
            "batch_size": current_batch * 2,
        })
        
        # Halve batch size (for memory)
        if current_batch > 1:
            scenarios.append({
                "name": "Halve batch size",
                "batch_size": max(1, current_batch // 2),
            })
        
        return scenarios


def analyze_bottlenecks(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    parallelism_config: Dict[str, Any],
    measured_throughput_tps: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyze performance bottlenecks for a configuration.
    
    Returns comprehensive bottleneck analysis.
    """
    
    detector = BottleneckDetector(
        gpu_type=hardware_config.get("gpu_type", "h100")
    )
    
    analysis = detector.analyze(
        model_params_b=model_config.get("parameters_billions", 70),
        batch_size=model_config.get("batch_size", 8),
        seq_length=model_config.get("max_sequence_length", 4096),
        num_gpus=hardware_config.get("num_gpus", 8),
        tp=parallelism_config.get("tensor_parallel", 1),
        pp=parallelism_config.get("pipeline_parallel", 1),
        dp=parallelism_config.get("data_parallel", 8),
        hidden_size=model_config.get("hidden_size", 8192),
        num_layers=model_config.get("num_layers", 80),
        has_nvlink=hardware_config.get("has_nvlink", True),
        measured_throughput_tps=measured_throughput_tps,
    )
    
    return analysis.to_dict()


def analyze_scaling(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    current_throughput_tps: float,
    max_gpus: int = 512,
) -> Dict[str, Any]:
    """
    Analyze scaling efficiency for a configuration.
    """
    
    analyzer = ScalingAnalyzer(
        gpu_type=hardware_config.get("gpu_type", "h100")
    )
    
    analysis = analyzer.analyze_scaling(
        model_params_b=model_config.get("parameters_billions", 70),
        batch_size=model_config.get("batch_size", 8),
        seq_length=model_config.get("max_sequence_length", 4096),
        current_gpus=hardware_config.get("num_gpus", 8),
        current_throughput_tps=current_throughput_tps,
        max_gpus=max_gpus,
        has_nvlink=hardware_config.get("has_nvlink", True),
    )
    
    return analysis.to_dict()


def analyze_whatif(
    current_config: Dict[str, Any],
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Perform what-if analysis for configuration changes.
    """
    
    analyzer = WhatIfAnalyzer(
        gpu_type=current_config.get("gpu_type", "h100"),
        gpu_memory_gb=current_config.get("gpu_memory_gb", 80),
    )
    
    # Generate common scenarios if not provided
    if scenarios is None:
        scenarios = analyzer.generate_common_scenarios(current_config)
    
    results = analyzer.analyze(current_config, scenarios)
    
    return {
        "current_config": current_config,
        "scenarios": [r.to_dict() for r in results],
        "summary": {
            "feasible_scenarios": sum(1 for r in results if r.is_feasible),
            "best_throughput_scenario": max(results, key=lambda r: r.projected_throughput if r.is_feasible else 0).scenario,
            "best_efficiency_scenario": max(results, key=lambda r: r.projected_efficiency if r.is_feasible else 0).scenario,
        }
    }



