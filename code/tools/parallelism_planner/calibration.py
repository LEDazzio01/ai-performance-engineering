#!/usr/bin/env python3
"""
Profiling-Based Calibration

Uses actual benchmark data to calibrate and improve parallelism estimates.
Loads historical benchmark results to refine throughput and memory predictions.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class BenchmarkDataPoint:
    """A single benchmark measurement."""
    
    # Configuration
    model_name: str
    model_params_b: float
    batch_size: int
    seq_length: int
    num_gpus: int
    tp_size: int
    pp_size: int
    dp_size: int
    
    # Measurements
    throughput_tps: float  # Tokens per second
    latency_ms: float
    memory_used_gb: float
    memory_peak_gb: float
    
    # Hardware context
    gpu_name: str
    gpu_memory_gb: float
    
    # Optional details
    precision: str = "bf16"
    sharding_strategy: str = "none"
    activation_checkpointing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params_b": self.model_params_b,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "num_gpus": self.num_gpus,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "dp_size": self.dp_size,
            "throughput_tps": self.throughput_tps,
            "latency_ms": self.latency_ms,
            "memory_used_gb": self.memory_used_gb,
            "memory_peak_gb": self.memory_peak_gb,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "precision": self.precision,
            "sharding_strategy": self.sharding_strategy,
            "activation_checkpointing": self.activation_checkpointing,
        }


@dataclass
class CalibrationModel:
    """Calibrated model for throughput/memory estimation."""
    
    # Base coefficients
    throughput_coeff: float = 1.0  # Multiplier for base estimate
    memory_coeff: float = 1.0      # Multiplier for base estimate
    
    # Scaling factors
    tp_efficiency: Dict[int, float] = field(default_factory=dict)  # TP size -> efficiency
    pp_efficiency: Dict[int, float] = field(default_factory=dict)  # PP size -> efficiency
    dp_efficiency: Dict[int, float] = field(default_factory=dict)  # DP size -> efficiency
    
    # Architecture-specific adjustments
    architecture_factors: Dict[str, float] = field(default_factory=dict)
    
    # Data points used for calibration
    num_data_points: int = 0
    calibration_error: float = 0.0  # Mean absolute percentage error
    
    def estimate_throughput(
        self,
        base_estimate: float,
        tp: int,
        pp: int,
        dp: int,
        architecture: str = "",
    ) -> float:
        """Apply calibration to base throughput estimate."""
        tp_eff = self.tp_efficiency.get(tp, 1.0 - 0.05 * (tp - 1))  # Default 5% loss per TP rank
        pp_eff = self.pp_efficiency.get(pp, 1.0 - 0.15 * (pp - 1))  # Default 15% loss per PP stage
        dp_eff = self.dp_efficiency.get(dp, 1.0)  # DP typically scales linearly
        
        arch_factor = self.architecture_factors.get(architecture, 1.0)
        
        return base_estimate * self.throughput_coeff * tp_eff * pp_eff * dp_eff * arch_factor
    
    def estimate_memory(
        self,
        base_estimate: float,
        architecture: str = "",
    ) -> float:
        """Apply calibration to base memory estimate."""
        arch_factor = self.architecture_factors.get(architecture, 1.0)
        return base_estimate * self.memory_coeff * arch_factor
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "throughput_coeff": self.throughput_coeff,
            "memory_coeff": self.memory_coeff,
            "tp_efficiency": self.tp_efficiency,
            "pp_efficiency": self.pp_efficiency,
            "dp_efficiency": self.dp_efficiency,
            "architecture_factors": self.architecture_factors,
            "num_data_points": self.num_data_points,
            "calibration_error": self.calibration_error,
        }


class CalibrationEngine:
    """Engine for calibrating estimates from benchmark data."""
    
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).parent.parent.parent
        self.data_points: List[BenchmarkDataPoint] = []
        self.calibration_model: Optional[CalibrationModel] = None
    
    def load_benchmark_data(
        self,
        benchmark_dirs: Optional[List[Path]] = None,
    ) -> int:
        """Load benchmark data from result files.
        
        Args:
            benchmark_dirs: Directories to search for benchmark results
            
        Returns:
            Number of data points loaded
        """
        if benchmark_dirs is None:
            benchmark_dirs = [
                self.repo_root / "artifacts",
                self.repo_root / "benchmark_profiles",
            ]
        
        loaded = 0
        
        for dir_path in benchmark_dirs:
            if not dir_path.exists():
                continue
            
            # Find benchmark result files
            for result_file in dir_path.glob("**/benchmark_test_results.json"):
                loaded += self._load_file(result_file)
            
            for result_file in dir_path.glob("**/results.json"):
                loaded += self._load_file(result_file)
        
        self.data_points.sort(key=lambda x: x.throughput_tps, reverse=True)
        return loaded
    
    def _load_file(self, path: Path) -> int:
        """Load benchmark data from a single file."""
        try:
            with open(path) as f:
                data = json.load(f)
            
            loaded = 0
            benchmarks = data.get("benchmarks", [])
            
            for b in benchmarks:
                point = self._parse_benchmark(b)
                if point:
                    self.data_points.append(point)
                    loaded += 1
            
            return loaded
        except Exception:
            return 0
    
    def _parse_benchmark(self, b: Dict[str, Any]) -> Optional[BenchmarkDataPoint]:
        """Parse a benchmark entry into a data point."""
        try:
            # Extract model info
            model_name = b.get("name", b.get("chapter", "unknown"))
            
            # Try to extract params from name
            params_match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_name)
            model_params_b = float(params_match.group(1)) if params_match else 7.0
            
            # Extract performance
            throughput = b.get("throughput_tps", b.get("tokens_per_second", 0))
            latency = b.get("mean_time_ms", b.get("latency_ms", 0))
            
            # Skip invalid entries
            if throughput <= 0 and latency <= 0:
                return None
            
            # Extract memory
            memory_used = b.get("memory_gb", b.get("memory_used_gb", 0))
            memory_peak = b.get("peak_memory_gb", memory_used)
            
            # Extract config
            config = b.get("config", {})
            batch_size = config.get("batch_size", b.get("batch_size", 1))
            seq_length = config.get("seq_length", b.get("seq_length", 2048))
            
            # Parallelism config
            tp = config.get("tp", config.get("tensor_parallel", 1))
            pp = config.get("pp", config.get("pipeline_parallel", 1))
            dp = config.get("dp", config.get("data_parallel", 1))
            num_gpus = tp * pp * dp
            
            # GPU info
            gpu_name = b.get("gpu_name", "Unknown")
            gpu_memory = b.get("gpu_memory_gb", 80)
            
            return BenchmarkDataPoint(
                model_name=model_name,
                model_params_b=model_params_b,
                batch_size=batch_size,
                seq_length=seq_length,
                num_gpus=num_gpus,
                tp_size=tp,
                pp_size=pp,
                dp_size=dp,
                throughput_tps=throughput,
                latency_ms=latency,
                memory_used_gb=memory_used,
                memory_peak_gb=memory_peak,
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_memory,
                precision=config.get("precision", "bf16"),
                sharding_strategy=config.get("sharding", "none"),
                activation_checkpointing=config.get("activation_checkpointing", False),
            )
        except Exception:
            return None
    
    def calibrate(self) -> CalibrationModel:
        """Calibrate model from loaded data points.
        
        Returns:
            Calibrated model for estimation
        """
        if not self.data_points:
            # Return default model
            return CalibrationModel(
                throughput_coeff=1.0,
                memory_coeff=1.1,  # Add 10% safety margin
                tp_efficiency={1: 1.0, 2: 0.95, 4: 0.88, 8: 0.80},
                pp_efficiency={1: 1.0, 2: 0.85, 4: 0.72, 8: 0.60},
                dp_efficiency={1: 1.0, 2: 0.98, 4: 0.95, 8: 0.92},
            )
        
        # Group by parallelism configuration
        tp_samples: Dict[int, List[float]] = {}
        pp_samples: Dict[int, List[float]] = {}
        dp_samples: Dict[int, List[float]] = {}
        
        throughput_ratios = []
        memory_ratios = []
        
        for point in self.data_points:
            # Track efficiency by parallelism type
            # Normalize by model size and batch size
            normalized_throughput = point.throughput_tps / (point.batch_size * point.seq_length)
            
            if point.tp_size not in tp_samples:
                tp_samples[point.tp_size] = []
            tp_samples[point.tp_size].append(normalized_throughput)
            
            if point.pp_size not in pp_samples:
                pp_samples[point.pp_size] = []
            pp_samples[point.pp_size].append(normalized_throughput)
            
            if point.dp_size not in dp_samples:
                dp_samples[point.dp_size] = []
            dp_samples[point.dp_size].append(normalized_throughput)
        
        # Calculate efficiency factors
        def calc_efficiency(samples: Dict[int, List[float]]) -> Dict[int, float]:
            if not samples or 1 not in samples:
                return {}
            
            base = sum(samples[1]) / len(samples[1]) if samples[1] else 1.0
            efficiency = {}
            for size, values in samples.items():
                avg = sum(values) / len(values)
                efficiency[size] = avg / base if base > 0 else 1.0
            return efficiency
        
        tp_eff = calc_efficiency(tp_samples)
        pp_eff = calc_efficiency(pp_samples)
        dp_eff = calc_efficiency(dp_samples)
        
        # Default values for missing sizes
        for tp in [1, 2, 4, 8]:
            if tp not in tp_eff:
                tp_eff[tp] = 1.0 - 0.05 * (tp - 1)
        for pp in [1, 2, 4, 8]:
            if pp not in pp_eff:
                pp_eff[pp] = 1.0 - 0.15 * (pp - 1)
        for dp in [1, 2, 4, 8]:
            if dp not in dp_eff:
                dp_eff[dp] = 1.0 - 0.02 * (dp - 1)
        
        model = CalibrationModel(
            throughput_coeff=1.0,
            memory_coeff=1.1,
            tp_efficiency=tp_eff,
            pp_efficiency=pp_eff,
            dp_efficiency=dp_eff,
            num_data_points=len(self.data_points),
        )
        
        self.calibration_model = model
        return model
    
    def get_similar_benchmarks(
        self,
        model_params_b: float,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        max_results: int = 5,
    ) -> List[BenchmarkDataPoint]:
        """Find similar benchmarks to a query configuration.
        
        Args:
            model_params_b: Model size in billions
            tp: Tensor parallel size
            pp: Pipeline parallel size
            dp: Data parallel size
            max_results: Maximum results to return
            
        Returns:
            List of similar benchmark data points
        """
        if not self.data_points:
            return []
        
        # Score each data point by similarity
        def similarity(point: BenchmarkDataPoint) -> float:
            params_diff = abs(math.log(point.model_params_b + 1) - math.log(model_params_b + 1))
            tp_match = 1.0 if point.tp_size == tp else 0.5
            pp_match = 1.0 if point.pp_size == pp else 0.5
            dp_match = 1.0 if point.dp_size == dp else 0.5
            
            return (1.0 / (1.0 + params_diff)) * tp_match * pp_match * dp_match
        
        scored = [(similarity(p), p) for p in self.data_points]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [p for _, p in scored[:max_results]]
    
    def format_calibration_report(self) -> str:
        """Generate a report of calibration data."""
        if not self.calibration_model:
            self.calibrate()
        
        model = self.calibration_model
        
        lines = [
            "=" * 70,
            "CALIBRATION REPORT",
            "=" * 70,
            "",
            f"Data Points: {model.num_data_points}",
            "",
            "Tensor Parallel Efficiency:",
        ]
        
        for tp, eff in sorted(model.tp_efficiency.items()):
            lines.append(f"  TP={tp}: {eff:.2%}")
        
        lines.extend([
            "",
            "Pipeline Parallel Efficiency:",
        ])
        for pp, eff in sorted(model.pp_efficiency.items()):
            lines.append(f"  PP={pp}: {eff:.2%}")
        
        lines.extend([
            "",
            "Data Parallel Efficiency:",
        ])
        for dp, eff in sorted(model.dp_efficiency.items()):
            lines.append(f"  DP={dp}: {eff:.2%}")
        
        if self.data_points:
            lines.extend([
                "",
                "Top Benchmark Results:",
            ])
            for point in self.data_points[:5]:
                lines.append(
                    f"  {point.model_name}: {point.throughput_tps:.0f} tps "
                    f"(TP={point.tp_size}, PP={point.pp_size}, DP={point.dp_size})"
                )
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)
