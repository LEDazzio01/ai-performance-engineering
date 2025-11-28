#!/usr/bin/env python3
"""
Parallelism Advisor - Main Entry Point

Provides a unified interface for analyzing model/hardware combinations
and recommending optimal parallelism strategies.

Usage:
    from tools.parallelism_planner import ParallelismAdvisor
    
    advisor = ParallelismAdvisor()
    
    # Quick recommendation
    result = advisor.recommend("meta-llama/Llama-3.1-70B")
    
    # Detailed analysis
    result = advisor.analyze(
        model="llama-3.1-405b",
        batch_size=32,
        seq_length=8192,
        goal="throughput",
        is_training=True,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .topology_detector import TopologyDetector, TopologyInfo
from .model_analyzer import ModelAnalyzer, ModelArchitecture
from .strategy_optimizer import (
    StrategyOptimizer,
    ParallelismStrategy,
    StrategyRecommendation,
    OptimizationGoal,
)


@dataclass
class AdvisorResult:
    """Complete analysis and recommendations."""
    
    # Input summary
    model_name: str
    model_architecture: ModelArchitecture
    topology: TopologyInfo
    
    # Analysis parameters
    batch_size: int
    seq_length: int
    goal: OptimizationGoal
    is_training: bool
    
    # Recommendations
    recommendations: List[StrategyRecommendation]
    
    # Best recommendation (first one)
    @property
    def best(self) -> Optional[StrategyRecommendation]:
        return self.recommendations[0] if self.recommendations else None
    
    @property
    def best_strategy(self) -> Optional[ParallelismStrategy]:
        return self.best.strategy if self.best else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "name": self.model_name,
                "architecture": self.model_architecture.to_dict(),
            },
            "topology": self.topology.to_dict(),
            "parameters": {
                "batch_size": self.batch_size,
                "seq_length": self.seq_length,
                "goal": self.goal.value,
                "is_training": self.is_training,
            },
            "recommendations": [r.to_dict() for r in self.recommendations],
            "best_strategy": self.best_strategy.to_dict() if self.best_strategy else None,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary(self) -> str:
        """Generate a one-line summary."""
        if not self.best:
            return f"No valid strategies for {self.model_name}"
        
        s = self.best.strategy
        return (
            f"{self.model_name}: TP={s.tp}×PP={s.pp}×DP={s.dp}"
            + (f"×CP={s.cp}" if s.cp > 1 else "")
            + (f"×EP={s.ep}" if s.ep > 1 else "")
            + f" (Score: {self.best.score:.0f}/100)"
        )


class ParallelismAdvisor:
    """Main interface for parallelism strategy recommendations."""
    
    def __init__(self, auto_detect_topology: bool = True):
        """Initialize the advisor.
        
        Args:
            auto_detect_topology: Automatically detect hardware topology
        """
        self.topology_detector = TopologyDetector()
        self.model_analyzer = ModelAnalyzer()
        self.strategy_optimizer = StrategyOptimizer()
        
        self._topology: Optional[TopologyInfo] = None
        self._auto_detect = auto_detect_topology
    
    @property
    def topology(self) -> Optional[TopologyInfo]:
        """Get cached topology or detect if needed."""
        if self._topology is None and self._auto_detect:
            try:
                self._topology = self.topology_detector.detect()
            except RuntimeError:
                pass  # CUDA not available
        return self._topology
    
    def set_topology(self, topology: TopologyInfo) -> None:
        """Set a custom topology (e.g., for simulation)."""
        self._topology = topology
    
    def recommend(
        self,
        model: Union[str, ModelArchitecture],
        batch_size: int = 1,
        seq_length: int = 2048,
        goal: Union[str, OptimizationGoal] = "throughput",
        is_training: bool = False,
        custom_topology: Optional[TopologyInfo] = None,
        max_strategies: int = 5,
    ) -> AdvisorResult:
        """Get parallelism recommendations for a model.
        
        Args:
            model: Model name (HuggingFace ID or preset) or ModelArchitecture
            batch_size: Target batch size
            seq_length: Target sequence length
            goal: Optimization goal ("throughput", "latency", "memory", "efficiency")
            is_training: Whether this is for training (vs inference)
            custom_topology: Override detected topology
            max_strategies: Maximum strategies to return
            
        Returns:
            AdvisorResult with recommendations
        """
        # Resolve model
        if isinstance(model, str):
            model_arch = self.model_analyzer.analyze(model)
            model_name = model
        else:
            model_arch = model
            model_name = model.name
        
        # Resolve topology
        topology = custom_topology or self.topology
        if topology is None:
            raise RuntimeError(
                "No hardware topology available. Either CUDA is not available "
                "or provide a custom_topology."
            )
        
        # Resolve goal
        if isinstance(goal, str):
            goal = OptimizationGoal(goal.lower())
        
        # Get recommendations
        recommendations = self.strategy_optimizer.recommend(
            topology=topology,
            model=model_arch,
            goal=goal,
            batch_size=batch_size,
            seq_length=seq_length,
            is_training=is_training,
            max_strategies=max_strategies,
        )
        
        return AdvisorResult(
            model_name=model_name,
            model_architecture=model_arch,
            topology=topology,
            batch_size=batch_size,
            seq_length=seq_length,
            goal=goal,
            is_training=is_training,
            recommendations=recommendations,
        )
    
    def analyze_model(self, model: str) -> ModelArchitecture:
        """Analyze a model's architecture."""
        return self.model_analyzer.analyze(model)
    
    def detect_topology(self, force_refresh: bool = False) -> TopologyInfo:
        """Detect hardware topology."""
        self._topology = self.topology_detector.detect(force_refresh=force_refresh)
        return self._topology
    
    def compare_strategies(
        self,
        model: Union[str, ModelArchitecture],
        strategies: List[ParallelismStrategy],
        batch_size: int = 1,
        seq_length: int = 2048,
        is_training: bool = False,
    ) -> List[StrategyRecommendation]:
        """Compare specific strategies for a model.
        
        Useful for what-if analysis.
        """
        if isinstance(model, str):
            model_arch = self.model_analyzer.analyze(model)
        else:
            model_arch = model
        
        topology = self.topology
        if topology is None:
            raise RuntimeError("No hardware topology available.")
        
        recommendations = []
        for strategy in strategies:
            analysis = self.strategy_optimizer._analyze_strategy(
                strategy, topology, model_arch, batch_size, seq_length, is_training
            )
            score, rationale, warnings, optimizations = self.strategy_optimizer._score_strategy(
                strategy, analysis, topology, model_arch,
                OptimizationGoal.THROUGHPUT, is_training
            )
            recommendations.append(StrategyRecommendation(
                strategy=strategy,
                analysis=analysis,
                score=score,
                rationale=rationale,
                warnings=warnings,
                optimizations=optimizations,
            ))
        
        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations
    
    def generate_report(
        self,
        model: Union[str, ModelArchitecture],
        batch_size: int = 1,
        seq_length: int = 2048,
        goal: str = "throughput",
        is_training: bool = False,
    ) -> str:
        """Generate a comprehensive human-readable report."""
        result = self.recommend(
            model=model,
            batch_size=batch_size,
            seq_length=seq_length,
            goal=goal,
            is_training=is_training,
        )
        
        lines = [
            self.topology_detector.format_topology_report(result.topology),
            "",
            self.model_analyzer.format_architecture_report(result.model_architecture),
            "",
            self.strategy_optimizer.format_recommendations(result.recommendations),
        ]
        
        return "\n".join(lines)
    
    def list_model_presets(self) -> List[str]:
        """List available model presets."""
        return self.model_analyzer.list_presets()
    
    # Convenience methods for common scenarios
    
    def recommend_for_inference(
        self,
        model: str,
        batch_size: int = 1,
        seq_length: int = 2048,
    ) -> AdvisorResult:
        """Get recommendations optimized for inference."""
        return self.recommend(
            model=model,
            batch_size=batch_size,
            seq_length=seq_length,
            goal="latency",
            is_training=False,
        )
    
    def recommend_for_training(
        self,
        model: str,
        batch_size: int = 8,
        seq_length: int = 4096,
    ) -> AdvisorResult:
        """Get recommendations optimized for training."""
        return self.recommend(
            model=model,
            batch_size=batch_size,
            seq_length=seq_length,
            goal="throughput",
            is_training=True,
        )
    
    def recommend_for_long_context(
        self,
        model: str,
        seq_length: int = 131072,
        batch_size: int = 1,
    ) -> AdvisorResult:
        """Get recommendations for long context workloads."""
        return self.recommend(
            model=model,
            batch_size=batch_size,
            seq_length=seq_length,
            goal="memory",  # Memory is usually the constraint
            is_training=False,
        )


def create_mock_topology_8xb200() -> TopologyInfo:
    """Create a mock 8x B200 topology for testing."""
    from .topology_detector import GPUInfo
    
    gpus = [
        GPUInfo(i, "NVIDIA B200", "10.0", 192, 148, "blackwell", True)
        for i in range(8)
    ]
    
    return TopologyInfo(
        num_gpus=8,
        gpus=gpus,
        total_memory_gb=192 * 8,
        interconnects=[],
        p2p_matrix=[[True] * 8 for _ in range(8)],
        bandwidth_matrix=[[900.0] * 8 for _ in range(8)],
        has_nvlink=True,
        has_nvswitch=True,
        nvlink_version="5.0",
        max_nvlink_bandwidth_gbps=900,
        numa_nodes=1,
        gpu_numa_mapping={i: 0 for i in range(8)},
        numa_distance_matrix=[[10]],
        cpu_type="aarch64",
        is_grace_cpu=True,
        has_nvlink_c2c=True,
        num_nodes=1,
        gpus_per_node=8,
    )


def create_mock_topology_8xh100() -> TopologyInfo:
    """Create a mock 8x H100 topology for testing."""
    from .topology_detector import GPUInfo
    
    gpus = [
        GPUInfo(i, "NVIDIA H100", "9.0", 80, 132, "hopper", True)
        for i in range(8)
    ]
    
    return TopologyInfo(
        num_gpus=8,
        gpus=gpus,
        total_memory_gb=80 * 8,
        interconnects=[],
        p2p_matrix=[[True] * 8 for _ in range(8)],
        bandwidth_matrix=[[600.0] * 8 for _ in range(8)],
        has_nvlink=True,
        has_nvswitch=True,
        nvlink_version="4.0",
        max_nvlink_bandwidth_gbps=600,
        numa_nodes=2,
        gpu_numa_mapping={i: i // 4 for i in range(8)},
        numa_distance_matrix=[[10, 20], [20, 10]],
        cpu_type="x86_64",
        is_grace_cpu=False,
        has_nvlink_c2c=False,
        num_nodes=1,
        gpus_per_node=8,
    )


# CLI Interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Parallelism Strategy Advisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick recommendation
    python -m tools.parallelism_planner.advisor llama-3.1-70b
    
    # Training configuration
    python -m tools.parallelism_planner.advisor llama-3.1-405b --training --batch-size 32
    
    # Long context inference
    python -m tools.parallelism_planner.advisor llama-3.1-70b --seq-length 131072
    
    # Use mock topology (no GPU required)
    python -m tools.parallelism_planner.advisor llama-3.1-70b --mock-topology 8xB200
        """,
    )
    
    parser.add_argument("model", help="Model name or HuggingFace ID")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--seq-length", "-s", type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--goal", "-g", choices=["throughput", "latency", "memory", "efficiency"],
                        default="throughput", help="Optimization goal")
    parser.add_argument("--training", "-t", action="store_true",
                        help="Configure for training (default: inference)")
    parser.add_argument("--mock-topology", choices=["8xB200", "8xH100"],
                        help="Use mock topology instead of detection")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available model presets")
    
    args = parser.parse_args()
    
    advisor = ParallelismAdvisor(auto_detect_topology=False)
    
    if args.list_presets:
        print("Available model presets:")
        for preset in advisor.list_model_presets():
            print(f"  {preset}")
        sys.exit(0)
    
    # Set topology
    if args.mock_topology:
        if args.mock_topology == "8xB200":
            advisor.set_topology(create_mock_topology_8xb200())
        else:
            advisor.set_topology(create_mock_topology_8xh100())
    else:
        try:
            advisor.detect_topology()
        except RuntimeError as e:
            print(f"Warning: Could not detect topology: {e}")
            print("Using mock 8xB200 topology...")
            advisor.set_topology(create_mock_topology_8xb200())
    
    try:
        if args.json:
            result = advisor.recommend(
                model=args.model,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                goal=args.goal,
                is_training=args.training,
            )
            print(result.to_json())
        else:
            report = advisor.generate_report(
                model=args.model,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                goal=args.goal,
                is_training=args.training,
            )
            print(report)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)



