"""
Parallelism Strategy Advisor

A comprehensive tool for analyzing model architectures and recommending optimal
parallelism strategies (DP, TP, PP, CP, EP, SP) based on actual hardware topology.

Features:
- Real-time hardware topology detection (NVLink mesh, P2P, NUMA, interconnects)
- Multi-node cluster support with cross-node strategies
- Model architecture analysis (from HuggingFace or custom specs)
- Smart parallelism recommendations with rationale
- ZeRO-3, FSDP, and HSDP sharding recommendations
- Profiling-based calibration from benchmark data
- Cost/throughput Pareto frontier analysis
- Framework launch commands (torchrun, DeepSpeed, Accelerate, Megatron)

Usage:
    from tools.parallelism_planner import ParallelismAdvisor
    
    advisor = ParallelismAdvisor()
    recommendations = advisor.recommend("meta-llama/Llama-3.1-70B")
    
    # With sharding recommendations
    from tools.parallelism_planner import ShardingOptimizer
    sharding = ShardingOptimizer().recommend(model_arch, dp_size=8, gpu_memory_gb=80)
    
    # Generate launch commands
    from tools.parallelism_planner import LaunchCommandGenerator
    gen = LaunchCommandGenerator()
    commands = gen.generate_all(launch_config, "train.py")
"""

from .topology_detector import TopologyDetector, TopologyInfo
from .model_analyzer import ModelAnalyzer, ModelArchitecture
from .strategy_optimizer import StrategyOptimizer, ParallelismStrategy, StrategyRecommendation
from .advisor import ParallelismAdvisor

# Multi-node cluster support
from .cluster_config import (
    ClusterDetector,
    ClusterTopology,
    NodeSpec,
    NetworkType,
    create_cluster_preset_dgx_h100_8x,
    create_cluster_preset_dgx_gb200_nvl72,
    create_cluster_preset_8xb200_single_node,
)

# Sharding strategies (ZeRO, FSDP, HSDP)
from .sharding_strategies import (
    ShardingOptimizer,
    ShardingStrategy,
    ShardingConfig,
    ShardingAnalysis,
    ShardingRecommendation,
)

# Launch command generation
from .launch_commands import (
    LaunchCommandGenerator,
    LaunchConfig,
    TorchrunGenerator,
    DeepSpeedGenerator,
    AccelerateGenerator,
    MegatronGenerator,
)

# Calibration from benchmark data
from .calibration import (
    CalibrationEngine,
    CalibrationModel,
    BenchmarkDataPoint,
)

# Pareto frontier analysis
from .pareto_analysis import (
    ParetoAnalyzer,
    ParetoFrontier,
    ConfigurationPoint,
    GPU_PRICING,
)

# Validation and dry-run
from .validation import (
    ConfigValidator,
    MemoryValidator,
    DryRunner,
    FrameworkConfigValidator,
    ValidationResult,
    DryRunResult,
    ValidationStatus,
    validate_full_configuration,
)

# Advanced optimizations
from .advanced_optimizations import (
    PrecisionOptimizer,
    CheckpointingOptimizer,
    PipelineScheduleOptimizer,
    MemoryEfficientOptimizerRecommender,
    KernelFusionRecommender,
    CommunicationOptimizer,
    CompoundOptimizationGenerator,
    get_advanced_optimization_report,
    PrecisionMode,
    CheckpointingStrategy,
    PipelineSchedule,
)

# Performance profiles
from .performance_profiles import (
    ProfileGenerator,
    ProfileSelector,
    PerformanceProfile,
    WorkloadType,
    OptimizationPriority,
    get_performance_profile,
    list_available_profiles,
)

# Bottleneck analysis
from .bottleneck_analysis import (
    BottleneckDetector,
    ScalingAnalyzer,
    WhatIfAnalyzer,
    BottleneckAnalysis,
    ScalingAnalysis,
    WhatIfResult,
    BottleneckType,
    analyze_bottlenecks,
    analyze_scaling,
    analyze_whatif,
)

# Auto-tuning
from .auto_tuning import (
    BatchSizeFinder,
    GradientAccumulationOptimizer,
    AutoTuner,
    BatchSizeResult,
    GradAccumResult,
    AutoTuneResult,
    find_max_batch_size,
    optimize_gradient_accumulation,
    auto_tune_config,
)

# Inference optimization
from .inference_optimization import (
    QuantizationOptimizer,
    KVCacheOptimizer,
    SpeculativeDecodingOptimizer,
    InferenceEngineRecommender,
    QuantizationType,
    InferenceEngine,
    QuantizationRecommendation,
    KVCacheOptimization,
    SpeculativeDecodingRecommendation,
    InferenceEngineRecommendation,
    InferenceOptimizationReport,
    get_inference_optimization_report,
)

# Distributed Training & Advanced Features (NEW!)
from .distributed_training import (
    NCCLTuningAdvisor,
    RLHFMemoryCalculator,
    MoEOptimizer,
    LongContextOptimizer,
    VLLMConfigGenerator,
    CommunicationOverlapAnalyzer,
    NCCLBackend,
    RLHFAlgorithm,
    MoEStrategy,
    NCCLConfig,
    RLHFMemoryEstimate,
    MoEConfig,
    LongContextConfig,
    VLLMConfig,
)

# Troubleshooting
from .troubleshooting import (
    DistributedTrainingTroubleshooter,
    MemoryAnalyzer,
    TroubleshootingIssue,
    MemoryBreakdown,
    ErrorCategory,
    diagnose_error,
    get_nccl_tuning,
    get_memory_breakdown,
    get_all_troubleshooting_topics,
)

# Configuration export
from .config_export import (
    ConfigExporter,
    ExportedConfig,
    export_training_config,
)

# RL/RLHF optimization
from .rl_optimization import (
    RLOptimizer,
    RLAlgorithm,
    ReferenceModelStrategy,
    RLMemoryPlan,
    RLTrainingConfig,
    get_rl_optimization,
)

# vLLM optimization
from .vllm_optimization import (
    VLLMOptimizer,
    VLLMConfig,
    VLLMPerformanceMetrics,
    VLLMScheduler,
    VLLMQuantization,
    SLARequirements,
    get_vllm_optimization,
)

# Large-scale cluster optimization
from .large_scale_optimization import (
    LargeScaleOptimizer,
    ClusterTopology,
    LargeScaleConfig,
    ScaleEfficiencyAnalysis,
    ClusterType,
    FaultToleranceStrategy,
    CheckpointStrategy,
    get_large_scale_optimization,
)

__all__ = [
    # Core
    "ParallelismAdvisor",
    "TopologyDetector",
    "TopologyInfo", 
    "ModelAnalyzer",
    "ModelArchitecture",
    "StrategyOptimizer",
    "ParallelismStrategy",
    "StrategyRecommendation",
    
    # Cluster
    "ClusterDetector",
    "ClusterTopology",
    "NodeSpec",
    "NetworkType",
    "create_cluster_preset_dgx_h100_8x",
    "create_cluster_preset_dgx_gb200_nvl72",
    "create_cluster_preset_8xb200_single_node",
    
    # Sharding
    "ShardingOptimizer",
    "ShardingStrategy",
    "ShardingConfig",
    "ShardingAnalysis",
    "ShardingRecommendation",
    
    # Launch commands
    "LaunchCommandGenerator",
    "LaunchConfig",
    "TorchrunGenerator",
    "DeepSpeedGenerator",
    "AccelerateGenerator",
    "MegatronGenerator",
    
    # Calibration
    "CalibrationEngine",
    "CalibrationModel",
    "BenchmarkDataPoint",
    
    # Pareto Analysis
    "ParetoAnalyzer",
    "ParetoFrontier",
    "ConfigurationPoint",
    "GPU_PRICING",
    
    # Validation
    "ConfigValidator",
    "MemoryValidator",
    "DryRunner",
    "FrameworkConfigValidator",
    "ValidationResult",
    "DryRunResult",
    "ValidationStatus",
    "validate_full_configuration",
    
    # Advanced Optimizations
    "PrecisionOptimizer",
    "CheckpointingOptimizer",
    "PipelineScheduleOptimizer",
    "MemoryEfficientOptimizerRecommender",
    "KernelFusionRecommender",
    "CommunicationOptimizer",
    "CompoundOptimizationGenerator",
    "get_advanced_optimization_report",
    "PrecisionMode",
    "CheckpointingStrategy",
    "PipelineSchedule",
    
    # Performance Profiles
    "ProfileGenerator",
    "ProfileSelector",
    "PerformanceProfile",
    "WorkloadType",
    "OptimizationPriority",
    "get_performance_profile",
    "list_available_profiles",
    
    # Bottleneck Analysis
    "BottleneckDetector",
    "ScalingAnalyzer",
    "WhatIfAnalyzer",
    "BottleneckAnalysis",
    "ScalingAnalysis",
    "WhatIfResult",
    "BottleneckType",
    "analyze_bottlenecks",
    "analyze_scaling",
    "analyze_whatif",
    
    # Auto-Tuning
    "BatchSizeFinder",
    "GradientAccumulationOptimizer",
    "AutoTuner",
    "BatchSizeResult",
    "GradAccumResult",
    "AutoTuneResult",
    "find_max_batch_size",
    "optimize_gradient_accumulation",
    "auto_tune_config",
    
    # Inference Optimization
    "QuantizationOptimizer",
    "KVCacheOptimizer",
    "SpeculativeDecodingOptimizer",
    "InferenceEngineRecommender",
    "QuantizationType",
    "InferenceEngine",
    "QuantizationRecommendation",
    "KVCacheOptimization",
    "SpeculativeDecodingRecommendation",
    "InferenceEngineRecommendation",
    "InferenceOptimizationReport",
    "get_inference_optimization_report",
    
    # Troubleshooting
    "DistributedTrainingTroubleshooter",
    "MemoryAnalyzer",
    "TroubleshootingIssue",
    "MemoryBreakdown",
    "ErrorCategory",
    "diagnose_error",
    "get_nccl_tuning",
    "get_memory_breakdown",
    "get_all_troubleshooting_topics",
    
    # Configuration Export
    "ConfigExporter",
    "ExportedConfig",
    "export_training_config",
    
    # Distributed Training & Advanced Features (NEW!)
    "NCCLTuningAdvisor",
    "RLHFMemoryCalculator",
    "MoEOptimizer",
    "LongContextOptimizer",
    "VLLMConfigGenerator",
    "CommunicationOverlapAnalyzer",
    "NCCLBackend",
    "RLHFAlgorithm",
    "MoEStrategy",
    "NCCLConfig",
    "RLHFMemoryEstimate",
    "MoEConfig",
    "LongContextConfig",
    "VLLMConfig",
    
    # RL/RLHF Optimization (Deep Integration)
    "RLOptimizer",
    "RLAlgorithm",
    "ReferenceModelStrategy",
    "RLMemoryPlan",
    "RLTrainingConfig",
    "get_rl_optimization",
    
    # vLLM Optimization (Deep Integration)
    "VLLMOptimizer",
    "VLLMPerformanceMetrics",
    "VLLMScheduler",
    "VLLMQuantization",
    "SLARequirements",
    "get_vllm_optimization",
    
    # Large-Scale Cluster Optimization
    "LargeScaleOptimizer",
    "ClusterTopology",
    "LargeScaleConfig",
    "ScaleEfficiencyAnalysis",
    "ClusterType",
    "FaultToleranceStrategy",
    "CheckpointStrategy",
    "get_large_scale_optimization",
]
