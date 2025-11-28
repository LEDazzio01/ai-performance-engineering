"""
Large-Scale Cluster Optimization Module for Parallelism Planner

Enterprise-scale distributed training optimization:
- Multi-cluster orchestration
- Fault tolerance and elastic training
- Network topology optimization
- Checkpoint strategy optimization
- Data pipeline optimization
- Communication efficiency analysis
- Cost optimization at scale
- SLURM/Kubernetes job configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math


class ClusterType(Enum):
    """Types of compute clusters."""
    DGX_CLOUD = "dgx_cloud"
    AWS_P5 = "aws_p5"
    AWS_P4D = "aws_p4d"
    GCP_A3 = "gcp_a3"
    AZURE_ND = "azure_nd"
    CUSTOM = "custom"


class FaultToleranceStrategy(Enum):
    """Fault tolerance strategies."""
    CHECKPOINT_RESTART = "checkpoint_restart"
    ELASTIC_TRAINING = "elastic_training"
    REDUNDANT_WORKERS = "redundant_workers"
    PIPELINE_BUBBLE_RECOVERY = "pipeline_bubble"


class CheckpointStrategy(Enum):
    """Checkpoint strategies for large-scale training."""
    SYNCHRONOUS = "synchronous"           # All ranks save together
    ASYNCHRONOUS = "asynchronous"         # Save in background
    SHARDED = "sharded"                   # Each rank saves its shard
    INCREMENTAL = "incremental"           # Only save changes
    DISTRIBUTED_FILESYSTEM = "distributed_fs"  # Save to distributed FS


@dataclass
class ClusterTopology:
    """Large-scale cluster topology."""
    num_nodes: int
    gpus_per_node: int
    gpu_memory_gb: float
    inter_node_bandwidth_gbps: float
    intra_node_bandwidth_gbps: float
    
    # Network topology
    network_type: str  # "infiniband", "roce", "ethernet"
    fat_tree_levels: int
    rail_optimized: bool
    
    # Storage
    shared_filesystem: bool
    storage_bandwidth_gbps: float
    
    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node
    
    @property
    def total_memory_gb(self) -> float:
        return self.total_gpus * self.gpu_memory_gb


@dataclass
class LargeScaleConfig:
    """Configuration for large-scale training."""
    # Parallelism
    tp: int
    pp: int
    dp: int
    cp: int
    ep: int
    
    # 3D parallelism mapping
    tp_within_node: bool
    pp_across_nodes: bool
    dp_hierarchical: bool
    
    # Fault tolerance
    fault_tolerance: FaultToleranceStrategy
    checkpoint_strategy: CheckpointStrategy
    checkpoint_interval_minutes: int
    
    # Communication
    gradient_compression: bool
    overlap_communication: bool
    hierarchical_allreduce: bool
    
    # Data pipeline
    dataloader_workers: int
    prefetch_factor: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parallelism": {
                "tp": self.tp,
                "pp": self.pp,
                "dp": self.dp,
                "cp": self.cp,
                "ep": self.ep,
            },
            "mapping": {
                "tp_within_node": self.tp_within_node,
                "pp_across_nodes": self.pp_across_nodes,
                "dp_hierarchical": self.dp_hierarchical,
            },
            "fault_tolerance": {
                "strategy": self.fault_tolerance.value,
                "checkpoint_strategy": self.checkpoint_strategy.value,
                "checkpoint_interval_minutes": self.checkpoint_interval_minutes,
            },
            "communication": {
                "gradient_compression": self.gradient_compression,
                "overlap_communication": self.overlap_communication,
                "hierarchical_allreduce": self.hierarchical_allreduce,
            },
            "data_pipeline": {
                "dataloader_workers": self.dataloader_workers,
                "prefetch_factor": self.prefetch_factor,
            },
        }


@dataclass
class ScaleEfficiencyAnalysis:
    """Analysis of scaling efficiency."""
    # Efficiency metrics
    mfu: float  # Model FLOPS Utilization
    communication_efficiency: float
    memory_efficiency: float
    
    # Bottleneck analysis
    compute_bound_pct: float
    memory_bound_pct: float
    communication_bound_pct: float
    
    # Scaling projections
    strong_scaling_efficiency: float
    weak_scaling_efficiency: float
    
    # Cost analysis
    gpu_hours_per_trillion_tokens: float
    cost_per_trillion_tokens: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "efficiency": {
                "mfu": self.mfu,
                "communication_efficiency": self.communication_efficiency,
                "memory_efficiency": self.memory_efficiency,
            },
            "bottleneck": {
                "compute_pct": self.compute_bound_pct,
                "memory_pct": self.memory_bound_pct,
                "communication_pct": self.communication_bound_pct,
            },
            "scaling": {
                "strong_scaling_efficiency": self.strong_scaling_efficiency,
                "weak_scaling_efficiency": self.weak_scaling_efficiency,
            },
            "cost": {
                "gpu_hours_per_trillion_tokens": self.gpu_hours_per_trillion_tokens,
                "cost_per_trillion_tokens": self.cost_per_trillion_tokens,
            },
        }


class LargeScaleOptimizer:
    """Optimizes configuration for large-scale distributed training."""
    
    # Cluster presets
    CLUSTER_PRESETS = {
        ClusterType.DGX_CLOUD: {
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
            "inter_node_bandwidth_gbps": 400,
            "intra_node_bandwidth_gbps": 900,
            "network_type": "infiniband",
        },
        ClusterType.AWS_P5: {
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
            "inter_node_bandwidth_gbps": 400,
            "intra_node_bandwidth_gbps": 900,
            "network_type": "efa",
        },
        ClusterType.AWS_P4D: {
            "gpus_per_node": 8,
            "gpu_memory_gb": 40,
            "inter_node_bandwidth_gbps": 400,
            "intra_node_bandwidth_gbps": 600,
            "network_type": "efa",
        },
    }
    
    def __init__(self, cluster: ClusterTopology):
        self.cluster = cluster
    
    def optimize(
        self,
        model_params_b: float,
        seq_length: int,
        global_batch_size: int,
        num_experts: int = 1,
    ) -> Tuple[LargeScaleConfig, ScaleEfficiencyAnalysis]:
        """
        Optimize configuration for large-scale training.
        """
        
        total_gpus = self.cluster.total_gpus
        gpus_per_node = self.cluster.gpus_per_node
        
        # Determine optimal parallelism
        # TP: within NVLink domain (usually 8)
        tp = min(8, gpus_per_node)
        
        # PP: for very large models
        if model_params_b > 100:
            # Need PP across nodes
            min_pp_for_memory = max(1, int(model_params_b * 2 / self.cluster.gpu_memory_gb / tp))
            pp = min(16, min_pp_for_memory)
        elif model_params_b > 70:
            pp = 4
        else:
            pp = 1
        
        # EP: for MoE
        ep = 1
        if num_experts > 1:
            ep = min(num_experts, total_gpus // (tp * pp))
        
        # DP: remaining GPUs
        dp = total_gpus // (tp * pp * ep)
        
        # CP: for very long sequences
        cp = 1
        if seq_length > 32768 and dp > 1:
            cp = min(4, dp)
            dp = dp // cp
        
        # Determine 3D parallelism mapping
        tp_within_node = tp <= gpus_per_node
        pp_across_nodes = pp > 1 and self.cluster.num_nodes > 1
        dp_hierarchical = dp > gpus_per_node and self.cluster.num_nodes > 1
        
        # Fault tolerance
        if self.cluster.num_nodes > 32:
            fault_tolerance = FaultToleranceStrategy.ELASTIC_TRAINING
            checkpoint_interval = 15  # More frequent for large scale
        elif self.cluster.num_nodes > 8:
            fault_tolerance = FaultToleranceStrategy.CHECKPOINT_RESTART
            checkpoint_interval = 30
        else:
            fault_tolerance = FaultToleranceStrategy.CHECKPOINT_RESTART
            checkpoint_interval = 60
        
        # Checkpoint strategy
        if model_params_b > 100:
            checkpoint_strategy = CheckpointStrategy.SHARDED
        elif self.cluster.shared_filesystem:
            checkpoint_strategy = CheckpointStrategy.ASYNCHRONOUS
        else:
            checkpoint_strategy = CheckpointStrategy.DISTRIBUTED_FILESYSTEM
        
        # Communication optimizations
        gradient_compression = self.cluster.num_nodes > 16 and self.cluster.network_type == "ethernet"
        overlap_communication = True
        hierarchical_allreduce = dp > gpus_per_node
        
        # Data pipeline
        dataloader_workers = min(8, gpus_per_node)
        prefetch_factor = 2
        
        config = LargeScaleConfig(
            tp=tp,
            pp=pp,
            dp=dp,
            cp=cp,
            ep=ep,
            tp_within_node=tp_within_node,
            pp_across_nodes=pp_across_nodes,
            dp_hierarchical=dp_hierarchical,
            fault_tolerance=fault_tolerance,
            checkpoint_strategy=checkpoint_strategy,
            checkpoint_interval_minutes=checkpoint_interval,
            gradient_compression=gradient_compression,
            overlap_communication=overlap_communication,
            hierarchical_allreduce=hierarchical_allreduce,
            dataloader_workers=dataloader_workers,
            prefetch_factor=prefetch_factor,
        )
        
        # Analyze efficiency
        efficiency = self._analyze_efficiency(config, model_params_b, seq_length, global_batch_size)
        
        return config, efficiency
    
    def _analyze_efficiency(
        self,
        config: LargeScaleConfig,
        model_params_b: float,
        seq_length: int,
        global_batch_size: int,
    ) -> ScaleEfficiencyAnalysis:
        """Analyze scaling efficiency."""
        
        total_gpus = self.cluster.total_gpus
        
        # Base MFU estimate
        base_mfu = 0.45  # Typical for well-optimized training
        
        # Adjust for communication overhead
        tp_overhead = 0.98 ** (config.tp - 1)
        pp_bubble = (config.pp - 1) / (config.pp * 4) if config.pp > 1 else 0
        dp_overhead = 0.99 if config.dp > 1 else 1.0
        
        mfu = base_mfu * tp_overhead * (1 - pp_bubble) * dp_overhead
        
        # Communication efficiency
        if self.cluster.network_type == "infiniband":
            comm_eff = 0.95
        elif self.cluster.network_type == "efa":
            comm_eff = 0.90
        else:
            comm_eff = 0.80
        
        if config.gradient_compression:
            comm_eff *= 0.95  # Slight overhead for compression
        
        # Memory efficiency
        mem_eff = min(1.0, (model_params_b * 2 * 1.5) / self.cluster.gpu_memory_gb)
        
        # Bottleneck analysis
        compute_pct = mfu * 100
        comm_pct = (1 - comm_eff) * 100
        memory_pct = 100 - compute_pct - comm_pct
        
        # Scaling efficiency
        strong_scaling = mfu * comm_eff
        weak_scaling = mfu * (1 - pp_bubble)
        
        # Cost analysis
        tflops_per_gpu = 1979  # H100
        tokens_per_second = (tflops_per_gpu * 1e12 * total_gpus * mfu) / (6 * model_params_b * 1e9)
        
        gpu_hours_per_trillion = 1e12 / tokens_per_second / 3600
        cost_per_gpu_hour = 4.0  # Estimate
        cost_per_trillion = gpu_hours_per_trillion * cost_per_gpu_hour * total_gpus
        
        return ScaleEfficiencyAnalysis(
            mfu=mfu,
            communication_efficiency=comm_eff,
            memory_efficiency=mem_eff,
            compute_bound_pct=compute_pct,
            memory_bound_pct=max(0, memory_pct),
            communication_bound_pct=comm_pct,
            strong_scaling_efficiency=strong_scaling,
            weak_scaling_efficiency=weak_scaling,
            gpu_hours_per_trillion_tokens=gpu_hours_per_trillion,
            cost_per_trillion_tokens=cost_per_trillion,
        )
    
    def generate_slurm_config(
        self,
        config: LargeScaleConfig,
        model_name: str,
        job_name: str = "train",
        time_hours: int = 48,
        partition: str = "gpu",
    ) -> str:
        """Generate SLURM job script for large-scale training."""
        
        total_gpus = self.cluster.total_gpus
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={self.cluster.num_nodes}
#SBATCH --ntasks-per-node={self.cluster.gpus_per_node}
#SBATCH --gpus-per-node={self.cluster.gpus_per_node}
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time={time_hours}:00:00
#SBATCH --partition={partition}
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ===== ENVIRONMENT =====
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE={total_gpus}
export GPUS_PER_NODE={self.cluster.gpus_per_node}

# NCCL optimizations for large scale
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
"""
        
        if self.cluster.network_type == "infiniband":
            script += """
# InfiniBand optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=1
"""
        
        if config.hierarchical_allreduce:
            script += """
# Hierarchical all-reduce
export NCCL_ALGO=Tree
export NCCL_TREE_THRESHOLD=0
"""
        
        script += f"""
# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== CHECKPOINT RECOVERY =====
CHECKPOINT_DIR="checkpoints/{job_name}"
mkdir -p $CHECKPOINT_DIR

# Find latest checkpoint
LATEST_CKPT=$(ls -t $CHECKPOINT_DIR/*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume $LATEST_CKPT"
else
    RESUME_ARG=""
fi

# ===== LAUNCH TRAINING =====
srun --kill-on-bad-exit=1 \\
    torchrun \\
    --nnodes=$SLURM_NNODES \\
    --nproc_per_node=$GPUS_PER_NODE \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    train.py \\
    --model {model_name} \\
    --tensor-parallel-size {config.tp} \\
    --pipeline-parallel-size {config.pp} \\
    --data-parallel-size {config.dp} \\
    --checkpoint-interval {config.checkpoint_interval_minutes * 60} \\
    --checkpoint-dir $CHECKPOINT_DIR \\
    $RESUME_ARG
"""
        
        return script
    
    def generate_kubernetes_config(
        self,
        config: LargeScaleConfig,
        model_name: str,
        image: str = "nvcr.io/nvidia/pytorch:24.01-py3",
    ) -> Dict[str, Any]:
        """Generate Kubernetes job configuration."""
        
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": f"train-{model_name.replace('/', '-')}",
            },
            "spec": {
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": 1,
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": image,
                                    "resources": {
                                        "limits": {
                                            "nvidia.com/gpu": self.cluster.gpus_per_node,
                                        }
                                    },
                                    "env": [
                                        {"name": "NCCL_DEBUG", "value": "WARN"},
                                        {"name": "NCCL_TIMEOUT", "value": "1800"},
                                    ],
                                }],
                            }
                        }
                    },
                    "Worker": {
                        "replicas": self.cluster.num_nodes - 1,
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": image,
                                    "resources": {
                                        "limits": {
                                            "nvidia.com/gpu": self.cluster.gpus_per_node,
                                        }
                                    },
                                }],
                            }
                        }
                    },
                },
            },
        }


def get_large_scale_optimization(
    model_config: Dict[str, Any],
    cluster_config: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get large-scale training optimization recommendations.
    """
    
    training = training_config or {}
    
    cluster = ClusterTopology(
        num_nodes=cluster_config.get("num_nodes", 8),
        gpus_per_node=cluster_config.get("gpus_per_node", 8),
        gpu_memory_gb=cluster_config.get("gpu_memory_gb", 80),
        inter_node_bandwidth_gbps=cluster_config.get("inter_node_bandwidth_gbps", 400),
        intra_node_bandwidth_gbps=cluster_config.get("intra_node_bandwidth_gbps", 900),
        network_type=cluster_config.get("network_type", "infiniband"),
        fat_tree_levels=cluster_config.get("fat_tree_levels", 2),
        rail_optimized=cluster_config.get("rail_optimized", True),
        shared_filesystem=cluster_config.get("shared_filesystem", True),
        storage_bandwidth_gbps=cluster_config.get("storage_bandwidth_gbps", 100),
    )
    
    optimizer = LargeScaleOptimizer(cluster)
    
    config, efficiency = optimizer.optimize(
        model_params_b=model_config.get("parameters_billions", 70),
        seq_length=model_config.get("max_sequence_length", 4096),
        global_batch_size=training.get("global_batch_size", 1024),
        num_experts=model_config.get("num_experts", 1),
    )
    
    slurm_script = optimizer.generate_slurm_config(
        config,
        model_name=model_config.get("name", "model"),
    )
    
    k8s_config = optimizer.generate_kubernetes_config(
        config,
        model_name=model_config.get("name", "model"),
    )
    
    return {
        "cluster": {
            "num_nodes": cluster.num_nodes,
            "total_gpus": cluster.total_gpus,
            "total_memory_gb": cluster.total_memory_gb,
        },
        "config": config.to_dict(),
        "efficiency": efficiency.to_dict(),
        "slurm_script": slurm_script,
        "kubernetes_config": k8s_config,
        "recommendations": _get_large_scale_recommendations(config, efficiency, cluster),
    }


def _get_large_scale_recommendations(
    config: LargeScaleConfig,
    efficiency: ScaleEfficiencyAnalysis,
    cluster: ClusterTopology,
) -> List[str]:
    """Generate recommendations for large-scale training."""
    recommendations = []
    
    # Parallelism recommendations
    recommendations.append(f"3D Parallelism: TP={config.tp} × PP={config.pp} × DP={config.dp}")
    
    if config.tp_within_node:
        recommendations.append("TP within node uses NVLink - optimal for communication")
    
    if config.pp_across_nodes:
        recommendations.append("PP across nodes - ensure low-latency interconnect")
    
    if config.dp_hierarchical:
        recommendations.append("Hierarchical DP all-reduce for cross-node gradient sync")
    
    # Efficiency recommendations
    if efficiency.mfu < 0.40:
        recommendations.append(f"⚠ MFU ({efficiency.mfu:.1%}) is low - check for bottlenecks")
    else:
        recommendations.append(f"✓ Good MFU ({efficiency.mfu:.1%})")
    
    if efficiency.communication_bound_pct > 30:
        recommendations.append("⚠ Communication bound - consider gradient compression")
    
    # Fault tolerance
    if cluster.num_nodes > 16:
        recommendations.append(f"Large cluster ({cluster.num_nodes} nodes) - checkpoint every {config.checkpoint_interval_minutes}min")
        recommendations.append("Consider elastic training for better fault tolerance")
    
    # Cost
    recommendations.append(f"Estimated cost: ${efficiency.cost_per_trillion_tokens:,.0f} per trillion tokens")
    
    return recommendations



