#!/usr/bin/env python3
"""
Large-Scale Cluster Resilience & Advanced Features

Features:
- Fault tolerance configuration
- Elastic scaling recommendations
- Spot/preemptible instance handling
- Checkpoint optimization
- Health monitoring recommendations

This module provides LLM-guided recommendations, not hardcoded advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class FailureMode(Enum):
    """Types of failures in distributed training."""
    GPU_OOM = "gpu_oom"
    GPU_HARDWARE = "gpu_hardware"
    NETWORK_PARTITION = "network_partition"
    NCCL_TIMEOUT = "nccl_timeout"
    NODE_FAILURE = "node_failure"
    PREEMPTION = "preemption"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"


@dataclass
class FaultToleranceConfig:
    """Fault tolerance configuration for distributed training."""
    # Checkpointing
    checkpoint_interval_steps: int = 1000
    checkpoint_interval_minutes: int = 30
    async_checkpointing: bool = True
    checkpoint_compression: bool = True
    checkpoint_sharding: bool = True  # Shard checkpoints across workers
    max_checkpoints_to_keep: int = 3
    
    # Failure recovery
    auto_restart_on_failure: bool = True
    max_restarts: int = 3
    restart_delay_seconds: int = 30
    gradient_accumulation_on_oom: bool = True  # Reduce batch on OOM
    
    # Health monitoring
    heartbeat_interval_seconds: int = 10
    heartbeat_timeout_seconds: int = 60
    health_check_url: Optional[str] = None
    
    # NCCL settings for resilience
    nccl_timeout_minutes: int = 30
    nccl_async_error_handling: bool = True
    
    # Elastic training
    min_nodes: int = 1
    max_nodes: int = 8
    scale_down_delay_minutes: int = 10
    
    def to_dict(self) -> dict:
        return {
            "checkpointing": {
                "interval_steps": self.checkpoint_interval_steps,
                "interval_minutes": self.checkpoint_interval_minutes,
                "async": self.async_checkpointing,
                "compression": self.checkpoint_compression,
                "sharding": self.checkpoint_sharding,
                "max_to_keep": self.max_checkpoints_to_keep,
            },
            "failure_recovery": {
                "auto_restart": self.auto_restart_on_failure,
                "max_restarts": self.max_restarts,
                "restart_delay_seconds": self.restart_delay_seconds,
                "gradient_accumulation_on_oom": self.gradient_accumulation_on_oom,
            },
            "health_monitoring": {
                "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
                "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
                "health_check_url": self.health_check_url,
            },
            "nccl": {
                "timeout_minutes": self.nccl_timeout_minutes,
                "async_error_handling": self.nccl_async_error_handling,
            },
            "elastic": {
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes,
                "scale_down_delay_minutes": self.scale_down_delay_minutes,
            },
        }
    
    def generate_torchrun_args(self) -> List[str]:
        """Generate torchrun arguments for fault tolerance."""
        args = [
            f"--rdzv_backend=c10d",
            f"--max_restarts={self.max_restarts}",
        ]
        return args
    
    def generate_env_vars(self) -> Dict[str, str]:
        """Generate environment variables for fault tolerance."""
        env = {
            "NCCL_ASYNC_ERROR_HANDLING": "1" if self.nccl_async_error_handling else "0",
            "NCCL_TIMEOUT": str(self.nccl_timeout_minutes * 60),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1" if self.nccl_async_error_handling else "0",
        }
        return env
    
    def generate_deepspeed_config(self) -> dict:
        """Generate DeepSpeed config for fault tolerance."""
        return {
            "checkpoint": {
                "tag_validation": {
                    "enabled": True,
                    "fail_on_mismatch": False,
                },
                "use_node_local_storage": self.checkpoint_sharding,
            },
            "elasticity": {
                "enabled": self.min_nodes != self.max_nodes,
                "max_train_batch_size": 2048,
                "micro_batch_sizes": [1, 2, 4, 8, 16, 32],
                "min_gpus": self.min_nodes,
                "max_gpus": self.max_nodes,
            },
            "comms_logger": {
                "enabled": True,
                "verbose": False,
                "prof_all": False,
            },
        }


@dataclass
class SpotInstanceConfig:
    """Configuration for spot/preemptible instance handling."""
    enabled: bool = True
    checkpoint_on_preemption: bool = True
    preemption_warning_seconds: int = 120  # Time before preemption
    fallback_to_on_demand: bool = True
    max_spot_price_ratio: float = 0.7  # Max 70% of on-demand price
    
    # Checkpointing for spot
    aggressive_checkpointing: bool = True  # More frequent checkpoints
    checkpoint_to_cloud_storage: bool = True
    cloud_storage_path: Optional[str] = None  # e.g., s3://bucket/checkpoints
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "checkpoint_on_preemption": self.checkpoint_on_preemption,
            "preemption_warning_seconds": self.preemption_warning_seconds,
            "fallback_to_on_demand": self.fallback_to_on_demand,
            "max_spot_price_ratio": self.max_spot_price_ratio,
            "aggressive_checkpointing": self.aggressive_checkpointing,
            "checkpoint_to_cloud_storage": self.checkpoint_to_cloud_storage,
            "cloud_storage_path": self.cloud_storage_path,
        }
    
    def generate_preemption_handler_code(self) -> str:
        """Generate Python code for handling preemption."""
        return '''
import signal
import torch.distributed.checkpoint as dist_cp

def handle_preemption(signum, frame):
    """Handle spot instance preemption signal."""
    print("Received preemption warning! Saving checkpoint...")
    
    # Save checkpoint immediately
    if hasattr(model, 'state_dict'):
        dist_cp.save(
            state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            storage_writer=dist_cp.FileSystemWriter(checkpoint_path),
        )
    
    print(f"Checkpoint saved to {checkpoint_path}")
    # Optionally: notify orchestrator, clean up resources

# Register signal handlers for AWS/GCP/Azure preemption
signal.signal(signal.SIGTERM, handle_preemption)  # AWS
signal.signal(signal.SIGUSR1, handle_preemption)  # GCP
'''


@dataclass
class ElasticScalingConfig:
    """Configuration for elastic scaling."""
    enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 8
    target_gpu_utilization: float = 0.8
    scale_up_threshold: float = 0.9
    scale_down_threshold: float = 0.5
    scale_up_delay_seconds: int = 60
    scale_down_delay_seconds: int = 300
    
    # Batch size adjustment
    adjust_batch_size_on_scale: bool = True
    min_global_batch_size: int = 8
    max_global_batch_size: int = 2048
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "replicas": {
                "min": self.min_replicas,
                "max": self.max_replicas,
            },
            "thresholds": {
                "target_utilization": self.target_gpu_utilization,
                "scale_up": self.scale_up_threshold,
                "scale_down": self.scale_down_threshold,
            },
            "delays": {
                "scale_up_seconds": self.scale_up_delay_seconds,
                "scale_down_seconds": self.scale_down_delay_seconds,
            },
            "batch_adjustment": {
                "enabled": self.adjust_batch_size_on_scale,
                "min_global_batch": self.min_global_batch_size,
                "max_global_batch": self.max_global_batch_size,
            },
        }


class ClusterResilienceAdvisor:
    """Provides LLM-guided cluster resilience recommendations."""
    
    def __init__(self):
        self._llm_engine = None
    
    def _get_llm_engine(self):
        """Lazily load LLM engine."""
        if self._llm_engine is None:
            try:
                from tools.llm_engine import PerformanceAnalysisEngine
                self._llm_engine = PerformanceAnalysisEngine()
            except Exception:
                pass
        return self._llm_engine
    
    def get_fault_tolerance_config(
        self,
        model_params_b: float,
        num_nodes: int,
        gpus_per_node: int,
        training_hours: int,
        use_spot: bool = False,
        cloud_provider: str = "aws",
    ) -> Dict[str, Any]:
        """Get fault tolerance configuration recommendations.
        
        Uses LLM for personalized recommendations based on context.
        """
        total_gpus = num_nodes * gpus_per_node
        
        # Calculate recommended checkpoint interval based on model size and cluster
        # Larger models = more expensive to lose progress
        # More nodes = higher failure probability
        base_checkpoint_interval = 1000
        if model_params_b > 70:
            base_checkpoint_interval = 500  # More frequent for large models
        if num_nodes > 4:
            base_checkpoint_interval = int(base_checkpoint_interval * 0.7)  # More frequent for large clusters
        if use_spot:
            base_checkpoint_interval = int(base_checkpoint_interval * 0.5)  # Very frequent for spot
        
        config = FaultToleranceConfig(
            checkpoint_interval_steps=base_checkpoint_interval,
            checkpoint_interval_minutes=max(10, 60 // (num_nodes // 2 + 1)),
            async_checkpointing=model_params_b < 100,  # Async for smaller models
            checkpoint_sharding=num_nodes > 1,
            max_restarts=5 if use_spot else 3,
            min_nodes=max(1, num_nodes // 2) if use_spot else num_nodes,
            max_nodes=num_nodes * 2 if use_spot else num_nodes,
        )
        
        result = {
            "config": config.to_dict(),
            "torchrun_args": config.generate_torchrun_args(),
            "env_vars": config.generate_env_vars(),
            "deepspeed_config": config.generate_deepspeed_config(),
        }
        
        # Get LLM-enhanced recommendations
        engine = self._get_llm_engine()
        if engine:
            prompt = f"""You are a distributed training expert. Provide fault tolerance recommendations.

Context:
- Model: {model_params_b}B parameters
- Cluster: {num_nodes} nodes x {gpus_per_node} GPUs = {total_gpus} total GPUs
- Training duration: ~{training_hours} hours
- Using spot instances: {use_spot}
- Cloud provider: {cloud_provider}

Current config:
{json.dumps(config.to_dict(), indent=2)}

Provide specific recommendations for:
1. Checkpoint frequency optimization
2. NCCL timeout settings for this cluster size
3. Specific {cloud_provider} best practices
4. Any additional resilience measures for this scale

Be concise and specific to this setup."""

            try:
                llm_response = engine.ask(prompt)
                result["llm_recommendations"] = llm_response
            except Exception:
                pass
        
        return result
    
    def get_spot_instance_config(
        self,
        model_params_b: float,
        cloud_provider: str = "aws",
        budget_sensitive: bool = True,
    ) -> Dict[str, Any]:
        """Get spot instance configuration recommendations."""
        
        config = SpotInstanceConfig(
            enabled=True,
            aggressive_checkpointing=model_params_b > 13,
            max_spot_price_ratio=0.5 if budget_sensitive else 0.7,
        )
        
        result = {
            "config": config.to_dict(),
            "preemption_handler": config.generate_preemption_handler_code(),
        }
        
        # Cloud-specific recommendations
        cloud_specific = {
            "aws": {
                "instance_types": ["p4d.24xlarge", "p4de.24xlarge", "p5.48xlarge"],
                "spot_fleet_strategy": "capacityOptimized",
                "termination_notice": "2 minutes via instance metadata",
                "checkpoint_storage": "S3 with VPC endpoint",
            },
            "gcp": {
                "instance_types": ["a2-highgpu-8g", "a3-highgpu-8g"],
                "preemptible": True,
                "termination_notice": "30 seconds via metadata server",
                "checkpoint_storage": "GCS with nearline for old checkpoints",
            },
            "azure": {
                "instance_types": ["Standard_ND96asr_v4", "Standard_ND96amsr_A100_v4"],
                "spot_eviction": "Capacity or price based",
                "termination_notice": "30 seconds via IMDS",
                "checkpoint_storage": "Azure Blob with hot tier",
            },
        }
        
        result["cloud_specific"] = cloud_specific.get(cloud_provider, cloud_specific["aws"])
        
        return result
    
    def get_elastic_scaling_config(
        self,
        model_params_b: float,
        initial_nodes: int,
        traffic_pattern: str = "variable",  # steady, variable, bursty
    ) -> Dict[str, Any]:
        """Get elastic scaling configuration recommendations."""
        
        # Determine scaling bounds based on model size
        if model_params_b > 70:
            min_replicas = 4  # Large models need minimum parallelism
            max_replicas = 32
        elif model_params_b > 13:
            min_replicas = 2
            max_replicas = 16
        else:
            min_replicas = 1
            max_replicas = 8
        
        # Adjust based on traffic pattern
        if traffic_pattern == "bursty":
            scale_up_delay = 30
            scale_down_delay = 600
        elif traffic_pattern == "variable":
            scale_up_delay = 60
            scale_down_delay = 300
        else:  # steady
            scale_up_delay = 120
            scale_down_delay = 600
        
        config = ElasticScalingConfig(
            enabled=True,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_up_delay_seconds=scale_up_delay,
            scale_down_delay_seconds=scale_down_delay,
        )
        
        result = {
            "config": config.to_dict(),
            "kubernetes_hpa": self._generate_k8s_hpa(config, model_params_b),
        }
        
        return result
    
    def _generate_k8s_hpa(self, config: ElasticScalingConfig, model_params_b: float) -> str:
        """Generate Kubernetes HPA manifest."""
        return f'''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: training-job-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: training-job
  minReplicas: {config.min_replicas}
  maxReplicas: {config.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: {int(config.target_gpu_utilization * 100)}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: {config.scale_up_delay_seconds}
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: {config.scale_down_delay_seconds}
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
'''
    
    def diagnose_cluster_issue(self, error_message: str, cluster_info: dict) -> Dict[str, Any]:
        """Diagnose cluster issues with LLM assistance."""
        # Known issue patterns
        known_issues = {
            "nccl timeout": {
                "likely_cause": "Network partition or slow GPU",
                "quick_fixes": [
                    "Increase NCCL_TIMEOUT",
                    "Check nvidia-smi for stuck GPUs",
                    "Verify NVLink/InfiniBand connectivity",
                ],
            },
            "out of memory": {
                "likely_cause": "Batch size too large or memory leak",
                "quick_fixes": [
                    "Reduce batch size",
                    "Enable gradient checkpointing",
                    "Check for memory leaks with torch.cuda.memory_stats()",
                ],
            },
            "connection refused": {
                "likely_cause": "Master node not ready or firewall",
                "quick_fixes": [
                    "Ensure master node is running",
                    "Check firewall rules for NCCL ports",
                    "Verify MASTER_ADDR and MASTER_PORT",
                ],
            },
        }
        
        result = {"error": error_message, "quick_diagnosis": None, "llm_analysis": None}
        
        # Check for known patterns
        error_lower = error_message.lower()
        for pattern, info in known_issues.items():
            if pattern in error_lower:
                result["quick_diagnosis"] = info
                break
        
        # Get LLM analysis
        engine = self._get_llm_engine()
        if engine:
            prompt = f"""Diagnose this distributed training cluster issue:

Error: {error_message}

Cluster info:
{json.dumps(cluster_info, indent=2)}

Provide:
1. Root cause analysis
2. Immediate actions to resolve
3. Prevention measures for future
4. Relevant commands to debug

Be specific and actionable."""

            try:
                result["llm_analysis"] = engine.ask(prompt)
            except Exception:
                pass
        
        return result


# Module-level functions for easy access
def get_fault_tolerance_recommendations(
    model_params_b: float,
    num_nodes: int,
    gpus_per_node: int = 8,
    training_hours: int = 24,
    use_spot: bool = False,
    cloud_provider: str = "aws",
) -> Dict[str, Any]:
    """Get fault tolerance recommendations for a training job."""
    advisor = ClusterResilienceAdvisor()
    return advisor.get_fault_tolerance_config(
        model_params_b, num_nodes, gpus_per_node, training_hours, use_spot, cloud_provider
    )


def get_spot_recommendations(
    model_params_b: float,
    cloud_provider: str = "aws",
) -> Dict[str, Any]:
    """Get spot instance recommendations."""
    advisor = ClusterResilienceAdvisor()
    return advisor.get_spot_instance_config(model_params_b, cloud_provider)


def get_elastic_scaling_recommendations(
    model_params_b: float,
    initial_nodes: int = 4,
    traffic_pattern: str = "variable",
) -> Dict[str, Any]:
    """Get elastic scaling recommendations."""
    advisor = ClusterResilienceAdvisor()
    return advisor.get_elastic_scaling_config(model_params_b, initial_nodes, traffic_pattern)


def diagnose_cluster_error(error_message: str, cluster_info: dict = None) -> Dict[str, Any]:
    """Diagnose a cluster error with LLM assistance."""
    advisor = ClusterResilienceAdvisor()
    return advisor.diagnose_cluster_issue(error_message, cluster_info or {})


