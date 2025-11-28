#!/usr/bin/env python3
"""
Extra Utilities for Parallelism Planning

Additional features:
- Training time estimator
- Checkpoint size calculator
- SLURM/PBS job script generation
- Scaling projections
- Model comparison
- Configuration export/import
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class TrainingTimeEstimate:
    """Estimated training time for a configuration."""
    
    total_tokens: int
    tokens_per_second: float
    num_gpus: int
    
    # Time estimates
    seconds: float
    hours: float
    days: float
    
    # Cost estimates
    gpu_hours: float
    estimated_cost: float
    
    # Checkpointing
    checkpoint_interval_tokens: int
    num_checkpoints: int
    checkpoint_size_gb: float
    total_checkpoint_storage_gb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "num_gpus": self.num_gpus,
            "time": {
                "seconds": self.seconds,
                "hours": self.hours,
                "days": self.days,
            },
            "cost": {
                "gpu_hours": self.gpu_hours,
                "estimated_cost": self.estimated_cost,
            },
            "checkpoints": {
                "interval_tokens": self.checkpoint_interval_tokens,
                "num_checkpoints": self.num_checkpoints,
                "size_per_checkpoint_gb": self.checkpoint_size_gb,
                "total_storage_gb": self.total_checkpoint_storage_gb,
            },
        }


class TrainingEstimator:
    """Estimate training time and costs."""
    
    def __init__(self, gpu_hourly_cost: float = 4.0):
        self.gpu_hourly_cost = gpu_hourly_cost
    
    def estimate(
        self,
        total_tokens: int,
        tokens_per_second: float,
        num_gpus: int,
        model_params_billion: float,
        checkpoint_interval_tokens: int = 1_000_000_000,  # 1B tokens
        precision_bytes: int = 2,  # BF16
    ) -> TrainingTimeEstimate:
        """Estimate training time and resources.
        
        Args:
            total_tokens: Total tokens to train on
            tokens_per_second: Throughput in tokens/second
            num_gpus: Number of GPUs used
            model_params_billion: Model size in billions
            checkpoint_interval_tokens: Save checkpoint every N tokens
            precision_bytes: Bytes per parameter (2 for BF16, 4 for FP32)
        """
        # Time calculations
        seconds = total_tokens / tokens_per_second if tokens_per_second > 0 else 0
        hours = seconds / 3600
        days = hours / 24
        
        # Cost calculations
        gpu_hours = hours * num_gpus
        estimated_cost = gpu_hours * self.gpu_hourly_cost
        
        # Checkpoint calculations
        # Checkpoint size: params + optimizer states (2x for Adam momentum/variance)
        # With ZeRO-3/FSDP, checkpoint is sharded but we estimate full size
        checkpoint_size_gb = model_params_billion * 1e9 * (precision_bytes + 8) / 1e9  # params + optimizer
        num_checkpoints = max(1, total_tokens // checkpoint_interval_tokens)
        total_checkpoint_storage = checkpoint_size_gb * min(num_checkpoints, 5)  # Keep last 5
        
        return TrainingTimeEstimate(
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            num_gpus=num_gpus,
            seconds=seconds,
            hours=hours,
            days=days,
            gpu_hours=gpu_hours,
            estimated_cost=estimated_cost,
            checkpoint_interval_tokens=checkpoint_interval_tokens,
            num_checkpoints=num_checkpoints,
            checkpoint_size_gb=checkpoint_size_gb,
            total_checkpoint_storage_gb=total_checkpoint_storage,
        )
    
    def format_estimate(self, estimate: TrainingTimeEstimate) -> str:
        """Format estimate as human-readable text."""
        lines = [
            "=" * 60,
            "TRAINING TIME & COST ESTIMATE",
            "=" * 60,
            "",
            f"Training Tokens: {estimate.total_tokens:,}",
            f"Throughput: {estimate.tokens_per_second:,.0f} tokens/sec",
            f"GPUs: {estimate.num_gpus}",
            "",
            "TIME ESTIMATE:",
            f"  {estimate.hours:.1f} hours ({estimate.days:.2f} days)",
            "",
            "COST ESTIMATE:",
            f"  GPU Hours: {estimate.gpu_hours:,.1f}",
            f"  Estimated Cost: ${estimate.estimated_cost:,.2f}",
            f"  (at ${self.gpu_hourly_cost:.2f}/GPU-hour)",
            "",
            "CHECKPOINTING:",
            f"  Checkpoint Size: {estimate.checkpoint_size_gb:.1f} GB",
            f"  Checkpoints: {estimate.num_checkpoints}",
            f"  Storage Needed: {estimate.total_checkpoint_storage_gb:.1f} GB",
            "=" * 60,
        ]
        return "\n".join(lines)


class ScalingProjector:
    """Project performance for different GPU counts."""
    
    def __init__(self):
        # Scaling efficiency assumptions
        self.dp_efficiency = {1: 1.0, 2: 0.98, 4: 0.95, 8: 0.92, 16: 0.88, 32: 0.82, 64: 0.75}
        self.tp_efficiency = {1: 1.0, 2: 0.95, 4: 0.88, 8: 0.80}
        self.pp_efficiency = {1: 1.0, 2: 0.85, 4: 0.72, 8: 0.60}
    
    def project(
        self,
        base_throughput: float,
        base_gpus: int,
        target_gpus: int,
        scaling_type: str = "dp",  # "dp", "tp", "pp", "hybrid"
    ) -> Dict[str, Any]:
        """Project throughput for different GPU counts.
        
        Args:
            base_throughput: Current throughput (tokens/sec)
            base_gpus: Current GPU count
            target_gpus: Target GPU count
            scaling_type: How to scale ("dp", "tp", "pp", "hybrid")
        """
        scale_factor = target_gpus / base_gpus
        
        if scaling_type == "dp":
            # Data parallel scales linearly with efficiency loss
            efficiency = self.dp_efficiency.get(target_gpus, 0.7)
            projected = base_throughput * scale_factor * efficiency
        elif scaling_type == "tp":
            # TP doesn't scale throughput, reduces latency
            efficiency = self.tp_efficiency.get(target_gpus, 0.7)
            projected = base_throughput * efficiency  # Slight loss
        elif scaling_type == "pp":
            # PP scales with bubble overhead
            efficiency = self.pp_efficiency.get(target_gpus, 0.5)
            projected = base_throughput * scale_factor * efficiency
        else:
            # Hybrid - assume mix of DP/TP
            efficiency = 0.85
            projected = base_throughput * scale_factor * efficiency
        
        return {
            "base_throughput": base_throughput,
            "base_gpus": base_gpus,
            "target_gpus": target_gpus,
            "scaling_type": scaling_type,
            "projected_throughput": projected,
            "efficiency": projected / (base_throughput * scale_factor),
            "speedup": projected / base_throughput,
        }
    
    def project_range(
        self,
        base_throughput: float,
        base_gpus: int,
        gpu_counts: List[int],
    ) -> List[Dict[str, Any]]:
        """Project for multiple GPU counts."""
        return [
            self.project(base_throughput, base_gpus, n, "dp")
            for n in gpu_counts
        ]


class JobScriptGenerator:
    """Generate SLURM/PBS job scripts."""
    
    def generate_slurm(
        self,
        job_name: str,
        num_nodes: int,
        gpus_per_node: int,
        time_hours: int,
        partition: str = "gpu",
        account: Optional[str] = None,
        script: str = "train.py",
        conda_env: Optional[str] = None,
        extra_sbatch: Optional[Dict[str, str]] = None,
        launch_command: Optional[str] = None,
    ) -> str:
        """Generate SLURM job script.
        
        Args:
            job_name: Job name
            num_nodes: Number of nodes
            gpus_per_node: GPUs per node
            time_hours: Walltime in hours
            partition: SLURM partition
            account: Account/allocation
            script: Training script
            conda_env: Conda environment to activate
            extra_sbatch: Additional SBATCH directives
            launch_command: Custom launch command (default: torchrun)
        """
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --nodes={num_nodes}",
            f"#SBATCH --gpus-per-node={gpus_per_node}",
            f"#SBATCH --ntasks-per-node={gpus_per_node}",
            f"#SBATCH --time={time_hours}:00:00",
            f"#SBATCH --partition={partition}",
            "#SBATCH --output=%x_%j.out",
            "#SBATCH --error=%x_%j.err",
        ]
        
        if account:
            lines.append(f"#SBATCH --account={account}")
        
        if extra_sbatch:
            for key, value in extra_sbatch.items():
                lines.append(f"#SBATCH --{key}={value}")
        
        lines.extend([
            "",
            "# Environment setup",
            "module purge",
            "module load cuda/12.4",
        ])
        
        if conda_env:
            lines.extend([
                f"source activate {conda_env}",
                "",
            ])
        
        lines.extend([
            "# Distributed setup",
            "export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)",
            "export MASTER_PORT=29500",
            "export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))",
            "",
            "# NCCL settings",
            'export NCCL_DEBUG=WARN',
            'export NCCL_IB_DISABLE=0',
            'export CUDA_DEVICE_MAX_CONNECTIONS=1',
            "",
            "echo \"Starting training on $SLURM_NNODES nodes with $SLURM_GPUS_PER_NODE GPUs each\"",
            "echo \"Master: $MASTER_ADDR:$MASTER_PORT\"",
            "",
        ])
        
        if launch_command:
            lines.append(launch_command)
        else:
            lines.extend([
                "srun torchrun \\",
                f"    --nnodes=$SLURM_NNODES \\",
                f"    --nproc_per_node=$SLURM_GPUS_PER_NODE \\",
                "    --rdzv_backend=c10d \\",
                "    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\",
                f"    {script}",
            ])
        
        return "\n".join(lines)
    
    def generate_pbs(
        self,
        job_name: str,
        num_nodes: int,
        gpus_per_node: int,
        time_hours: int,
        queue: str = "gpu",
        account: Optional[str] = None,
        script: str = "train.py",
    ) -> str:
        """Generate PBS job script."""
        lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -l select={num_nodes}:ngpus={gpus_per_node}",
            f"#PBS -l walltime={time_hours}:00:00",
            f"#PBS -q {queue}",
            "#PBS -j oe",
        ]
        
        if account:
            lines.append(f"#PBS -A {account}")
        
        lines.extend([
            "",
            "cd $PBS_O_WORKDIR",
            "",
            "# Environment setup",
            "module load cuda/12.4",
            "",
            "# Get node list",
            "NODES=$(cat $PBS_NODEFILE | sort -u)",
            "MASTER_ADDR=$(echo $NODES | head -n 1)",
            "export MASTER_PORT=29500",
            "",
            f"mpirun -np {num_nodes * gpus_per_node} \\",
            "    --hostfile $PBS_NODEFILE \\",
            f"    python {script}",
        ])
        
        return "\n".join(lines)


class ConfigExporter:
    """Export/import parallelism configurations."""
    
    def export_config(
        self,
        config: Dict[str, Any],
        path: Path,
        format: str = "json",
    ) -> None:
        """Export configuration to file."""
        if format == "json":
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
        elif format == "yaml":
            import yaml
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def import_config(self, path: Path) -> Dict[str, Any]:
        """Import configuration from file."""
        suffix = path.suffix.lower()
        
        with open(path) as f:
            if suffix in (".json",):
                return json.load(f)
            elif suffix in (".yaml", ".yml"):
                import yaml
                return yaml.safe_load(f)
            else:
                # Try JSON first, then YAML
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    import yaml
                    f.seek(0)
                    return yaml.safe_load(f)
    
    def create_training_config(
        self,
        model: str,
        tp: int,
        pp: int,
        dp: int,
        sharding: str,
        batch_size: int,
        seq_length: int,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a complete training configuration."""
        return {
            "model": model,
            "parallelism": {
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "data_parallel": dp,
                "sharding": sharding,
            },
            "training": {
                "micro_batch_size": batch_size,
                "sequence_length": seq_length,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
            },
            "extra": kwargs,
        }


class ModelComparator:
    """Compare parallelism strategies for multiple models."""
    
    def __init__(self):
        from .model_analyzer import ModelAnalyzer
        from .strategy_optimizer import StrategyOptimizer
        
        self.analyzer = ModelAnalyzer()
        self.optimizer = StrategyOptimizer()
    
    def compare(
        self,
        models: List[str],
        topology,
        batch_size: int = 1,
        seq_length: int = 2048,
        is_training: bool = False,
    ) -> Dict[str, Any]:
        """Compare parallelism recommendations for multiple models."""
        results = {}
        
        for model_name in models:
            try:
                model = self.analyzer.analyze(model_name)
                recs = self.optimizer.recommend(
                    topology=topology,
                    model=model,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    is_training=is_training,
                    max_strategies=1,
                )
                
                if recs:
                    best = recs[0]
                    results[model_name] = {
                        "params_b": model.total_params_billion,
                        "best_config": f"TP{best.strategy.tp}_PP{best.strategy.pp}_DP{best.strategy.dp}",
                        "memory_per_gpu_gb": best.analysis.memory_per_gpu_gb,
                        "throughput_tps": best.analysis.estimated_throughput_tps,
                        "score": best.score,
                    }
                else:
                    results[model_name] = {"error": "No valid configuration"}
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        return results
    
    def format_comparison(self, results: Dict[str, Any]) -> str:
        """Format comparison as table."""
        lines = [
            "=" * 90,
            "MODEL COMPARISON",
            "=" * 90,
            f"{'Model':<20} {'Params':<10} {'Best Config':<20} {'Memory/GPU':<12} {'Throughput':<15}",
            "-" * 90,
        ]
        
        for model, data in results.items():
            if "error" in data:
                lines.append(f"{model:<20} ERROR: {data['error']}")
            else:
                lines.append(
                    f"{model:<20} {data['params_b']:>7.1f}B   {data['best_config']:<20} "
                    f"{data['memory_per_gpu_gb']:>8.1f} GB   {data['throughput_tps']:>12,.0f} tps"
                )
        
        lines.append("=" * 90)
        return "\n".join(lines)



