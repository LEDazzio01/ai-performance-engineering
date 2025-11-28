"""
Cluster Commands - SLURM, cloud cost, scaling predictions, diagnostics.

Provides commands for:
- SLURM script generation
- Cloud cost calculation
- Multi-GPU scaling predictions
- Cluster diagnostics
- Power/energy analysis
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional, Dict, Any

from tools.core.engine import get_engine


def _print_header(title: str, emoji: str = "🖥️"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


# Cloud pricing (approximate, on-demand)
CLOUD_PRICING = {
    "aws": {
        "p4d.24xlarge": {"gpus": 8, "gpu_type": "A100-40GB", "hourly": 32.77},
        "p5.48xlarge": {"gpus": 8, "gpu_type": "H100-80GB", "hourly": 98.32},
    },
    "gcp": {
        "a2-highgpu-8g": {"gpus": 8, "gpu_type": "A100-40GB", "hourly": 29.39},
        "a3-highgpu-8g": {"gpus": 8, "gpu_type": "H100-80GB", "hourly": 85.00},
    },
    "azure": {
        "NC96ads_A100_v4": {"gpus": 4, "gpu_type": "A100-80GB", "hourly": 14.69},
        "ND96isr_H100_v5": {"gpus": 8, "gpu_type": "H100-80GB", "hourly": 98.00},
    },
    "lambda": {
        "gpu_1x_h100_pcie": {"gpus": 1, "gpu_type": "H100-PCIe", "hourly": 2.49},
        "gpu_8x_h100_sxm5": {"gpus": 8, "gpu_type": "H100-SXM5", "hourly": 23.92},
    },
}


# =============================================================================
# SLURM SCRIPT GENERATION
# =============================================================================

def slurm_generate(args) -> int:
    """Generate SLURM script for training."""
    _print_header("SLURM Script Generator", "📜")
    
    nodes = getattr(args, 'nodes', 1)
    gpus = getattr(args, 'gpus', 8)
    model = getattr(args, 'model', 'llama-70b')
    time_limit = getattr(args, 'time', '24:00:00')
    partition = getattr(args, 'partition', 'gpu')
    
    print(f"  Nodes: {nodes}")
    print(f"  GPUs per node: {gpus}")
    print(f"  Model: {model}")
    print(f"  Time: {time_limit}")
    
    try:
        core = get_engine().cluster
        generated = core.slurm(model=model, nodes=nodes, gpus=gpus, framework="pytorch")
        script = generated.get("script")
        if script:
            print("\n" + "=" * 70)
            print(script)
            print("=" * 70)
        else:
            raise ValueError("No script returned from core.slurm")
    except Exception:
        # Fallback to local template if core is unavailable
        total_gpus = nodes * gpus
        # Estimate model size
        model_lower = model.lower()
        if '70b' in model_lower or '65b' in model_lower:
            model_size = 70
        elif '13b' in model_lower:
            model_size = 13
        elif '7b' in model_lower:
            model_size = 7
        else:
            model_size = 7
        
        # Recommend parallelism
        if model_size >= 70:
            tp = min(8, gpus)
            pp = max(1, total_gpus // (tp * 4))
            dp = total_gpus // (tp * pp)
        else:
            tp = 1
            pp = 1
            dp = total_gpus
    
        script = f'''#!/bin/bash
#SBATCH --job-name={model.replace("/", "-")}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus}
#SBATCH --gpus-per-node={gpus}
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Environment setup
module load cuda/12.1
module load nccl/2.18
source ~/venv/bin/activate

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Distributed settings
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# Launch training
srun --ntasks=$SLURM_NTASKS \\
     --ntasks-per-node=$SLURM_GPUS_PER_NODE \\
     python -m torch.distributed.run \\
     --nproc_per_node=$SLURM_GPUS_PER_NODE \\
     --nnodes=$SLURM_NNODES \\
     --node_rank=$SLURM_NODEID \\
     --master_addr=$MASTER_ADDR \\
     --master_port=$MASTER_PORT \\
     train.py \\
     --model {model} \\
     --tensor_parallel_size {tp} \\
     --pipeline_parallel_size {pp} \\
     --data_parallel_size {dp}
'''
    
        print("\n" + "=" * 70)
        print(script)
        print("=" * 70)
    
    # Save option
    output = getattr(args, 'output', None)
    if output:
        with open(output, 'w') as f:
            f.write(script)
        print(f"\n  ✅ Saved to: {output}")
    else:
        print(f"\n  💡 Save with: aisp cluster slurm --output train.slurm")
    
    return 0


# =============================================================================
# CLOUD COST
# =============================================================================

def cloud_cost(args) -> int:
    """Calculate cloud training costs."""
    _print_header("Cloud Cost Calculator", "💰")
    
    model_size = getattr(args, 'model_size', 70)
    tokens = getattr(args, 'tokens', 1e12)  # 1T tokens
    batch_size = getattr(args, 'batch_size', 1024)
    
    print(f"  Model Size: {model_size}B parameters")
    print(f"  Training Tokens: {tokens/1e12:.1f}T")
    print(f"  Batch Size: {batch_size}")
    
    try:
        estimate = get_engine().cost.cloud_estimate({
            "model_size": model_size,
            "tokens": tokens,
            "batch_size": batch_size,
        })
        print(json.dumps(estimate, indent=2))
    except Exception:
        # Fallback to static estimate
        flops = 6 * model_size * 1e9 * tokens
        
        print(f"\n  Estimated FLOPs: {flops/1e21:.1f} ZFLOPs")
        
        print("\n  Cloud Cost Comparison:")
        print("-" * 70)
        print(f"  {'Provider':<10} {'Instance':<25} {'GPUs':<6} {'$/hr':<8} {'Days':<8} {'Total $':<12}")
        print("-" * 70)
        
        for provider, instances in CLOUD_PRICING.items():
            for instance, info in instances.items():
                num_gpus = info['gpus']
                gpu_type = info['gpu_type']
                hourly = info['hourly']
                
                # Estimate TFLOPS based on GPU type
                if 'H100' in gpu_type:
                    tflops = 1979  # H100 FP8
                elif 'A100-80' in gpu_type:
                    tflops = 312  # A100 BF16
                else:
                    tflops = 156  # A100-40GB
                
                total_tflops = num_gpus * tflops
                
                # Time estimate (with efficiency factor ~0.4)
                hours = flops / (total_tflops * 1e12 * 3600 * 0.4)
                days = hours / 24
                total_cost = hours * hourly
                
                print(f"  {provider:<10} {instance:<25} {num_gpus:<6} ${hourly:<7.2f} {days:<8.1f} ${total_cost:,.0f}")
        
        print("-" * 70)
        print(f"\n  💡 Costs are estimates. Actual costs depend on efficiency and spot pricing.")
        print(f"     Spot instances can reduce costs by 60-80%.")
    
    return 0


# =============================================================================
# SCALING PREDICTIONS
# =============================================================================

def scaling_predict(args) -> int:
    """Predict multi-GPU scaling efficiency."""
    _print_header("Scaling Prediction", "📈")
    
    gpus = getattr(args, 'gpus', 8)
    model_size = getattr(args, 'model_size', 70)
    batch_size = getattr(args, 'batch_size', 32)
    
    print(f"  Model Size: {model_size}B")
    print(f"  Target GPUs: {gpus}")
    print(f"  Batch Size: {batch_size}")
    
    # Simple scaling model based on empirical data
    # Efficiency drops due to communication overhead
    
    print("\n  Predicted Scaling Efficiency:")
    print("-" * 50)
    print(f"  {'GPUs':<8} {'Efficiency':<15} {'Speedup':<12} {'Throughput':<15}")
    print("-" * 50)
    
    base_throughput = 1000  # tokens/sec baseline
    
    for n in [1, 2, 4, 8, 16, 32, 64]:
        if n > gpus:
            break
        
        # Communication overhead model
        if n == 1:
            efficiency = 1.0
        else:
            # Overhead increases with sqrt(n) for AllReduce
            comm_overhead = 0.1 * (n ** 0.5)
            efficiency = max(0.5, 1.0 - comm_overhead)
        
        speedup = n * efficiency
        throughput = base_throughput * speedup
        
        eff_str = f"{efficiency*100:.0f}%"
        print(f"  {n:<8} {eff_str:<15} {speedup:.1f}x{'':<8} {throughput:.0f} tok/s")
    
    print("-" * 50)
    print(f"\n  💡 Actual efficiency depends on:")
    print("     • Interconnect (NVLink > PCIe > Ethernet)")
    print("     • Model architecture (MoE has lower efficiency)")
    print("     • Batch size (larger = better scaling)")
    
    return 0


# =============================================================================
# CLUSTER DIAGNOSTICS
# =============================================================================

def cluster_diagnose(args) -> int:
    """Diagnose cluster issues."""
    _print_header("Cluster Diagnostics", "🔍")
    
    issues = []
    warnings = []
    
    # Check SLURM
    try:
        result = subprocess.run(['sinfo', '-V'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✅ SLURM: {result.stdout.strip()}")
        else:
            warnings.append("SLURM not responding")
    except FileNotFoundError:
        print("  ℹ️ SLURM: Not installed (standalone mode)")
    except Exception as e:
        warnings.append(f"SLURM check failed: {e}")
    
    # Check InfiniBand
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse active ports
            active = result.stdout.count('State: Active')
            print(f"  ✅ InfiniBand: {active} active ports")
        else:
            print("  ℹ️ InfiniBand: Not available")
    except FileNotFoundError:
        print("  ℹ️ InfiniBand: Not installed")
    except Exception as e:
        warnings.append(f"IB check failed: {e}")
    
    # Check GPUs
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            print(f"  ✅ GPUs: {len(gpus)} detected")
            for i, gpu in enumerate(gpus):
                print(f"     GPU {i}: {gpu}")
        else:
            issues.append("nvidia-smi failed")
    except Exception as e:
        issues.append(f"GPU check failed: {e}")
    
    # Check NCCL
    try:
        import torch.distributed as dist
        if dist.is_nccl_available():
            print("  ✅ NCCL: Available")
        else:
            warnings.append("NCCL not available")
    except:
        warnings.append("Could not check NCCL")
    
    # Summary
    print("\n  Summary:")
    print("-" * 50)
    
    if issues:
        print(f"  ❌ Issues ({len(issues)}):")
        for issue in issues:
            print(f"      • {issue}")
    
    if warnings:
        print(f"  ⚠️ Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"      • {warning}")
    
    if not issues and not warnings:
        print("  ✅ No issues detected")
    
    return 0


# =============================================================================
# POWER ANALYSIS
# =============================================================================

def power_analysis(args) -> int:
    """Analyze power consumption and efficiency."""
    _print_header("Power Analysis", "⚡")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,power.draw,power.limit,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            print("  ❌ nvidia-smi failed")
            return 1
        
        total_power = 0
        total_limit = 0
        
        print(f"  {'GPU':<8} {'Name':<20} {'Power':<12} {'Limit':<10} {'Temp':<8} {'Util':<8}")
        print("-" * 70)
        
        for i, line in enumerate(result.stdout.strip().split('\n')):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                name, power, limit, temp, util = parts[:5]
                power = float(power)
                limit = float(limit)
                total_power += power
                total_limit += limit
                
                efficiency = (power / limit * 100) if limit > 0 else 0
                print(f"  {i:<8} {name[:20]:<20} {power:.0f}W{'':<6} {limit:.0f}W{'':<4} {temp}°C{'':<3} {util}%")
        
        print("-" * 70)
        print(f"  Total: {total_power:.0f}W / {total_limit:.0f}W ({total_power/total_limit*100:.0f}%)")
        
        # Cost estimation
        kwh_cost = 0.12  # $/kWh average
        daily_kwh = total_power * 24 / 1000
        daily_cost = daily_kwh * kwh_cost
        
        print(f"\n  Energy Cost (at ${kwh_cost}/kWh):")
        print(f"    Daily:   {daily_kwh:.1f} kWh = ${daily_cost:.2f}")
        print(f"    Monthly: {daily_kwh*30:.0f} kWh = ${daily_cost*30:.2f}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1
    
    return 0


# =============================================================================
# COMMAND REGISTRATION
# =============================================================================

def register_commands(subparsers):
    """Register cluster commands."""
    cluster_parser = subparsers.add_parser("cluster", help="Cluster management and cloud")
    cluster_subparsers = cluster_parser.add_subparsers(dest="cluster_command")
    
    # SLURM
    slurm_p = cluster_subparsers.add_parser("slurm", help="Generate SLURM script")
    slurm_p.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    slurm_p.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    slurm_p.add_argument("--model", default="llama-70b", help="Model name")
    slurm_p.add_argument("--time", default="24:00:00", help="Time limit")
    slurm_p.add_argument("--partition", default="gpu", help="SLURM partition")
    slurm_p.add_argument("--output", "-o", help="Output file")
    slurm_p.set_defaults(func=slurm_generate)
    
    # Cloud cost
    cost_p = cluster_subparsers.add_parser("cost", help="Cloud cost calculator")
    cost_p.add_argument("--model-size", type=float, default=70, help="Model size in B")
    cost_p.add_argument("--tokens", type=float, default=1e12, help="Training tokens")
    cost_p.add_argument("--batch-size", type=int, default=1024, help="Global batch size")
    cost_p.set_defaults(func=cloud_cost)
    
    # Scaling
    scale_p = cluster_subparsers.add_parser("scaling", help="Predict scaling efficiency")
    scale_p.add_argument("--gpus", type=int, default=8, help="Target GPU count")
    scale_p.add_argument("--model-size", type=float, default=70, help="Model size in B")
    scale_p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    scale_p.set_defaults(func=scaling_predict)
    
    # Diagnostics
    diag_p = cluster_subparsers.add_parser("diagnose", help="Cluster diagnostics")
    diag_p.set_defaults(func=cluster_diagnose)
    
    # Power
    power_p = cluster_subparsers.add_parser("power", help="Power analysis")
    power_p.set_defaults(func=power_analysis)

