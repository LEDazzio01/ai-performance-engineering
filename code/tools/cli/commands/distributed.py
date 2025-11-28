"""
Distributed Commands - Parallelism planning, topology, NCCL tuning.

Uses tools/parallelism_planner/ and tools/optimization_intelligence.py.
"""

from __future__ import annotations

import json

from tools.core.engine import get_engine


def _print_header(title: str, emoji: str = "🌐"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def plan_parallelism(args) -> int:
    """Plan parallelism strategy (TP/PP/DP)."""
    _print_header("Parallelism Planning", "📐")
    
    model_size = getattr(args, 'model_size', 70)
    num_gpus = getattr(args, 'gpus', 8)
    num_nodes = getattr(args, 'nodes', 1)
    
    print(f"  Model: {model_size}B parameters")
    print(f"  GPUs: {num_gpus} ({num_nodes} node(s))")
    print("-" * 70)
    
    try:
        engine = get_engine()
        result = engine.distributed.plan(
            model_size=model_size,
            gpus=num_gpus,
            nodes=num_nodes,
        )
        
        if isinstance(result, dict):
            tp = result.get("tensor_parallel", 1)
            pp = result.get("pipeline_parallel", 1)
            dp = result.get("data_parallel", 1)
            
            print(f"\n  Recommended Strategy:")
            print(f"    Tensor Parallel (TP): {tp}")
            print(f"    Pipeline Parallel (PP): {pp}")
            print(f"    Data Parallel (DP): {dp}")
            
            if "communication" in result:
                comm = result["communication"]
                print(f"\n  Communication:")
                print(f"    Intra-node: {comm.get('intra_node', 'NVLink')}")
                print(f"    Inter-node: {comm.get('inter_node', 'N/A')}")
            
            if "launch_command" in result:
                print(f"\n  Launch Command:")
                print(f"    {result['launch_command']}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    return 0


def topology(args) -> int:
    """Analyze GPU topology."""
    _print_header("GPU Topology", "🔗")
    
    try:
        topo = get_engine().distributed.topology() or {}
        matrix = topo.get("topology_matrix") or topo.get("matrix")
        nvlink = topo.get("nvlink") or {}

        if matrix:
            print("\n  GPU Interconnect Matrix:")
            for line in str(matrix).splitlines():
                print(f"    {line}")
        else:
            print("  Topology matrix not available.")

        status = nvlink.get("status") or nvlink.get("nvlink_status")
        if status:
            print("\n  NVLink Status:")
            for line in str(status).splitlines():
                print(f"    {line}")
        else:
            print("\n  NVLink status not available.")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 1


def nccl_tuning(args) -> int:
    """NCCL tuning recommendations."""
    _print_header("NCCL Tuning", "📡")
    
    nodes = getattr(args, "nodes", 1)
    gpus = getattr(args, "gpus", 8)
    diagnose = getattr(args, "diagnose", False)

    try:
        recs = get_engine().distributed.nccl(nodes=nodes, gpus=gpus, diagnose=diagnose) or {}
        print(json.dumps(recs, indent=2))
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 1


def zero_config(args) -> int:
    """ZeRO/FSDP configuration helper."""
    _print_header("ZeRO/FSDP Configuration", "💾")
    
    model_size = getattr(args, 'model_size', 70)
    num_gpus = getattr(args, 'gpus', 8)
    
    print(f"  Model: {model_size}B parameters")
    print(f"  GPUs: {num_gpus}")
    try:
        recs = get_engine().distributed.fsdp(model=str(model_size))
        print(json.dumps(recs, indent=2))
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 1
