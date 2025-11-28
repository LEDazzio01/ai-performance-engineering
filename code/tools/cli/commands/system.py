"""
System Commands - GPU, environment, dependencies, preflight checks.

All system-level diagnostics and information gathering.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from tools.core.perf_core import get_core


def _print_header(title: str, emoji: str = "📊"):
    """Print a formatted header."""
    print(f"\n{emoji} {title}")
    print("=" * 70)


def _print_section(title: str):
    """Print a section divider."""
    print(f"\n  {title}")
    print("  " + "-" * 60)


# =============================================================================
# SYSTEM STATUS
# =============================================================================

def system_status(args) -> int:
    """Comprehensive system status - GPU, software, dependencies."""
    _print_header("System Status", "🚀")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    core = get_core()

    # GPU Status
    _print_section("GPU")
    try:
        info = core.get_gpu_info() or {}
        gpus = info.get("gpus") or [info] if info else []
        for i, gpu in enumerate(gpus):
            name = gpu.get("name", f"GPU {i}")
            temp = gpu.get("temperature", gpu.get("temperature_gpu_c", "N/A"))
            power = gpu.get("power", gpu.get("power_draw_w", "N/A"))
            util = gpu.get("utilization", gpu.get("utilization_gpu_pct", 0))
            mem_used = gpu.get("memory_used", 0) or gpu.get("memory_used_mb", 0)
            mem_total = gpu.get("memory_total", 1) or gpu.get("memory_total_mb", 1)
            mem_pct = (mem_used / mem_total) * 100 if mem_total else 0
            print(f"    {name}: Temp {temp}°C | Power {power}W | Util {util}% | Mem {mem_used/1024:.1f}/{mem_total/1024:.1f} GB ({mem_pct:.0f}%)")
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    # Software Stack
    _print_section("Software Stack")
    try:
        sw = core.get_software_info() or {}
        print(f"    PyTorch: {sw.get('pytorch','N/A')}")
        print(f"    CUDA:    {sw.get('cuda_runtime','N/A')} | Driver: {sw.get('driver_version','N/A')}")
        print(f"    Arch:    {sw.get('architecture','N/A')} (sm{sw.get('compute_capability','?')})")
    except Exception as e:
        print(f"    ❌ Software info error: {e}")
    
    # Key Libraries
    _print_section("Key Libraries")
    try:
        deps = core.get_dependency_health() or {}
        issues = deps.get("issues", [])
        if issues:
            print("    ⚠️ Issues detected:")
            for issue in issues:
                print(f"      • {issue}")
        else:
            print("    ✅ Dependencies healthy")
    except Exception as e:
        print(f"    ❌ Dependency check error: {e}")
    
    # LLM Status
    _print_section("LLM Backend")
    try:
        from tools.llm_engine import PerformanceAnalysisEngine
        engine = PerformanceAnalysisEngine()
        print(f"    ✅ LLM Available ({engine.config.provider}:{engine.config.model})")
    except Exception as e:
        print(f"    ⚠️ LLM not configured: {e}")
        print("       Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")
    
    print()
    return 0


# =============================================================================
# GPU INFO
# =============================================================================

def gpu_info(args) -> int:
    """Detailed GPU information."""
    _print_header("GPU Information", "🎮")
    
    core = get_core()
    try:
        info = core.get_gpu_info() or {}
        gpus = info.get("gpus") or [info] if info else []
        for i, gpu in enumerate(gpus):
            print(f"\n  GPU {i}: {gpu.get('name','')}")
            print(f"    Memory: {gpu.get('memory_total', gpu.get('memory_total_mb',0))/1024:.1f} GB")
            print(f"    PCIe:   Gen{gpu.get('pcie_gen','?')} x{gpu.get('pcie_width','?')}")
            print(f"    Compute: {gpu.get('compute_capability','?')}")
        
        # Topology
        _print_section("Topology")
        topo = core.get_gpu_topology()
        matrix = topo.get("topology_matrix") if isinstance(topo, dict) else None
        if matrix:
            for line in str(matrix).splitlines()[:15]:
                print(f"    {line}")
        else:
            print("    Topology info not available")
        
        # NVLink
        _print_section("NVLink Status")
        nv = core.get_nvlink_status()
        if nv.get("nvlink_available"):
            print("    NVLink: Connected")
            if nv.get("nvlink_status"):
                for line in nv["nvlink_status"].splitlines()[:10]:
                    print(f"    {line}")
        else:
            print("    NVLink: Not available")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()
    return 0


# =============================================================================
# ENVIRONMENT
# =============================================================================

def show_env(args) -> int:
    """Show relevant environment variables."""
    _print_header("Environment", "🌐")
    
    env_vars = [
        # CUDA
        ("CUDA_VISIBLE_DEVICES", "Visible GPUs"),
        ("CUDA_HOME", "CUDA installation"),
        ("CUDA_DEVICE_MAX_CONNECTIONS", "Max GPU connections"),
        
        # NCCL
        ("NCCL_DEBUG", "NCCL logging"),
        ("NCCL_IB_DISABLE", "InfiniBand disabled"),
        ("NCCL_P2P_DISABLE", "P2P disabled"),
        
        # PyTorch
        ("TORCH_CUDA_ARCH_LIST", "CUDA arch list"),
        ("PYTORCH_CUDA_ALLOC_CONF", "Allocator config"),
        
        # LLM
        ("OPENAI_API_KEY", "OpenAI API key"),
        ("ANTHROPIC_API_KEY", "Anthropic API key"),
        ("LLM_PROVIDER", "LLM provider"),
        ("LLM_ANALYSIS_ENABLED", "LLM analysis toggle"),
    ]
    
    _print_section("Performance-Related")
    for var, desc in env_vars:
        value = os.environ.get(var, "")
        if "KEY" in var and value:
            value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"    {var:35} = {value or '(not set)':20}  # {desc}")
    
    _print_section("Paths")
    print(f"    Python:     {sys.executable}")
    print(f"    Site-pkgs:  {next(iter(sys.path[1:2]), 'N/A')}")
    
    print()
    return 0


# =============================================================================
# DEPENDENCIES
# =============================================================================

def check_deps(args) -> int:
    """Check dependencies and their versions."""
    _print_header("Dependencies", "📦")
    
    deps = [
        # Core
        ("torch", "PyTorch", "2.3+"),
        ("numpy", "NumPy", "1.24+"),
        
        # GPU
        ("triton", "Triton", "3.0+"),
        ("flash_attn", "Flash Attention", "2.5+"),
        ("transformer_engine", "Transformer Engine", "1.0+"),
        
        # Distributed
        ("deepspeed", "DeepSpeed", "0.14+"),
        
        # Inference
        ("vllm", "vLLM", "0.4+"),
        
        # Profiling
        ("torch.profiler", "PyTorch Profiler", "built-in"),
    ]
    
    issues = []
    
    for pkg, name, recommended in deps:
        try:
            if "." in pkg:
                # Submodule
                parts = pkg.split(".")
                mod = __import__(parts[0])
                for part in parts[1:]:
                    mod = getattr(mod, part)
                version = "available"
            else:
                mod = __import__(pkg)
                version = getattr(mod, '__version__', 'unknown')
            
            print(f"    ✅ {name:25} {version:15} (recommended: {recommended})")
        except ImportError:
            print(f"    ❌ {name:25} {'not installed':15}")
            issues.append(name)
    
    if issues:
        print(f"\n  ⚠️ Missing: {', '.join(issues)}")
        print("     Install with: pip install <package>")
    else:
        print("\n  ✅ All dependencies satisfied!")
    
    print()
    return 0


# =============================================================================
# PREFLIGHT
# =============================================================================

def preflight(args) -> int:
    """Pre-flight checks before running optimizations."""
    _print_header("Pre-flight Checks", "✈️")
    
    all_ok = True
    
    # Check 1: GPU available
    print("\n  1. GPU Availability")
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"     ✅ {count} GPU(s) available")
        else:
            print("     ❌ No CUDA GPUs available")
            all_ok = False
    except ImportError:
        print("     ❌ PyTorch not installed")
        all_ok = False
    
    # Check 2: Memory
    print("\n  2. GPU Memory")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split('\n')):
                free, total = map(int, line.split(','))
                pct = free / total * 100
                if pct > 80:
                    print(f"     ✅ GPU {i}: {free/1024:.1f}/{total/1024:.1f} GB free ({pct:.0f}%)")
                else:
                    print(f"     ⚠️ GPU {i}: {free/1024:.1f}/{total/1024:.1f} GB free ({pct:.0f}%) - may need to free memory")
    except Exception:
        pass
    
    # Check 3: LLM
    print("\n  3. LLM Backend")
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        print("     ✅ API key configured")
    else:
        print("     ⚠️ No API key - LLM features will be limited")
    
    # Check 4: Key libraries
    print("\n  4. Key Libraries")
    key_libs = ["flash_attn", "triton"]
    for lib in key_libs:
        try:
            __import__(lib)
            print(f"     ✅ {lib} installed")
        except ImportError:
            print(f"     ⚠️ {lib} not installed - some features unavailable")
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("  ✅ All pre-flight checks passed!")
    else:
        print("  ⚠️ Some checks failed - see above")
    print()
    
    return 0 if all_ok else 1

