"""
Profiling Commands - Deep profiling, flame graphs, memory analysis, NCU integration.

Provides commands for:
- Flame graph generation and visualization
- Memory timeline analysis
- Kernel breakdown and analysis
- HTA (Holistic Trace Analysis)
- NCU deep dive (warp divergence, bank conflicts, occupancy)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from tools.core.perf_core import get_core
from tools.profiling.flame_graph import FlameGraphGenerator


def _print_header(title: str, emoji: str = "🔍"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def _check_profiler_available() -> bool:
    """Check if PyTorch profiler is available."""
    try:
        import torch
        from torch.profiler import profile
        return True
    except ImportError:
        return False


def _check_ncu_available() -> bool:
    """Check if NVIDIA Nsight Compute (ncu) is available."""
    try:
        result = subprocess.run(['ncu', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


# =============================================================================
# FLAME GRAPH
# =============================================================================

def flame_graph(args) -> int:
    """Generate flame graph from profile data."""
    _print_header("Flame Graph", "🔥")
    
    profile_file = getattr(args, 'file', None)
    output = getattr(args, 'output', 'flame.json')
    core = get_core()

    try:
        if profile_file:
            trace_path = Path(profile_file)
            if not trace_path.exists():
                print(f"  ❌ Profile file not found: {profile_file}")
                return 1
            data = FlameGraphGenerator().from_chrome_trace(trace_path)
            print(f"  Parsed Chrome trace: {trace_path}")
        else:
            data = core.get_flame_graph_data()
            if not data:
                print("  ❌ No profile traces found in artifacts.")
                return 1
            print("  Using most recent trace discovered in artifacts.")

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  ✅ Flame graph JSON written to: {output}")
        return 0
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1


def _generate_flame_html(kernels: list) -> str:
    """Generate a simple HTML flame graph visualization."""
    total = sum(d['total_us'] for _, d in kernels)
    
    rows = []
    for name, data in kernels[:30]:
        pct = data['total_us'] / total * 100
        width = max(pct, 1)
        rows.append(f'''
        <div class="bar" style="width: {width}%;" title="{name}: {data['total_us']/1000:.2f}ms ({pct:.1f}%)">
            <span>{name[:40]}</span>
        </div>''')
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Flame Graph - aisp</title>
    <style>
        body {{ font-family: system-ui; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #ff6b6b; }}
        .bar {{ 
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            margin: 2px 0; padding: 8px; border-radius: 4px;
            min-width: 100px; white-space: nowrap; overflow: hidden;
        }}
        .bar span {{ font-size: 12px; }}
    </style>
</head>
<body>
    <h1>🔥 Kernel Flame Graph</h1>
    <p>Total time: {total/1000:.2f}ms across {len(kernels)} kernel types</p>
    {''.join(rows)}
</body>
</html>'''


# =============================================================================
# MEMORY TIMELINE
# =============================================================================

def memory_timeline(args) -> int:
    """Analyze memory allocation timeline."""
    _print_header("Memory Timeline", "💾")
    
    profile_file = getattr(args, 'file', None)
    output = getattr(args, 'output', None)
    core = get_core()

    try:
        if profile_file:
            profile_path = Path(profile_file)
            if not profile_path.exists():
                print(f"  ❌ Profile file not found: {profile_file}")
                return 1
            with open(profile_path) as f:
                data = json.load(f)
        else:
            data = core.get_memory_timeline()
            if not data:
                print("  ❌ No memory timeline data available.")
                return 1
            print("  Using most recent memory timeline from artifacts.")

        samples = data.get("timeline", data.get("samples", []))
        peak = data.get("peak_memory_mb") or data.get("peak_mb") or 0
        print(f"  Timeline points: {len(samples)}")
        print(f"  Peak memory:     {peak:.2f} MB")

        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n  ✅ Memory timeline written to: {output}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1

    print()
    return 0


# =============================================================================
# KERNEL BREAKDOWN
# =============================================================================

def kernel_breakdown(args) -> int:
    """Analyze kernel execution breakdown."""
    _print_header("Kernel Breakdown", "⚡")
    
    profile_file = getattr(args, 'file', None)
    
    if not profile_file:
        print("  Using latest kernel breakdown from artifacts...")
        try:
            data = get_core().get_kernel_breakdown()
            if not data:
                print("  ❌ No kernel breakdown available.")
                return 1
            summary = data.get("summary", {})
            total_ms = summary.get("total_kernel_time_ms", 0)
            print(f"  Total kernel time: {total_ms:.2f} ms")
            top = data.get("top_kernels", [])[:10]
            if top:
                print("\n  Top kernels:")
                for k in top:
                    print(f"    {k.get('name','unknown')[:50]:50} {k.get('duration_ms',0):8.2f} ms")
            return 0
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return 1
    
    profile_path = Path(profile_file)
    if not profile_path.exists():
        print(f"  ❌ Profile file not found: {profile_file}")
        return 1
    
    try:
        with open(profile_path) as f:
            trace_data = json.load(f)
        
        events = trace_data.get('traceEvents', [])
        
        # Categorize kernels
        categories = {
            'gemm': [],
            'attention': [],
            'elementwise': [],
            'memory': [],
            'communication': [],
            'other': [],
        }
        
        for event in events:
            if event.get('cat') != 'kernel':
                continue
            
            name = event.get('name', '').lower()
            dur = event.get('dur', 0)
            
            if 'gemm' in name or 'matmul' in name or 'mm_' in name:
                categories['gemm'].append((name, dur))
            elif 'attention' in name or 'flash' in name or 'sdpa' in name:
                categories['attention'].append((name, dur))
            elif 'nccl' in name or 'allreduce' in name:
                categories['communication'].append((name, dur))
            elif 'copy' in name or 'memcpy' in name or 'memset' in name:
                categories['memory'].append((name, dur))
            elif 'elementwise' in name or 'unary' in name or 'binary' in name:
                categories['elementwise'].append((name, dur))
            else:
                categories['other'].append((name, dur))
        
        # Print breakdown
        total_time = sum(dur for cat in categories.values() for _, dur in cat)
        
        print(f"  Total kernel time: {total_time/1000:.2f}ms\n")
        print("  Category Breakdown:")
        print("-" * 70)
        
        for cat, kernels in sorted(categories.items(), key=lambda x: sum(d for _, d in x[1]), reverse=True):
            if not kernels:
                continue
            cat_time = sum(dur for _, dur in kernels)
            pct = cat_time / total_time * 100 if total_time > 0 else 0
            print(f"\n  {cat.upper():15} {cat_time/1000:8.2f}ms ({pct:5.1f}%) - {len(kernels)} kernels")
            
            # Top 3 in category
            top = sorted(kernels, key=lambda x: x[1], reverse=True)[:3]
            for name, dur in top:
                print(f"    • {name[:45]:45} {dur/1000:6.2f}ms")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1
    
    print()
    return 0


# =============================================================================
# HTA ANALYSIS
# =============================================================================

def hta_analysis(args) -> int:
    """Holistic Trace Analysis (HTA) for comprehensive profiling."""
    _print_header("HTA Analysis", "📊")
    
    profile_file = getattr(args, 'file', None)
    
    print("  HTA provides holistic analysis of GPU traces including:")
    print("    • Idle time analysis")
    print("    • Kernel launch overhead")
    print("    • Memory bandwidth utilization")
    print("    • Communication/computation overlap")
    
    try:
        if profile_file:
            from hta.trace_analysis import TraceAnalysis

            print(f"\n  Analyzing: {profile_file}")
            analyzer = TraceAnalysis(trace_dir=profile_file)

            # Idle time analysis
            idle = analyzer.get_idle_time_breakdown()
            print("\n  Idle Time Breakdown:")
            print(f"    {idle}")
        else:
            data = get_core().get_hta_analysis()
            if not data:
                print("  ❌ No HTA analysis available.")
                return 1
            print("  Using latest HTA analysis from artifacts.")
            print(json.dumps(data, indent=2))
        
    except ImportError:
        print("\n  ⚠️ HTA not installed. Install with:")
        print("    pip install HolisticTraceAnalysis")
        print("\n  Or use manual analysis:")
        print("    aisp profile kernels <trace.json>")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1
    
    print()
    return 0


# =============================================================================
# NCU DEEP DIVE
# =============================================================================

def ncu_analysis(args) -> int:
    """NCU (Nsight Compute) deep dive analysis."""
    _print_header("NCU Deep Dive", "🔬")
    
    if not _check_ncu_available():
        print("  ❌ NCU (Nsight Compute) not found")
        print("\n  Install from: https://developer.nvidia.com/nsight-compute")
        print("  Or load module: module load nsight-compute")
        return 1
    
    kernel = getattr(args, 'kernel', None)
    script = getattr(args, 'script', None)
    
    if not script:
        print("  Usage: aisp profile ncu <script.py> [--kernel <name>]")
        print("\n  Examples:")
        print("    aisp profile ncu train.py")
        print("    aisp profile ncu train.py --kernel ampere_fp16_gemm")
        return 1
    
    print(f"  Running NCU on: {script}")
    
    cmd = ['ncu', '--set', 'full', '-o', 'ncu_report']
    if kernel:
        cmd.extend(['--kernel-name', kernel])
    cmd.extend(['python', script])
    
    print(f"  Command: {' '.join(cmd)}")
    print("\n  This may take several minutes...")
    
    # Note: Actually running NCU would require proper setup
    print("\n  ⚠️ NCU profiling requires running as root or with proper permissions")
    print("  Generated report would include:")
    print("    • Warp divergence analysis")
    print("    • Bank conflict detection")
    print("    • Occupancy analysis")
    print("    • Memory throughput")
    print("    • Roofline position")
    
    return 0


def warp_divergence(args) -> int:
    """Analyze warp divergence in kernels."""
    _print_header("Warp Divergence Analysis", "🔀")
    
    print("  Warp divergence occurs when threads in a warp take different branches.")
    print("  This causes serialized execution and reduced efficiency.\n")
    
    print("  Common causes:")
    print("    • if/else statements with thread-dependent conditions")
    print("    • Variable loop iteration counts")
    print("    • Early return statements")
    
    print("\n  Detection methods:")
    print("    1. NCU metrics: smsp__branch_efficiency")
    print("    2. NSight Compute GUI: Warp State Statistics")
    print("    3. CUDA profiler: branch_efficiency event")
    
    print("\n  To analyze your code:")
    print("    aisp profile ncu <script.py>")
    
    # Get book citations
    try:
        from tools.core.book import get_citations, format_citations
        citations = get_citations("warp divergence branch efficiency", max_results=2)
        if citations:
            print(format_citations(citations))
    except:
        pass
    
    return 0


def bank_conflicts(args) -> int:
    """Analyze shared memory bank conflicts."""
    _print_header("Bank Conflict Analysis", "🏦")
    
    print("  Bank conflicts occur when multiple threads access the same shared memory bank.")
    print("  This serializes memory accesses and reduces throughput.\n")
    
    print("  Shared memory organization:")
    print("    • 32 banks (one per warp lane)")
    print("    • Successive 32-bit words map to successive banks")
    print("    • Stride-1 access = no conflicts")
    print("    • Stride-32 access = 32-way conflict (worst case)")
    
    print("\n  Detection methods:")
    print("    1. NCU metrics: l1tex__data_bank_conflicts_pipe_lsu")
    print("    2. NSight Compute: Shared Memory section")
    
    print("\n  Common fixes:")
    print("    • Add padding to avoid stride conflicts")
    print("    • Reorganize data layout")
    print("    • Use different access patterns")
    
    # Get book citations
    try:
        from tools.core.book import get_citations, format_citations
        citations = get_citations("bank conflicts shared memory", max_results=2)
        if citations:
            print(format_citations(citations))
    except:
        pass
    
    return 0


def occupancy_analysis(args) -> int:
    """Analyze GPU occupancy."""
    _print_header("Occupancy Analysis", "📈")
    
    print("  Occupancy = active warps / maximum warps per SM\n")
    
    print("  Factors affecting occupancy:")
    print("    • Registers per thread (CUDA register file is limited)")
    print("    • Shared memory per block")
    print("    • Block size (threads per block)")
    
    # Get current GPU info
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"\n  Your GPU: {props.name}")
            print(f"    Max threads/SM:     {props.max_threads_per_multi_processor}")
            print(f"    Max threads/block:  {props.max_threads_per_block}")
            print(f"    Registers/SM:       {props.regs_per_multiprocessor}")
            print(f"    Shared memory/SM:   {props.max_shared_memory_per_multiprocessor / 1024:.0f} KB")
            print(f"    Warp size:          {props.warp_size}")
    except:
        pass
    
    print("\n  Use NCU for kernel-specific occupancy:")
    print("    aisp profile ncu <script.py>")
    print("\n  Key NCU metrics:")
    print("    • sm__warps_active.avg.pct_of_peak_sustained")
    print("    • launch__occupancy_limit_registers")
    print("    • launch__occupancy_limit_shared_mem")
    
    # Get book citations
    try:
        from tools.core.book import get_citations, format_citations
        citations = get_citations("occupancy warps SM utilization", max_results=2)
        if citations:
            print(format_citations(citations))
    except:
        pass
    
    return 0


# =============================================================================
# COMMAND REGISTRATION
# =============================================================================

def register_commands(subparsers):
    """Register profiling commands."""
    profile_parser = subparsers.add_parser("profile", help="Deep profiling and analysis")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command")
    
    # Flame graph
    flame_p = profile_subparsers.add_parser("flame", help="Generate flame graph")
    flame_p.add_argument("file", nargs="?", help="Profile JSON file")
    flame_p.add_argument("-o", "--output", default="flame.html", help="Output HTML file")
    flame_p.set_defaults(func=flame_graph)
    
    # Memory timeline
    mem_p = profile_subparsers.add_parser("memory", help="Memory timeline analysis")
    mem_p.add_argument("file", nargs="?", help="Profile file (optional, live if omitted)")
    mem_p.set_defaults(func=memory_timeline)
    
    # Kernel breakdown
    kern_p = profile_subparsers.add_parser("kernels", help="Kernel breakdown analysis")
    kern_p.add_argument("file", nargs="?", help="Profile JSON file")
    kern_p.set_defaults(func=kernel_breakdown)
    
    # HTA
    hta_p = profile_subparsers.add_parser("hta", help="Holistic Trace Analysis")
    hta_p.add_argument("file", nargs="?", help="Trace directory")
    hta_p.set_defaults(func=hta_analysis)
    
    # NCU
    ncu_p = profile_subparsers.add_parser("ncu", help="NCU deep dive")
    ncu_p.add_argument("script", nargs="?", help="Python script to profile")
    ncu_p.add_argument("--kernel", help="Specific kernel to analyze")
    ncu_p.set_defaults(func=ncu_analysis)
    
    # Warp divergence
    warp_p = profile_subparsers.add_parser("warp-divergence", help="Warp divergence analysis")
    warp_p.set_defaults(func=warp_divergence)
    
    # Bank conflicts
    bank_p = profile_subparsers.add_parser("bank-conflicts", help="Bank conflict analysis")
    bank_p.set_defaults(func=bank_conflicts)
    
    # Occupancy
    occ_p = profile_subparsers.add_parser("occupancy", help="Occupancy analysis")
    occ_p.set_defaults(func=occupancy_analysis)

