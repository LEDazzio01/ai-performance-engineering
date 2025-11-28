"""
Monitoring Commands - Live monitoring, regression detection, metrics.
"""

from __future__ import annotations

import time

from tools.core.perf_core import get_core


def _print_header(title: str, emoji: str = "📊"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def live_monitor(args) -> int:
    """Real-time GPU monitoring."""
    _print_header("Live GPU Monitor", "📈")
    
    print("  Press Ctrl+C to stop\n")
    
    try:
        while True:
            info = get_core().get_gpu_info() or {}

            # Clear line and print
            print("\033[2J\033[H")  # Clear screen
            print("  GPU Monitor (Ctrl+C to stop)")
            print("  " + "=" * 60)

            gpus = info.get("gpus") or [info] if info else []
            for i, gpu in enumerate(gpus):
                if not gpu:
                    continue
                name = gpu.get("name", f"GPU {i}")
                temp = gpu.get("temperature", gpu.get("temperature_gpu_c", "N/A"))
                power = gpu.get("power", gpu.get("power_draw_w", "N/A"))
                util = gpu.get("utilization", gpu.get("utilization_gpu_pct", 0)) or 0
                mem_used = gpu.get("memory_used") or gpu.get("memory_used_mb") or 0
                mem_total = gpu.get("memory_total") or gpu.get("memory_total_mb") or 1
                mem_pct = (mem_used / mem_total) * 100 if mem_total else 0

                util_color = "\033[32m" if util < 50 else "\033[33m" if util < 80 else "\033[31m"
                reset = "\033[0m"

                print(f"\n  GPU {i}: {name}")
                print(f"    Temperature: {temp}°C")
                print(f"    Power:       {power}W")
                print(f"    Utilization: {util_color}{util}%{reset}")
                print(f"    Memory:      {mem_used/1024:.1f}/{mem_total/1024:.1f} GB ({mem_pct:.0f}%)")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n  Monitoring stopped.")
    except Exception as e:
        print(f"  Error: {e}")
    
    return 0


def regression(args) -> int:
    """Detect performance regressions."""
    _print_header("Regression Detection", "⚠️")
    
    print("  Checking for performance regressions using latest benchmark artifacts...")

    from pathlib import Path
    from tools.analysis.detect_regressions import RegressionDetector
    detector = RegressionDetector()

    artifacts = sorted(Path("artifacts").rglob("benchmark_test_results.json"), key=lambda p: p.stat().st_mtime)
    if Path("benchmark_test_results.json").exists():
        artifacts.append(Path("benchmark_test_results.json"))
    artifacts = sorted(set(artifacts), key=lambda p: p.stat().st_mtime)

    if len(artifacts) < 2:
        print("  ❌ Need at least two benchmark result files to compare.")
        return 1

    baseline = artifacts[-2]
    current = artifacts[-1]
    print(f"  Baseline: {baseline}")
    print(f"  Current:  {current}")

    import json
    with baseline.open() as f:
        baseline_data = json.load(f)
    with current.open() as f:
        current_data = json.load(f)

    regressions = []
    for b_bench, c_bench in zip(baseline_data.get("results", []), current_data.get("results", [])):
        for b_entry, c_entry in zip(b_bench.get("benchmarks", []), c_bench.get("benchmarks", [])):
            b_metrics = detector.extract_metrics(b_entry)
            c_metrics = detector.extract_metrics(c_entry)
            regressions.extend(detector.compare_metrics(c_metrics, b_metrics))

    if not regressions:
        print("  ✅ No regressions detected based on available metrics.")
        return 0

    print(f"  ⚠️ Detected {len(regressions)} regression(s):")
    for reg in regressions:
        print(f"    • {reg.get('metric')}: Δ{reg.get('delta_pct'):.2f}% (threshold {reg.get('threshold_pct')}%)")
    return 0


def metrics(args) -> int:
    """Collect and display metrics."""
    _print_header("Performance Metrics", "📏")
    
    print("\n  Current GPU Metrics:")

    try:
        info = get_core().get_gpu_info() or {}
        gpus = info.get("gpus") or [info] if info else []
        for i, gpu in enumerate(gpus):
            print(f"\n  GPU {i}: {gpu.get('name','')}")
            print(f"    Power Draw:      {gpu.get('power', gpu.get('power_draw_w','N/A'))} W")
            print(f"    Clocks SM/MEM:   {gpu.get('clock_graphics','?')} / {gpu.get('clock_memory','?')} MHz")
            print(f"    Memory Used:     {gpu.get('memory_used',0)/1024:.1f} / {gpu.get('memory_total',1)/1024:.1f} GB")
            print(f"    GPU Utilization: {gpu.get('utilization', gpu.get('utilization_gpu_pct',0))}%")
            print(f"    Mem Utilization: {gpu.get('utilization_memory',0):.0f}%")
    except Exception as e:
        print(f"  Error: {e}")
    
    return 0
