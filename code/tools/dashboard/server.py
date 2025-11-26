#!/usr/bin/env python3
"""
GPU Performance Lab Dashboard Server

A sleek web dashboard for viewing benchmark results, LLM analysis,
optimization insights, deep profiling comparisons, and live optimization streaming.

Features:
- Benchmark results visualization
- LLM-generated insights
- nsys/ncu profile comparison with recommendations
- Live optimization console with SSE streaming

Usage:
    python -m tools.dashboard.server [--port 8080] [--data results.json]
"""

import argparse
import http.server
import json
import os
import queue
import re
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import threading
import time
import uuid


# Find the code root (3 levels up from this file)
CODE_ROOT = Path(__file__).parent.parent.parent

# Add tools to path for imports
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Global optimization job store for SSE streaming
_optimization_jobs: Dict[str, Dict[str, Any]] = {}
_job_events: Dict[str, queue.Queue] = {}


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves the dashboard and API endpoints."""
    
    def __init__(self, *args, data_file: Optional[Path] = None, **kwargs):
        self.data_file = data_file
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/api/data':
            self.send_json_response(self.load_benchmark_data())
        elif self.path == '/api/gpu':
            self.send_json_response(self.get_gpu_info())
        elif self.path == '/api/llm-analysis':
            self.send_json_response(self.load_llm_analysis())
        elif self.path == '/api/profiles':
            self.send_json_response(self.load_profile_data())
        elif self.path == '/api/available':
            self.send_json_response(self.get_available_benchmarks())
        elif self.path == '/api/scan-all':
            self.send_json_response(self.scan_all_chapters_and_labs())
        # CSV Export endpoints
        elif self.path == '/api/export/csv':
            self.send_csv_response(self.export_benchmarks_csv())
        elif self.path == '/api/export/csv/detailed':
            self.send_csv_response(self.export_detailed_csv())
        # PDF Export endpoints
        elif self.path == '/api/export/pdf':
            self.export_pdf_report()
        elif self.path == '/api/export/html':
            self.export_html_report()
        # Profiler visualization endpoints
        elif self.path == '/api/profiler/flame':
            self.send_json_response(self.get_flame_graph_data())
        elif self.path == '/api/profiler/memory':
            self.send_json_response(self.get_memory_timeline())
        elif self.path == '/api/profiler/timeline':
            self.send_json_response(self.get_cpu_gpu_timeline())
        elif self.path == '/api/profiler/kernels':
            self.send_json_response(self.get_kernel_breakdown())
        elif self.path == '/api/profiler/hta':
            self.send_json_response(self.get_hta_analysis())
        elif self.path == '/api/profiler/compile':
            self.send_json_response(self.get_compile_analysis())
        # NEW: Deep profile comparison endpoints
        elif self.path == '/api/deep-profile/list':
            self.send_json_response(self.list_deep_profile_pairs())
        elif self.path.startswith('/api/deep-profile/compare/'):
            chapter = self.path.split('/api/deep-profile/compare/')[1]
            self.send_json_response(self.compare_profiles(chapter))
        elif self.path == '/api/deep-profile/recommendations':
            self.send_json_response(self.get_profile_recommendations())
        # NEW: Live optimization SSE streaming
        elif self.path.startswith('/api/optimize/stream/'):
            job_id = self.path.split('/api/optimize/stream/')[1]
            self.stream_optimization_events(job_id)
        elif self.path == '/api/optimize/jobs':
            self.send_json_response(self.list_optimization_jobs())
        # NEW: Multi-metric analysis endpoints
        elif self.path == '/api/analysis/pareto':
            self.send_json_response(self.get_pareto_frontier())
        elif self.path == '/api/analysis/tradeoffs':
            self.send_json_response(self.get_tradeoff_analysis())
        elif self.path == '/api/analysis/recommendations':
            self.send_json_response(self.get_constraint_recommendations())
        elif self.path == '/api/analysis/leaderboards':
            self.send_json_response(self.get_categorized_leaderboards())
        elif self.path.startswith('/api/'):
            self.send_json_response({"error": "Unknown API endpoint"})
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests for starting optimizations."""
        if self.path == '/api/optimize/start':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.start_optimization_job(params)
            self.send_json_response(result)
        elif self.path == '/api/optimize/stop':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.stop_optimization_job(params.get('job_id'))
            self.send_json_response(result)
        else:
            self.send_json_response({"error": "Unknown POST endpoint"})
    
    def send_json_response(self, data: dict):
        """Send a JSON response."""
        response = json.dumps(data, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def send_csv_response(self, csv_data: str, filename: str = "benchmark_results.csv"):
        """Send a CSV response for download."""
        response = csv_data.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv')
        self.send_header('Content-Length', len(response))
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def export_benchmarks_csv(self) -> str:
        """Export benchmark results to CSV format."""
        data = self.load_benchmark_data()
        benchmarks = data.get('benchmarks', [])
        
        lines = [
            "Benchmark,Baseline Time (ms),Optimized Time (ms),Speedup,Chapter/Lab,Status"
        ]
        
        for b in benchmarks:
            name = b.get('name', 'Unknown')
            baseline = b.get('baseline_time_ms', 0)
            optimized = b.get('optimized_time_ms', 0)
            speedup = b.get('speedup', 1.0)
            chapter = b.get('chapter', '')
            status = "✓ Optimized" if speedup > 1.1 else "⚠ Needs Work"
            
            lines.append(f'"{name}",{baseline:.4f},{optimized:.4f},{speedup:.2f},"{chapter}","{status}"')
        
        return '\n'.join(lines)
    
    def export_detailed_csv(self) -> str:
        """Export detailed benchmark results including all metrics."""
        data = self.load_benchmark_data()
        benchmarks = data.get('benchmarks', [])
        
        lines = [
            "Benchmark,Chapter,Baseline Time (ms),Optimized Time (ms),Speedup,"
            "Baseline Memory (MB),Optimized Memory (MB),Memory Reduction (%),"
            "Techniques Applied,LLM Patches,Patch Success Rate"
        ]
        
        for b in benchmarks:
            name = b.get('name', 'Unknown')
            chapter = b.get('chapter', '')
            baseline = b.get('baseline_time_ms', 0)
            optimized = b.get('optimized_time_ms', 0)
            speedup = b.get('speedup', 1.0)
            baseline_mem = b.get('baseline_memory_mb', 0)
            optimized_mem = b.get('optimized_memory_mb', 0)
            mem_reduction = ((baseline_mem - optimized_mem) / baseline_mem * 100) if baseline_mem > 0 else 0
            techniques = b.get('techniques', [])
            techniques_str = '; '.join(techniques) if techniques else ''
            llm_patches = b.get('llm_patches_applied', 0)
            patch_success = b.get('patch_success_rate', 0)
            
            lines.append(
                f'"{name}","{chapter}",{baseline:.4f},{optimized:.4f},{speedup:.2f},'
                f'{baseline_mem:.2f},{optimized_mem:.2f},{mem_reduction:.1f},'
                f'"{techniques_str}",{llm_patches},{patch_success:.0f}'
            )
        
        return '\n'.join(lines)
    
    def get_flame_graph_data(self) -> dict:
        """Get flame graph data from profile traces."""
        flame_data = {
            "name": "GPU Execution",
            "value": 0,
            "children": []
        }
        
        # Try to find profile traces
        profile_dirs = [
            CODE_ROOT / "benchmark_profiles",
            CODE_ROOT / "artifacts" / "profiles",
            CODE_ROOT / "profiles",
        ]
        
        trace_files = []
        for profile_dir in profile_dirs:
            if profile_dir.exists():
                trace_files.extend(profile_dir.glob("**/*.json"))
        
        # Parse the most recent trace
        if trace_files:
            trace_files = sorted(trace_files, key=lambda f: f.stat().st_mtime, reverse=True)
            try:
                with open(trace_files[0]) as f:
                    trace = json.load(f)
                
                events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
                
                # Build flame graph structure
                kernel_times = {}
                for event in events:
                    if event.get('ph') == 'X' and event.get('dur', 0) > 10:
                        name = event.get('name', 'unknown')
                        cat = event.get('cat', 'other')
                        dur = event.get('dur', 0)
                        
                        # Group by category
                        if cat not in kernel_times:
                            kernel_times[cat] = {}
                        if name not in kernel_times[cat]:
                            kernel_times[cat][name] = 0
                        kernel_times[cat][name] += dur
                
                for cat, kernels in kernel_times.items():
                    cat_node = {
                        "name": cat,
                        "value": sum(kernels.values()),
                        "children": [
                            {"name": k, "value": v, "children": []}
                            for k, v in sorted(kernels.items(), key=lambda x: -x[1])[:20]
                        ]
                    }
                    flame_data["children"].append(cat_node)
                
                flame_data["value"] = sum(c["value"] for c in flame_data["children"])
                flame_data["trace_file"] = str(trace_files[0].name)
                
            except Exception as e:
                flame_data["error"] = str(e)
        else:
            flame_data["message"] = "No profile traces found. Run benchmarks with profiling enabled."
        
        return flame_data
    
    def get_memory_timeline(self) -> dict:
        """Get memory usage timeline data."""
        memory_data = {
            "timeline": [],
            "peak_mb": 0,
            "summary": {
                "total_allocated_mb": 0,
                "peak_allocated_mb": 0,
                "num_allocations": 0,
            }
        }
        
        # Try to find memory snapshot files
        profile_dirs = [
            CODE_ROOT / "benchmark_profiles",
            CODE_ROOT / "artifacts" / "profiles",
        ]
        
        memory_files = []
        for profile_dir in profile_dirs:
            if profile_dir.exists():
                memory_files.extend(profile_dir.glob("**/*memory*.json"))
                memory_files.extend(profile_dir.glob("**/*memory*.pickle"))
        
        if memory_files:
            memory_files = sorted(memory_files, key=lambda f: f.stat().st_mtime, reverse=True)
            try:
                if memory_files[0].suffix == '.json':
                    with open(memory_files[0]) as f:
                        data = json.load(f)
                    memory_data.update(data)
            except Exception as e:
                memory_data["error"] = str(e)
        else:
            # Generate sample timeline from benchmark data
            benchmarks = self.load_benchmark_data().get('benchmarks', [])
            for i, b in enumerate(benchmarks[:20]):
                memory_data["timeline"].append({
                    "time_ms": i * 100,
                    "allocated_mb": b.get('optimized_memory_mb') or (50 + i * 10),
                    "peak_mb": b.get('baseline_memory_mb') or (100 + i * 5),
                })
            memory_data["peak_mb"] = max(
                (t.get('peak_mb') or 0 for t in memory_data["timeline"]), default=0
            )
        
        return memory_data
    
    def get_cpu_gpu_timeline(self) -> dict:
        """Get CPU/GPU parallel timeline data."""
        timeline_data = {
            "cpu": [],
            "gpu": [],
            "streams": {},
            "summary": {
                "total_time_ms": 0,
                "cpu_time_ms": 0,
                "gpu_time_ms": 0,
                "overlap_ms": 0,
            }
        }
        
        # Try to find Chrome traces
        profile_dirs = [
            CODE_ROOT / "benchmark_profiles",
            CODE_ROOT / "artifacts" / "profiles",
        ]
        
        trace_files = []
        for profile_dir in profile_dirs:
            if profile_dir.exists():
                trace_files.extend(profile_dir.glob("**/*.json"))
        
        if trace_files:
            trace_files = sorted(trace_files, key=lambda f: f.stat().st_mtime, reverse=True)
            try:
                with open(trace_files[0]) as f:
                    trace = json.load(f)
                
                events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
                
                min_ts = float('inf')
                max_ts = 0
                
                for event in events[:500]:  # Limit for performance
                    if event.get('ph') != 'X':
                        continue
                    
                    ts = event.get('ts', 0)
                    dur = event.get('dur', 0)
                    name = event.get('name', '')
                    cat = event.get('cat', '').lower()
                    
                    min_ts = min(min_ts, ts)
                    max_ts = max(max_ts, ts + dur)
                    
                    event_data = {
                        "name": name[:50],
                        "start_ms": ts / 1000,
                        "duration_ms": dur / 1000,
                    }
                    
                    if 'cuda' in cat or 'kernel' in cat:
                        timeline_data["gpu"].append(event_data)
                    else:
                        timeline_data["cpu"].append(event_data)
                
                if max_ts > min_ts:
                    timeline_data["summary"]["total_time_ms"] = (max_ts - min_ts) / 1000
                
            except Exception as e:
                timeline_data["error"] = str(e)
        
        return timeline_data
    
    def get_kernel_breakdown(self) -> dict:
        """Get detailed kernel timing breakdown."""
        kernel_data = {
            "kernels": [],
            "summary": {
                "total_kernels": 0,
                "total_time_us": 0,
                "avg_kernel_time_us": 0,
            },
            "by_type": {}
        }
        
        # Parse from flame graph data
        flame = self.get_flame_graph_data()
        
        all_kernels = []
        for category in flame.get("children", []):
            cat_name = category.get("name", "other")
            for kernel in category.get("children", []):
                all_kernels.append({
                    "name": kernel["name"],
                    "time_us": kernel["value"],
                    "category": cat_name,
                })
                
                if cat_name not in kernel_data["by_type"]:
                    kernel_data["by_type"][cat_name] = 0
                kernel_data["by_type"][cat_name] += kernel["value"]
        
        # Sort by time
        kernel_data["kernels"] = sorted(all_kernels, key=lambda k: -k["time_us"])[:50]
        kernel_data["summary"]["total_kernels"] = len(all_kernels)
        kernel_data["summary"]["total_time_us"] = sum(k["time_us"] for k in all_kernels)
        if all_kernels:
            kernel_data["summary"]["avg_kernel_time_us"] = (
                kernel_data["summary"]["total_time_us"] / len(all_kernels)
            )
        
        return kernel_data
    
    def get_hta_analysis(self) -> dict:
        """Get HTA (Holistic Trace Analysis) results."""
        hta_data = {
            "temporal_breakdown": {
                "compute_pct": 70,
                "idle_pct": 15,
                "communication_pct": 10,
                "memory_pct": 5,
            },
            "top_kernels": [],
            "recommendations": [],
            "bottlenecks": [],
        }
        
        # Try to find HTA reports
        hta_files = list(CODE_ROOT.glob("**/hta_report*.json"))
        hta_files.extend(CODE_ROOT.glob("**/hta_analysis*.json"))
        
        if hta_files:
            try:
                with open(sorted(hta_files, key=lambda f: f.stat().st_mtime)[-1]) as f:
                    data = json.load(f)
                hta_data.update(data)
            except Exception:
                pass
        else:
            # Generate from kernel data
            kernel_data = self.get_kernel_breakdown()
            
            total_time = kernel_data["summary"]["total_time_us"]
            if total_time > 0:
                for kernel in kernel_data["kernels"][:10]:
                    hta_data["top_kernels"].append({
                        "name": kernel["name"],
                        "time_us": kernel["time_us"],
                        "pct": kernel["time_us"] / total_time * 100,
                    })
            
            # Generate recommendations based on data
            if kernel_data["by_type"]:
                top_type = max(kernel_data["by_type"].items(), key=lambda x: x[1])
                hta_data["recommendations"].append(
                    f"Optimize {top_type[0]} operations ({top_type[1]/1000:.1f}ms total)"
                )
        
        return hta_data
    
    def get_compile_analysis(self) -> dict:
        """Get torch.compile analysis results."""
        compile_data = {
            "speedup": 0,
            "compile_time_ms": 0,
            "graph_breaks": 0,
            "fusion_ratio": 0,
            "recommendations": [],
            "mode_comparison": {},
        }
        
        # Try to find compile analysis files
        compile_files = list(CODE_ROOT.glob("**/compile_report*.json"))
        compile_files.extend(CODE_ROOT.glob("**/torch_compile*.json"))
        
        if compile_files:
            try:
                with open(sorted(compile_files, key=lambda f: f.stat().st_mtime)[-1]) as f:
                    data = json.load(f)
                compile_data.update(data)
            except Exception:
                pass
        else:
            # Estimate from benchmark data
            benchmarks = self.load_benchmark_data().get('benchmarks', [])
            
            # Find benchmarks that mention torch.compile
            compiled_benchmarks = [
                b for b in benchmarks 
                if 'compile' in b.get('name', '').lower() or 
                   'compile' in str(b.get('techniques', [])).lower()
            ]
            
            if compiled_benchmarks:
                avg_speedup = sum(b.get('speedup', 1) for b in compiled_benchmarks) / len(compiled_benchmarks)
                compile_data["speedup"] = avg_speedup
                compile_data["recommendations"].append(
                    f"torch.compile shows {avg_speedup:.2f}x average speedup across {len(compiled_benchmarks)} benchmarks"
                )
        
        return compile_data
    
    def get_available_benchmarks(self) -> dict:
        """Scan all chapters and labs for available benchmarks."""
        available = {
            "chapters": [],
            "labs": [],
            "total_chapters": 0,
            "total_labs": 0,
            "total_benchmarks": 0,
        }
        
        # Scan chapters (ch1, ch2, ... ch20)
        for ch_dir in sorted(CODE_ROOT.glob("ch[0-9]*")):
            if ch_dir.is_dir():
                chapter_info = self._scan_directory(ch_dir, "chapter")
                if chapter_info['benchmarks']:
                    available['chapters'].append(chapter_info)
        
        # Scan labs
        labs_dir = CODE_ROOT / 'labs'
        if labs_dir.exists():
            for lab_dir in sorted(labs_dir.iterdir()):
                if lab_dir.is_dir() and not lab_dir.name.startswith('.'):
                    lab_info = self._scan_directory(lab_dir, "lab")
                    if lab_info['benchmarks']:
                        available['labs'].append(lab_info)
        
        available['total_chapters'] = len(available['chapters'])
        available['total_labs'] = len(available['labs'])
        available['total_benchmarks'] = sum(
            len(ch['benchmarks']) for ch in available['chapters']
        ) + sum(
            len(lab['benchmarks']) for lab in available['labs']
        )
        
        return available
    
    def _scan_directory(self, directory: Path, dir_type: str) -> dict:
        """Scan a directory for baseline/optimized file pairs."""
        info = {
            "name": directory.name,
            "path": str(directory.relative_to(CODE_ROOT)),
            "type": dir_type,
            "benchmarks": [],
            "has_expectations": False,
            "has_profiles": False,
        }
        
        # Find all baseline files
        baseline_files = list(directory.glob("baseline_*.py")) + list(directory.glob("baseline_*.cu"))
        
        for baseline in baseline_files:
            # Extract benchmark name
            name = baseline.stem.replace("baseline_", "")
            file_type = "python" if baseline.suffix == ".py" else "cuda"
            
            # Find corresponding optimized files
            optimized_files = list(directory.glob(f"optimized_{name}*.py")) + \
                              list(directory.glob(f"optimized_{name}*.cu"))
            
            benchmark_info = {
                "name": name,
                "type": file_type,
                "baseline_file": baseline.name,
                "optimized_files": [f.name for f in optimized_files],
                "optimization_count": len(optimized_files),
            }
            
            info['benchmarks'].append(benchmark_info)
        
        # Check for expectations file
        info['has_expectations'] = (
            (directory / 'expectations_b200.json').exists() or
            (directory / 'expectations_gb10.json').exists()
        )
        
        # Check for profile data
        profile_dir = CODE_ROOT / 'benchmark_profiles' / directory.name
        info['has_profiles'] = profile_dir.exists() and any(profile_dir.iterdir()) if profile_dir.exists() else False
        
        return info
    
    def scan_all_chapters_and_labs(self) -> dict:
        """Comprehensive scan of all chapters and labs with detailed info."""
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan_results": [],
            "summary": {
                "total_directories": 0,
                "total_benchmarks": 0,
                "with_expectations": 0,
                "with_profiles": 0,
                "with_llm_analysis": 0,
                "python_benchmarks": 0,
                "cuda_benchmarks": 0,
            }
        }
        
        # Scan all chapters
        for ch_dir in sorted(CODE_ROOT.glob("ch[0-9]*")):
            if ch_dir.is_dir():
                scan = self._detailed_scan(ch_dir, "chapter")
                if scan:
                    result['scan_results'].append(scan)
                    result['summary']['total_directories'] += 1
                    result['summary']['total_benchmarks'] += scan['benchmark_count']
                    result['summary']['python_benchmarks'] += scan['python_count']
                    result['summary']['cuda_benchmarks'] += scan['cuda_count']
                    if scan['has_expectations']:
                        result['summary']['with_expectations'] += 1
                    if scan['has_profiles']:
                        result['summary']['with_profiles'] += 1
                    if scan['llm_analysis_count'] > 0:
                        result['summary']['with_llm_analysis'] += 1
        
        # Scan all labs
        labs_dir = CODE_ROOT / 'labs'
        if labs_dir.exists():
            for lab_dir in sorted(labs_dir.iterdir()):
                if lab_dir.is_dir() and not lab_dir.name.startswith('.'):
                    scan = self._detailed_scan(lab_dir, "lab")
                    if scan:
                        result['scan_results'].append(scan)
                        result['summary']['total_directories'] += 1
                        result['summary']['total_benchmarks'] += scan['benchmark_count']
                        result['summary']['python_benchmarks'] += scan['python_count']
                        result['summary']['cuda_benchmarks'] += scan['cuda_count']
                        if scan['has_expectations']:
                            result['summary']['with_expectations'] += 1
                        if scan['has_profiles']:
                            result['summary']['with_profiles'] += 1
                        if scan['llm_analysis_count'] > 0:
                            result['summary']['with_llm_analysis'] += 1
        
        return result
    
    def _detailed_scan(self, directory: Path, dir_type: str) -> Optional[dict]:
        """Detailed scan of a single directory."""
        baseline_py = list(directory.glob("baseline_*.py"))
        baseline_cu = list(directory.glob("baseline_*.cu"))
        
        if not baseline_py and not baseline_cu:
            return None
        
        # Count optimized files
        optimized_py = list(directory.glob("optimized_*.py"))
        optimized_cu = list(directory.glob("optimized_*.cu"))
        
        # Check for expectations
        has_expectations = (
            (directory / 'expectations_b200.json').exists() or
            (directory / 'expectations_gb10.json').exists()
        )
        
        # Check for profiles
        profile_dir = CODE_ROOT / 'benchmark_profiles' / directory.name
        has_profiles = profile_dir.exists() and any(profile_dir.iterdir()) if profile_dir.exists() else False
        
        # Count LLM analysis files
        llm_analysis_count = 0
        if profile_dir.exists():
            llm_analysis_count = len(list(profile_dir.glob("llm_analysis_*.md")))
        
        # Also check for LLM explanation files in the directory itself
        llm_analysis_count += len(list(directory.glob("*_llm_explanation.md")))
        
        # Check for results in benchmark_test_results.json
        has_results = self._check_has_results(directory.name)
        
        benchmarks = []
        for baseline in baseline_py + baseline_cu:
            name = baseline.stem.replace("baseline_", "")
            file_type = "python" if baseline.suffix == ".py" else "cuda"
            
            # Find optimized variants
            optimized = [f.name for f in directory.glob(f"optimized_{name}*.py")] + \
                        [f.name for f in directory.glob(f"optimized_{name}*.cu")]
            
            benchmarks.append({
                "name": name,
                "type": file_type,
                "baseline": baseline.name,
                "optimized": optimized,
                "optimization_count": len(optimized),
            })
        
        return {
            "name": directory.name,
            "path": str(directory.relative_to(CODE_ROOT)),
            "type": dir_type,
            "benchmark_count": len(baseline_py) + len(baseline_cu),
            "python_count": len(baseline_py),
            "cuda_count": len(baseline_cu),
            "optimized_count": len(optimized_py) + len(optimized_cu),
            "has_expectations": has_expectations,
            "has_profiles": has_profiles,
            "has_results": has_results,
            "llm_analysis_count": llm_analysis_count,
            "benchmarks": benchmarks,
        }
    
    def _check_has_results(self, directory_name: str) -> bool:
        """Check if directory has results in benchmark_test_results.json."""
        results_file = CODE_ROOT / 'benchmark_test_results.json'
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    for result in data.get('results', []):
                        if result.get('chapter') == directory_name:
                            return True
                        # Also check labs path
                        if f"labs/{directory_name}" in str(result.get('chapter', '')):
                            return True
            except:
                pass
        return False
    
    def load_benchmark_data(self) -> dict:
        """Load and aggregate benchmark data from ALL result files."""
        all_benchmarks = {}  # Key: (chapter, name) -> best result
        all_results = []
        latest_timestamp = None
        
        # 1. Scan ALL artifacts folders for results
        artifacts_dir = CODE_ROOT / 'artifacts'
        if artifacts_dir.exists():
            for result_file in sorted(artifacts_dir.rglob('benchmark_test_results.json')):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                        timestamp = data.get('timestamp', '')
                        if not latest_timestamp or timestamp > latest_timestamp:
                            latest_timestamp = timestamp
                        
                        for chapter_result in data.get('results', []):
                            chapter = chapter_result.get('chapter', 'unknown')
                            for bench in chapter_result.get('benchmarks', []):
                                name = bench.get('example', 'unknown')
                                key = (chapter, name)
                                speedup = bench.get('best_speedup', 0)
                                
                                # Keep best result for each benchmark
                                if key not in all_benchmarks or speedup > all_benchmarks[key].get('best_speedup', 0):
                                    all_benchmarks[key] = bench
                                    all_benchmarks[key]['_chapter'] = chapter
                except Exception as e:
                    pass  # Skip invalid files
        
        # 2. Also check root-level file - prefer this over artifacts (most recent run)
        default_path = CODE_ROOT / 'benchmark_test_results.json'
        if default_path.exists():
            try:
                with open(default_path) as f:
                    data = json.load(f)
                    for chapter_result in data.get('results', []):
                        chapter = chapter_result.get('chapter', 'unknown')
                        for bench in chapter_result.get('benchmarks', []):
                            name = bench.get('example', 'unknown')
                            key = (chapter, name)
                            speedup = bench.get('best_speedup', 0)
                            # Use >= to prefer root file (most recent) when speedups are equal
                            if key not in all_benchmarks or speedup >= all_benchmarks[key].get('best_speedup', 0):
                                all_benchmarks[key] = bench
                                all_benchmarks[key]['_chapter'] = chapter
            except:
                pass
        
        # 3. Transform aggregated results
        if not all_benchmarks:
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "benchmarks": [],
                "summary": {"total_benchmarks": 0, "avg_speedup": 0, "max_speedup": 0}
            }
        
        return self._transform_aggregated_data(all_benchmarks, latest_timestamp)
    
    def _transform_aggregated_data(self, all_benchmarks: dict, timestamp: str) -> dict:
        """Transform aggregated benchmark data to dashboard format."""
        benchmarks = []
        all_speedups = []
        successful = 0
        failed = 0
        failed_regression = 0
        
        for (chapter, name), bench in all_benchmarks.items():
            baseline_time = bench.get('baseline_time_ms') or 0
            best_speedup = bench.get('best_speedup') or 1.0
            status = bench.get('status', 'unknown')
            bench_type = bench.get('type', 'python')
            
            # Count status
            if status == 'succeeded':
                successful += 1
            elif 'regression' in str(status):
                failed_regression += 1
            elif 'failed' in str(status):
                failed += 1
            
            optimized_time = baseline_time / best_speedup if best_speedup and best_speedup > 0 else baseline_time
            gpu_metrics = bench.get('baseline_gpu_metrics', {})
            
            optimizations = []
            for opt in bench.get('optimizations', []):
                optimizations.append({
                    'technique': opt.get('technique', ''),
                    'speedup': opt.get('speedup', 1.0),
                    'time_ms': opt.get('time_ms', 0),
                    'file': opt.get('file', '')
                })
            
            # Extract memory and optimization goal
            baseline_memory_mb = bench.get('baseline_memory_mb')
            best_memory_savings_pct = bench.get('best_memory_savings_pct', 0)
            optimization_goal = bench.get('optimization_goal', 'speed')
            
            # Get best optimized memory from raw benchmark optimizations
            optimized_memory_mb = None
            for opt in bench.get('optimizations', []):
                if opt.get('memory_mb'):
                    if optimized_memory_mb is None or opt.get('memory_mb') < optimized_memory_mb:
                        optimized_memory_mb = opt.get('memory_mb')
            
            # If no optimization memory found but we have savings, compute it
            if optimized_memory_mb is None and baseline_memory_mb and best_memory_savings_pct:
                optimized_memory_mb = baseline_memory_mb * (1 - best_memory_savings_pct / 100)
            
            benchmarks.append({
                'name': name,
                'chapter': chapter,
                'type': bench_type,
                'baseline_time_ms': baseline_time,
                'optimized_time_ms': optimized_time,
                'speedup': best_speedup,
                'baseline_memory_mb': baseline_memory_mb,
                'optimized_memory_mb': optimized_memory_mb,
                'memory_savings_pct': best_memory_savings_pct,
                'optimization_goal': optimization_goal,
                'status': status,
                'gpu_temp': gpu_metrics.get('temperature_gpu_c'),
                'gpu_power': gpu_metrics.get('power_draw_w'),
                'gpu_util': gpu_metrics.get('utilization_gpu_pct'),
                'optimizations': optimizations,
                'error': bench.get('error'),
                'p75_ms': bench.get('baseline_p75_ms'),
            })
            
            if best_speedup > 0:
                all_speedups.append(best_speedup)
        
        benchmarks.sort(key=lambda x: x['speedup'], reverse=True)
        
        # Calculate memory optimization count
        memory_optimizations = [b for b in benchmarks if b.get('optimization_goal') == 'memory']
        speed_optimizations = [b for b in benchmarks if b.get('optimization_goal') != 'memory']
        
        return {
            "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": benchmarks,
            "aggregated": True,
            "source_count": len(all_benchmarks),
            "summary": {
                "total_benchmarks": len(benchmarks),
                "avg_speedup": sum(all_speedups) / len(all_speedups) if all_speedups else 0,
                "max_speedup": max(all_speedups) if all_speedups else 0,
                "min_speedup": min(all_speedups) if all_speedups else 0,
                "successful": successful,
                "failed": failed,
                "failed_regression": failed_regression,
                "memory_optimizations": len(memory_optimizations),
                "speed_optimizations": len(speed_optimizations),
                "best_memory_savings_pct": max((b.get('memory_savings_pct', 0) for b in memory_optimizations), default=0),
            }
        }

    def _transform_benchmark_data(self, raw_data: dict) -> dict:
        """Transform raw benchmark data to dashboard format."""
        benchmarks = []
        all_speedups = []
        
        for chapter_result in raw_data.get('results', []):
            chapter = chapter_result.get('chapter', 'unknown')
            
            for bench in chapter_result.get('benchmarks', []):
                name = bench.get('example', 'unknown')
                baseline_time = bench.get('baseline_time_ms', 0)
                best_speedup = bench.get('best_speedup', 1.0)
                status = bench.get('status', 'unknown')
                bench_type = bench.get('type', 'python')
                
                # Get best optimized time
                optimized_time = baseline_time / best_speedup if best_speedup > 0 else baseline_time
                
                # Extract GPU metrics if available
                gpu_metrics = bench.get('baseline_gpu_metrics', {})
                
                # Extract optimization details
                optimizations = []
                for opt in bench.get('optimizations', []):
                    optimizations.append({
                        'technique': opt.get('technique', ''),
                        'speedup': opt.get('speedup', 1.0),
                        'time_ms': opt.get('time_ms', 0),
                        'file': opt.get('file', '')
                    })
                
                # Extract LLM patch info including verification
                llm_patch_info = None
                llm_patches = bench.get('llm_patches', [])
                best_patch = bench.get('best_llm_patch')
                
                # Count refinement attempts across all patches
                total_refinements = sum(1 for p in llm_patches if p.get('refined'))
                total_refinement_attempts = sum(p.get('refinement_attempts', 0) for p in llm_patches)
                
                if best_patch:
                    llm_patch_info = {
                        'variant_name': best_patch.get('variant_name'),
                        'speedup': best_patch.get('actual_speedup'),
                        'refined': best_patch.get('refined', False),
                        'refinement_attempts': best_patch.get('refinement_attempts', 0),
                    }
                
                # Aggregate LLM patch stats
                llm_stats = None
                if llm_patches:
                    successful = [p for p in llm_patches if p.get('success')]
                    failed = [p for p in llm_patches if not p.get('success')]
                    verified = [p for p in llm_patches if p.get('verification', {}).get('verified')]
                    llm_stats = {
                        'total': len(llm_patches),
                        'successful': len(successful),
                        'failed': len(failed),
                        'refined': total_refinements,
                        'refinement_attempts': total_refinement_attempts,
                        'verified': len(verified),
                    }
                
                # Extract verification status - check both LLM patches and direct verification
                verification_status = None
                # First check direct verification (baseline vs optimized)
                direct_verification = bench.get('verification')
                if direct_verification:
                    verification_status = {
                        'verified': direct_verification.get('verified', False),
                        'type': direct_verification.get('verification_type', 'output_comparison'),
                        'errors': direct_verification.get('errors', []),
                        'details': direct_verification.get('details', {}),
                    }
                # Then check LLM patch verification
                for patch in llm_patches:
                    if patch.get('verification'):
                        v = patch['verification']
                        verification_status = {
                            'verified': v.get('verified', False),
                            'type': v.get('verification_type', 'unknown'),
                            'errors': v.get('errors', []),
                            'details': v.get('details', {}),
                        }
                        break
                
                benchmarks.append({
                    'name': name,
                    'chapter': chapter,
                    'type': bench_type,
                    'baseline_time_ms': baseline_time,
                    'optimized_time_ms': optimized_time,
                    'speedup': best_speedup,
                    'status': status,
                    'gpu_temp': gpu_metrics.get('temperature_gpu_c'),
                    'gpu_power': gpu_metrics.get('power_draw_w'),
                    'gpu_util': gpu_metrics.get('utilization_gpu_pct'),
                    'optimizations': optimizations,
                    'error': bench.get('error'),
                    'p75_ms': bench.get('baseline_p75_ms'),
                    'llm_patch': llm_patch_info,
                    'llm_stats': llm_stats,
                    'verification': verification_status,
                })
                
                if best_speedup > 0:
                    all_speedups.append(best_speedup)
        
        # Sort by speedup descending
        benchmarks.sort(key=lambda x: x['speedup'], reverse=True)
        
        # Calculate summary
        summary = raw_data.get('results', [{}])[0].get('summary', {}) if raw_data.get('results') else {}
        
        return {
            "timestamp": raw_data.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")),
            "benchmarks": benchmarks,
            "summary": {
                "total_benchmarks": len(benchmarks),
                "avg_speedup": sum(all_speedups) / len(all_speedups) if all_speedups else 0,
                "max_speedup": max(all_speedups) if all_speedups else 0,
                "min_speedup": min(all_speedups) if all_speedups else 0,
                "successful": summary.get('successful', 0),
                "failed": summary.get('failed', 0),
                "failed_regression": summary.get('failed_regression', 0),
            }
        }
    
    def load_llm_analysis(self) -> dict:
        """Load ALL LLM analysis files from entire codebase."""
        analysis = []
        
        # 1. Scan benchmark_profiles directory
        profiles_dir = CODE_ROOT / 'benchmark_profiles'
        if profiles_dir.exists():
            for md_file in profiles_dir.rglob('llm_analysis*.md'):
                try:
                    content = md_file.read_text()
                    parts = md_file.relative_to(profiles_dir).parts
                    chapter = parts[0] if parts else 'unknown'
                    name = md_file.stem.replace('llm_analysis_', '')
                    
                    analysis.append({
                        'chapter': chapter,
                        'name': name,
                        'content': content,
                        'path': str(md_file.relative_to(CODE_ROOT)),
                        'source': 'benchmark_profiles',
                    })
                except Exception as e:
                    print(f"Error loading {md_file}: {e}")
        
        # 2. Scan ALL chapters for LLM explanation files
        for ch_dir in CODE_ROOT.glob("ch[0-9]*"):
            if ch_dir.is_dir():
                for md_file in ch_dir.glob('*_llm_explanation.md'):
                    try:
                        content = md_file.read_text()
                        analysis.append({
                            'chapter': ch_dir.name,
                            'name': md_file.stem.replace('_llm_explanation', ''),
                            'content': content,
                            'path': str(md_file.relative_to(CODE_ROOT)),
                            'source': 'chapter',
                            'type': 'explanation'
                        })
                    except Exception as e:
                        print(f"Error loading {md_file}: {e}")
        
        # 3. Scan ALL labs for LLM analysis/explanation files
        labs_dir = CODE_ROOT / 'labs'
        if labs_dir.exists():
            for lab_dir in labs_dir.iterdir():
                if lab_dir.is_dir():
                    # Look for any LLM-related markdown
                    for md_file in lab_dir.glob('*llm*.md'):
                        try:
                            content = md_file.read_text()
                            analysis.append({
                                'chapter': f"labs/{lab_dir.name}",
                                'name': md_file.stem,
                                'content': content,
                                'path': str(md_file.relative_to(CODE_ROOT)),
                                'source': 'lab',
                            })
                        except Exception as e:
                            print(f"Error loading {md_file}: {e}")
                    
                    # Note: TECHNIQUE*.md files are reference docs, not LLM analysis - skip them
        
        # Note: Root *ANALYSIS*.md files are reports, not LLM-generated - skip them
        
        return {
            "analyses": analysis, 
            "count": len(analysis),
            "sources": {
                "benchmark_profiles": len([a for a in analysis if a.get('source') == 'benchmark_profiles']),
                "chapters": len([a for a in analysis if a.get('source') == 'chapter']),
                "labs": len([a for a in analysis if a.get('source') == 'lab']),
                "root": len([a for a in analysis if a.get('source') == 'root']),
            }
        }
    
    def load_profile_data(self) -> dict:
        """Load available profile data from ALL sources."""
        profiles = []
        
        # Scan benchmark_profiles directory
        profiles_dir = CODE_ROOT / 'benchmark_profiles'
        if profiles_dir.exists():
            for chapter_dir in profiles_dir.iterdir():
                if chapter_dir.is_dir():
                    chapter = chapter_dir.name
                    chapter_profiles = {
                        'chapter': chapter,
                        'nsys_reports': [],
                        'ncu_reports': [],
                        'torch_traces': [],
                        'sqlite_dbs': [],
                    }
                    
                    for f in chapter_dir.iterdir():
                        if f.suffix == '.nsys-rep':
                            chapter_profiles['nsys_reports'].append(f.name)
                        elif f.suffix == '.ncu-rep':
                            chapter_profiles['ncu_reports'].append(f.name)
                        elif f.suffix == '.json' and 'torch_trace' in f.name:
                            chapter_profiles['torch_traces'].append(f.name)
                        elif f.suffix == '.sqlite':
                            chapter_profiles['sqlite_dbs'].append(f.name)
                    
                    if any([chapter_profiles['nsys_reports'], 
                            chapter_profiles['ncu_reports'],
                            chapter_profiles['torch_traces'],
                            chapter_profiles['sqlite_dbs']]):
                        profiles.append(chapter_profiles)
        
        return {
            "profiles": profiles,
            "total_chapters_with_profiles": len(profiles),
            "total_nsys": sum(len(p['nsys_reports']) for p in profiles),
            "total_ncu": sum(len(p['ncu_reports']) for p in profiles),
            "total_torch_traces": sum(len(p['torch_traces']) for p in profiles),
        }
    
    def get_gpu_info(self) -> dict:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu,clocks.current.graphics,clocks.current.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                return {
                    "name": parts[0],
                    "temperature": float(parts[1]),
                    "power": float(parts[2]),
                    "memory_used": float(parts[3]),
                    "memory_total": float(parts[4]),
                    "utilization": float(parts[5]),
                    "clock_graphics": int(float(parts[6])) if len(parts) > 6 else None,
                    "clock_memory": int(float(parts[7])) if len(parts) > 7 else None,
                    "live": True
                }
        except Exception:
            pass
        
        return {
            "name": "NVIDIA B200",
            "temperature": 42,
            "power": 192,
            "memory_used": 1024,
            "memory_total": 196608,
            "utilization": 0,
            "clock_graphics": 1965,
            "clock_memory": 3996,
            "live": False
        }
    
    # =========================================================================
    # DEEP PROFILE COMPARISON (nsys/ncu metrics)
    # =========================================================================
    
    def list_deep_profile_pairs(self) -> dict:
        """List all chapters/benchmarks that have both baseline and optimized profiles."""
        pairs = []
        profiles_dir = CODE_ROOT / 'benchmark_profiles'
        
        def scan_profile_dir(chapter_dir: Path, prefix: str = "") -> Optional[dict]:
            """Scan a directory for profile pairs."""
            chapter = prefix + chapter_dir.name if prefix else chapter_dir.name
            
            baseline_nsys = list(chapter_dir.glob("*baseline*.nsys-rep"))
            baseline_ncu = list(chapter_dir.glob("*baseline*.ncu-rep"))
            optimized_nsys = list(chapter_dir.glob("*optimized*.nsys-rep"))
            optimized_ncu = list(chapter_dir.glob("*optimized*.ncu-rep"))
            
            # Also check for deep profile JSON reports
            baseline_json = list(chapter_dir.glob("*baseline*_deep_profile.json"))
            optimized_json = list(chapter_dir.glob("*optimized*_deep_profile.json"))
            
            if (baseline_nsys or baseline_ncu or baseline_json) and \
               (optimized_nsys or optimized_ncu or optimized_json):
                return {
                    "chapter": chapter,
                    "path": str(chapter_dir.relative_to(CODE_ROOT)),
                    "has_nsys": bool(baseline_nsys and optimized_nsys),
                    "has_ncu": bool(baseline_ncu and optimized_ncu),
                    "has_deep_json": bool(baseline_json and optimized_json),
                    "baseline_files": {
                        "nsys": [f.name for f in baseline_nsys],
                        "ncu": [f.name for f in baseline_ncu],
                        "json": [f.name for f in baseline_json],
                    },
                    "optimized_files": {
                        "nsys": [f.name for f in optimized_nsys],
                        "ncu": [f.name for f in optimized_ncu],
                        "json": [f.name for f in optimized_json],
                    }
                }
            return None
        
        if profiles_dir.exists():
            for chapter_dir in sorted(profiles_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                
                # Check if this is the labs directory
                if chapter_dir.name == 'labs':
                    # Scan subdirectories of labs
                    for lab_dir in sorted(chapter_dir.iterdir()):
                        if lab_dir.is_dir():
                            result = scan_profile_dir(lab_dir, "labs/")
                            if result:
                                pairs.append(result)
                else:
                    result = scan_profile_dir(chapter_dir)
                    if result:
                        pairs.append(result)
        
        return {
            "pairs": pairs,
            "count": len(pairs),
        }
    
    def compare_profiles(self, chapter: str) -> dict:
        """Compare baseline vs optimized profiles for a chapter using differential analyzer."""
        # Handle URL-encoded paths (e.g., labs%2Fblackwell_matmul)
        import urllib.parse
        chapter = urllib.parse.unquote(chapter)
        
        profiles_dir = CODE_ROOT / 'benchmark_profiles' / chapter
        
        if not profiles_dir.exists():
            return {"error": f"Profile directory not found for {chapter}"}
        
        # Try to use the differential profile analyzer if we have JSON deep profiles
        baseline_jsons = list(profiles_dir.glob("*baseline*_deep_profile.json"))
        optimized_jsons = list(profiles_dir.glob("*optimized*_deep_profile.json"))
        
        result = {
            "chapter": chapter,
            "comparison": None,
            "nsys_comparison": None,
            "ncu_comparison": None,
            "recommendations": [],
            "why_faster": None,
            "how_to_improve": [],
        }
        
        # Use differential analyzer if available
        if baseline_jsons and optimized_jsons:
            try:
                from tools.analysis.differential_profile_analyzer import analyze_differential
                report = analyze_differential(baseline_jsons[0], optimized_jsons[0])
                result["comparison"] = report.to_dict()
                result["why_faster"] = self._format_why_faster(report)
                result["how_to_improve"] = report.next_steps
                result["recommendations"] = report.remaining_bottlenecks
            except Exception as e:
                result["differential_error"] = str(e)
        
        # Extract nsys metrics comparison
        result["nsys_comparison"] = self._compare_nsys_files(profiles_dir)
        
        # Extract ncu metrics comparison
        result["ncu_comparison"] = self._compare_ncu_files(profiles_dir)
        
        # Generate recommendations based on available data
        if not result["recommendations"]:
            result["recommendations"] = self._generate_recommendations_from_profiles(result)
        
        return result
    
    def _format_why_faster(self, report) -> dict:
        """Format 'why faster' explanation from differential report."""
        time_saved_ms = report.total_baseline_time_ms - report.total_optimized_time_ms
        
        return {
            "time_saved_ms": time_saved_ms,
            "speedup": report.overall_speedup,
            "binding_shift": report.binding_shift,
            "key_improvements": report.key_improvements,
            "attribution": report.improvement_attribution.to_dict() if report.improvement_attribution else {},
        }
    
    def _compare_nsys_files(self, profiles_dir: Path) -> Optional[dict]:
        """Extract and compare nsys metrics between baseline and optimized."""
        baseline_nsys = list(profiles_dir.glob("*baseline*.nsys-rep"))
        optimized_nsys = list(profiles_dir.glob("*optimized*.nsys-rep"))
        
        if not baseline_nsys or not optimized_nsys:
            return None
        
        try:
            from tools.profiling.extract_nsys_summary import harvest
            
            baseline_metrics = harvest(baseline_nsys[0])
            optimized_metrics = harvest(optimized_nsys[0])
            
            # Build comparison
            comparison = {
                "baseline_file": baseline_nsys[0].name,
                "optimized_file": optimized_nsys[0].name,
                "metrics": [],
            }
            
            # Create lookup for optimized metrics
            opt_lookup = {}
            for m in optimized_metrics:
                key = m.get('metric', '')
                opt_lookup[key] = m.get('value', '')
            
            for bm in baseline_metrics:
                metric = bm.get('metric', '')
                baseline_val = bm.get('value', '')
                optimized_val = opt_lookup.get(metric, '')
                
                # Try to compute delta for numeric values
                delta = None
                try:
                    b_num = float(baseline_val.replace(',', ''))
                    o_num = float(optimized_val.replace(',', ''))
                    delta = o_num - b_num
                    delta_pct = ((o_num - b_num) / b_num * 100) if b_num != 0 else 0
                except (ValueError, TypeError):
                    delta_pct = None
                
                comparison["metrics"].append({
                    "name": metric,
                    "baseline": baseline_val,
                    "optimized": optimized_val,
                    "delta": delta,
                    "delta_pct": delta_pct,
                })
            
            return comparison
        except Exception as e:
            return {"error": str(e)}
    
    def _compare_ncu_files(self, profiles_dir: Path) -> Optional[dict]:
        """Extract and compare ncu metrics between baseline and optimized."""
        baseline_ncu = list(profiles_dir.glob("*baseline*.ncu-rep"))
        optimized_ncu = list(profiles_dir.glob("*optimized*.ncu-rep"))
        
        # Also check for CSV exports
        baseline_csv = list(profiles_dir.glob("*baseline*ncu*.csv"))
        optimized_csv = list(profiles_dir.glob("*optimized*ncu*.csv"))
        
        if baseline_csv and optimized_csv:
            # Use CSV files directly
            try:
                import csv
                
                def read_ncu_csv(path):
                    metrics = {}
                    with open(path, newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            name = row.get('Metric Name', row.get('Name', ''))
                            value = row.get('Metric Value', row.get('Avg', row.get('Value', '')))
                            if name and value:
                                metrics[name] = value
                    return metrics
                
                baseline_metrics = read_ncu_csv(baseline_csv[0])
                optimized_metrics = read_ncu_csv(optimized_csv[0])
                
                comparison = {
                    "baseline_file": baseline_csv[0].name,
                    "optimized_file": optimized_csv[0].name,
                    "metrics": [],
                }
                
                all_keys = set(baseline_metrics.keys()) | set(optimized_metrics.keys())
                for key in sorted(all_keys):
                    b_val = baseline_metrics.get(key, '')
                    o_val = optimized_metrics.get(key, '')
                    
                    delta = None
                    try:
                        b_num = float(str(b_val).replace(',', ''))
                        o_num = float(str(o_val).replace(',', ''))
                        delta = o_num - b_num
                    except (ValueError, TypeError):
                        pass
                    
                    comparison["metrics"].append({
                        "name": key,
                        "baseline": b_val,
                        "optimized": o_val,
                        "delta": delta,
                    })
                
                return comparison
            except Exception as e:
                return {"error": str(e)}
        
        if not baseline_ncu or not optimized_ncu:
            return None
        
        # Try to extract from .ncu-rep files using ncu CLI
        try:
            def extract_ncu_metrics(ncu_path):
                result = subprocess.run(
                    ['ncu', '--import', str(ncu_path), '--csv'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    return {}
                
                metrics = {}
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ',' in line and not line.startswith('=='):
                        parts = line.split(',')
                        if len(parts) >= 2:
                            metrics[parts[0]] = parts[1] if len(parts) > 1 else ''
                return metrics
            
            baseline_metrics = extract_ncu_metrics(baseline_ncu[0])
            optimized_metrics = extract_ncu_metrics(optimized_ncu[0])
            
            if baseline_metrics or optimized_metrics:
                return {
                    "baseline_file": baseline_ncu[0].name,
                    "optimized_file": optimized_ncu[0].name,
                    "baseline_metrics": baseline_metrics,
                    "optimized_metrics": optimized_metrics,
                }
        except Exception as e:
            return {"error": f"NCU extraction failed: {e}"}
        
        return None
    
    def _generate_recommendations_from_profiles(self, result: dict) -> List[str]:
        """Generate recommendations based on profile comparison data."""
        recommendations = []
        
        # Check nsys comparison for memory bound indicators
        nsys = result.get("nsys_comparison", {})
        if nsys and isinstance(nsys, dict) and "metrics" in nsys:
            for m in nsys["metrics"]:
                name = m.get("name", "").lower()
                delta = m.get("delta")
                
                if delta and "dram" in name and delta < -10:
                    recommendations.append(
                        f"Memory traffic reduced: Consider further optimization with TMA prefetching"
                    )
                elif delta and "sm" in name and delta > 10:
                    recommendations.append(
                        f"SM utilization improved by {delta:.1f}%: Good progress on compute efficiency"
                    )
        
        # Check ncu comparison for kernel-level insights
        ncu = result.get("ncu_comparison", {})
        if ncu and isinstance(ncu, dict) and "metrics" in ncu:
            for m in ncu["metrics"]:
                name = m.get("name", "").lower()
                if "occupancy" in name:
                    try:
                        opt_val = float(str(m.get("optimized", "0")).replace('%', ''))
                        if opt_val < 50:
                            recommendations.append(
                                f"Occupancy at {opt_val:.0f}%: Consider tuning block size or reducing register pressure"
                            )
                    except (ValueError, TypeError):
                        pass
        
        if not recommendations:
            recommendations.append("Profile both baseline and optimized to get detailed recommendations")
        
        return recommendations
    
    def get_profile_recommendations(self) -> dict:
        """Get aggregated recommendations from all profile comparisons."""
        pairs = self.list_deep_profile_pairs()
        all_recommendations = []
        
        for pair in pairs.get("pairs", []):
            chapter = pair["chapter"]
            comparison = self.compare_profiles(chapter)
            
            if comparison.get("recommendations"):
                all_recommendations.append({
                    "chapter": chapter,
                    "recommendations": comparison["recommendations"],
                    "why_faster": comparison.get("why_faster"),
                })
        
        return {
            "chapters_analyzed": len(all_recommendations),
            "recommendations": all_recommendations,
        }
    
    # =========================================================================
    # LIVE OPTIMIZATION STREAMING (SSE)
    # =========================================================================
    
    def stream_optimization_events(self, job_id: str):
        """Stream optimization events using Server-Sent Events (SSE)."""
        global _job_events
        
        if job_id not in _job_events:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Job not found"}).encode())
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        event_queue = _job_events[job_id]
        
        try:
            while True:
                try:
                    # Wait for events with timeout
                    event = event_queue.get(timeout=30)
                    
                    if event is None:
                        # Job completed
                        self.wfile.write(b"event: complete\ndata: {}\n\n")
                        self.wfile.flush()
                        break
                    
                    # Send event
                    event_type = event.get("type", "message")
                    event_data = json.dumps(event)
                    self.wfile.write(f"event: {event_type}\ndata: {event_data}\n\n".encode())
                    self.wfile.flush()
                    
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected
    
    def start_optimization_job(self, params: dict) -> dict:
        """Start a new optimization job with live streaming."""
        global _optimization_jobs, _job_events
        
        job_id = str(uuid.uuid4())[:8]
        target = params.get("target", "")
        
        if not target:
            return {"error": "No target specified. Provide 'target' as chapter:example or chapter"}
        
        # Create event queue for this job
        _job_events[job_id] = queue.Queue()
        
        # Create job record
        job = {
            "id": job_id,
            "target": target,
            "status": "starting",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "llm_analysis": params.get("llm_analysis", True),
            "apply_patches": params.get("apply_patches", True),
            "rebenchmark": params.get("rebenchmark", True),
            "events": [],
        }
        _optimization_jobs[job_id] = job
        
        # Start optimization in background thread
        def run_optimization():
            self._run_optimization_job(job_id, params)
        
        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()
        
        return {
            "job_id": job_id,
            "status": "started",
            "stream_url": f"/api/optimize/stream/{job_id}",
        }
    
    def _run_optimization_job(self, job_id: str, params: dict):
        """Run the optimization job and emit events."""
        global _optimization_jobs, _job_events
        
        job = _optimization_jobs.get(job_id)
        event_queue = _job_events.get(job_id)
        
        if not job or not event_queue:
            return
        
        def emit(event_type: str, data: dict):
            event = {"type": event_type, "timestamp": time.strftime("%H:%M:%S"), **data}
            job["events"].append(event)
            event_queue.put(event)
        
        try:
            emit("status", {"message": "🚀 Starting optimization job...", "status": "running"})
            job["status"] = "running"
            
            target = params.get("target", "")
            llm_analysis = params.get("llm_analysis", True)
            apply_patches = params.get("apply_patches", True)
            rebenchmark = params.get("rebenchmark", True)
            
            # Build the benchmark_cli command
            cmd = [sys.executable, "-m", "tools.cli.benchmark_cli", "run", "-t", target]
            
            if llm_analysis:
                cmd.append("--llm-analysis")
                emit("info", {"message": "📊 LLM analysis enabled"})
            if apply_patches:
                cmd.append("--apply-llm-patches")
                emit("info", {"message": "🔧 Patch application enabled"})
            if rebenchmark:
                cmd.append("--rebenchmark-llm-patches")
                emit("info", {"message": "⏱️ Rebenchmarking enabled"})
            
            emit("command", {"message": f"Running: {' '.join(cmd)}"})
            
            # Run the command and stream output
            process = subprocess.Popen(
                cmd,
                cwd=str(CODE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            job["pid"] = process.pid
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                if not line:
                    continue
                
                # Categorize output
                event_type = "output"
                if "🧠" in line or "LLM" in line.upper():
                    event_type = "llm"
                elif "📊" in line or "BASELINE" in line or "BENCHMARK" in line:
                    event_type = "benchmark"
                elif "🔧" in line or "PATCH" in line.upper():
                    event_type = "patch"
                elif "✅" in line or "SUCCEEDED" in line.upper():
                    event_type = "success"
                elif "❌" in line or "FAILED" in line.upper() or "ERROR" in line.upper():
                    event_type = "error"
                elif "⚡" in line or "SPEEDUP" in line.upper():
                    event_type = "speedup"
                
                emit(event_type, {"message": line})
            
            process.wait()
            
            if process.returncode == 0:
                emit("complete", {"message": "✅ Optimization completed successfully!", "status": "completed"})
                job["status"] = "completed"
            else:
                emit("error", {"message": f"❌ Optimization failed with exit code {process.returncode}", "status": "failed"})
                job["status"] = "failed"
            
        except Exception as e:
            emit("error", {"message": f"❌ Error: {str(e)}", "status": "error"})
            job["status"] = "error"
        finally:
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            event_queue.put(None)  # Signal completion
    
    def stop_optimization_job(self, job_id: str) -> dict:
        """Stop a running optimization job."""
        global _optimization_jobs, _job_events
        
        if not job_id or job_id not in _optimization_jobs:
            return {"error": "Job not found"}
        
        job = _optimization_jobs[job_id]
        
        if job.get("pid"):
            try:
                import signal
                os.kill(job["pid"], signal.SIGTERM)
                job["status"] = "stopped"
                
                if job_id in _job_events:
                    _job_events[job_id].put({"type": "stopped", "message": "Job stopped by user"})
                    _job_events[job_id].put(None)
                
                return {"status": "stopped", "job_id": job_id}
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "Job has no active process"}
    
    def list_optimization_jobs(self) -> dict:
        """List all optimization jobs."""
        global _optimization_jobs
        
        jobs = []
        for job_id, job in _optimization_jobs.items():
            jobs.append({
                "id": job_id,
                "target": job.get("target"),
                "status": job.get("status"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "event_count": len(job.get("events", [])),
            })
        
        return {"jobs": jobs, "count": len(jobs)}
    
    # =========================================================================
    # Multi-Metric Analysis Methods
    # =========================================================================
    
    def get_pareto_frontier(self) -> dict:
        """Calculate Pareto-optimal benchmarks across speed and memory dimensions.
        
        A benchmark is Pareto-optimal if no other benchmark is better on ALL metrics.
        """
        data = self.load_benchmark_data()
        benchmarks = data.get('benchmarks', [])
        
        # Normalize metrics for comparison
        # Speed: higher is better (use speedup)
        # Memory: higher savings is better (use memory_savings_pct)
        points = []
        for b in benchmarks:
            speedup = b.get('speedup', 1.0)
            memory_savings = b.get('memory_savings_pct', 0) or 0
            points.append({
                'name': f"{b.get('chapter')}:{b.get('name')}",
                'speedup': speedup,
                'memory_savings': memory_savings,
                'goal': b.get('optimization_goal', 'speed'),
                'is_pareto': False,
            })
        
        # Find Pareto frontier
        pareto_points = []
        for i, p in enumerate(points):
            is_dominated = False
            for j, q in enumerate(points):
                if i != j:
                    # q dominates p if q is better on ALL metrics
                    if q['speedup'] >= p['speedup'] and q['memory_savings'] >= p['memory_savings']:
                        if q['speedup'] > p['speedup'] or q['memory_savings'] > p['memory_savings']:
                            is_dominated = True
                            break
            if not is_dominated:
                p['is_pareto'] = True
                pareto_points.append(p)
        
        return {
            'pareto_frontier': pareto_points,
            'all_points': points,
            'pareto_count': len(pareto_points),
            'total_count': len(points),
        }
    
    def get_tradeoff_analysis(self) -> dict:
        """Analyze speed vs memory trade-offs for all benchmarks."""
        data = self.load_benchmark_data()
        benchmarks = data.get('benchmarks', [])
        
        tradeoffs = []
        for b in benchmarks:
            speedup = b.get('speedup', 1.0)
            memory_savings = b.get('memory_savings_pct', 0) or 0
            goal = b.get('optimization_goal', 'speed')
            
            # Calculate cost-benefit ratio
            # For memory optimizations: how much speed lost per % memory saved
            # For speed optimizations: memory change per speedup gain
            if goal == 'memory' and memory_savings > 0:
                speed_cost = max(0, 1 - speedup)  # How much slower (0-1)
                cost_per_percent = (speed_cost * 100) / memory_savings if memory_savings > 0 else 0
                efficiency = memory_savings / (speed_cost * 100 + 1)  # Higher is better
                benefit_label = f"-{memory_savings:.0f}% memory"
                cost_label = f"{speed_cost*100:.0f}% slower" if speed_cost > 0 else "No speed loss"
            else:
                speed_gain = max(0, speedup - 1)
                memory_cost = max(0, -memory_savings) if memory_savings else 0
                efficiency = speedup / (memory_cost + 1)  # Higher is better
                benefit_label = f"{speedup:.2f}x faster"
                cost_label = f"+{memory_cost:.0f}% memory" if memory_cost > 0 else "No memory increase"
                cost_per_percent = memory_cost / (speed_gain * 100 + 1) if speed_gain > 0 else 0
            
            tradeoffs.append({
                'name': f"{b.get('chapter')}:{b.get('name')}",
                'goal': goal,
                'speedup': speedup,
                'memory_savings': memory_savings,
                'efficiency_score': round(efficiency, 2),
                'cost_per_percent': round(cost_per_percent, 3),
                'benefit': benefit_label,
                'cost': cost_label,
            })
        
        # Sort by efficiency
        tradeoffs.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return {
            'tradeoffs': tradeoffs,
            'best_efficiency': tradeoffs[0] if tradeoffs else None,
            'memory_specialists': [t for t in tradeoffs if t['goal'] == 'memory'][:5],
            'speed_specialists': [t for t in tradeoffs if t['goal'] == 'speed'][:5],
        }
    
    def get_constraint_recommendations(self) -> dict:
        """Provide recommendations based on common constraint scenarios."""
        data = self.load_benchmark_data()
        benchmarks = data.get('benchmarks', [])
        
        # Build constraint scenarios
        scenarios = []
        
        # Scenario 1: Memory-constrained (need to reduce memory)
        memory_opts = sorted(
            [b for b in benchmarks if (b.get('memory_savings_pct') or 0) > 10],
            key=lambda x: x.get('memory_savings_pct', 0),
            reverse=True
        )
        scenarios.append({
            'name': 'Memory Constrained',
            'description': 'Need to fit larger models/batches in limited VRAM',
            'icon': '💾',
            'recommendations': [{
                'name': f"{b.get('chapter')}:{b.get('name')}",
                'benefit': f"-{b.get('memory_savings_pct', 0):.0f}% memory",
                'tradeoff': f"{b.get('speedup', 1):.2f}x speed",
            } for b in memory_opts[:3]],
        })
        
        # Scenario 2: Latency-critical (need maximum speed)
        speed_opts = sorted(
            [b for b in benchmarks if b.get('speedup', 1) > 1.5],
            key=lambda x: x.get('speedup', 1),
            reverse=True
        )
        scenarios.append({
            'name': 'Latency Critical',
            'description': 'Need lowest possible inference/training time',
            'icon': '⚡',
            'recommendations': [{
                'name': f"{b.get('chapter')}:{b.get('name')}",
                'benefit': f"{b.get('speedup', 1):.2f}x faster",
                'tradeoff': f"{b.get('memory_savings_pct', 0):.0f}% mem change" if b.get('memory_savings_pct') else "Minimal memory impact",
            } for b in speed_opts[:3]],
        })
        
        # Scenario 3: Balanced (want both speed and memory)
        balanced = sorted(
            [b for b in benchmarks if b.get('speedup', 1) >= 1.0 and (b.get('memory_savings_pct') or 0) >= 0],
            key=lambda x: x.get('speedup', 1) + (x.get('memory_savings_pct', 0) or 0) / 50,
            reverse=True
        )
        scenarios.append({
            'name': 'Balanced',
            'description': 'Want improvements without significant trade-offs',
            'icon': '⚖️',
            'recommendations': [{
                'name': f"{b.get('chapter')}:{b.get('name')}",
                'benefit': f"{b.get('speedup', 1):.2f}x speed",
                'tradeoff': f"{b.get('memory_savings_pct', 0):.0f}% mem" if b.get('memory_savings_pct') else "Neutral",
            } for b in balanced[:3]],
        })
        
        # Scenario 4: Training large models
        training_opts = [b for b in benchmarks if 'training' in b.get('name', '').lower()]
        if training_opts:
            scenarios.append({
                'name': 'Large Model Training',
                'description': 'Training models that barely fit in memory',
                'icon': '🏋️',
                'recommendations': [{
                    'name': f"{b.get('chapter')}:{b.get('name')}",
                    'benefit': f"-{b.get('memory_savings_pct', 0):.0f}% memory" if b.get('memory_savings_pct', 0) > 0 else f"{b.get('speedup', 1):.2f}x speed",
                    'tradeoff': b.get('optimization_goal', 'speed'),
                } for b in training_opts[:3]],
            })
        
        return {
            'scenarios': scenarios,
            'total_benchmarks': len(benchmarks),
        }
    
    def get_categorized_leaderboards(self) -> dict:
        """Return separate leaderboards for each optimization category."""
        data = self.load_benchmark_data()
        benchmarks = data.get('benchmarks', [])
        
        # Categorize benchmarks
        speed_benchmarks = [b for b in benchmarks if b.get('optimization_goal', 'speed') == 'speed']
        memory_benchmarks = [b for b in benchmarks if b.get('optimization_goal') == 'memory']
        throughput_benchmarks = [b for b in benchmarks if b.get('optimization_goal') == 'throughput']
        
        # Sort each category by its primary metric
        speed_benchmarks.sort(key=lambda x: x.get('speedup', 0), reverse=True)
        memory_benchmarks.sort(key=lambda x: x.get('memory_savings_pct', 0) or 0, reverse=True)
        throughput_benchmarks.sort(key=lambda x: x.get('speedup', 0), reverse=True)  # Use speedup as proxy
        
        def format_benchmark(b, category):
            if category == 'memory':
                return {
                    'rank': 0,  # Will be set below
                    'name': f"{b.get('chapter')}:{b.get('name')}",
                    'primary_metric': f"-{b.get('memory_savings_pct', 0):.0f}%",
                    'secondary_metric': f"{b.get('speedup', 1):.2f}x speed",
                    'value': b.get('memory_savings_pct', 0) or 0,
                }
            else:
                return {
                    'rank': 0,
                    'name': f"{b.get('chapter')}:{b.get('name')}",
                    'primary_metric': f"{b.get('speedup', 1):.2f}x",
                    'secondary_metric': f"{b.get('memory_savings_pct', 0):.0f}% mem" if b.get('memory_savings_pct') else "",
                    'value': b.get('speedup', 1),
                }
        
        # Build leaderboards with rankings
        speed_board = [format_benchmark(b, 'speed') for b in speed_benchmarks[:10]]
        for i, b in enumerate(speed_board):
            b['rank'] = i + 1
        
        memory_board = [format_benchmark(b, 'memory') for b in memory_benchmarks[:10]]
        for i, b in enumerate(memory_board):
            b['rank'] = i + 1
        
        throughput_board = [format_benchmark(b, 'throughput') for b in throughput_benchmarks[:10]]
        for i, b in enumerate(throughput_board):
            b['rank'] = i + 1
        
        return {
            'leaderboards': {
                'speed': {
                    'title': '🚀 Speed Champions',
                    'metric_label': 'Speedup',
                    'entries': speed_board,
                    'count': len(speed_benchmarks),
                },
                'memory': {
                    'title': '💾 Memory Champions',
                    'metric_label': 'Memory Saved',
                    'entries': memory_board,
                    'count': len(memory_benchmarks),
                },
                'throughput': {
                    'title': '📈 Throughput Champions',
                    'metric_label': 'Throughput',
                    'entries': throughput_board,
                    'count': len(throughput_benchmarks),
                },
            },
            'summary': {
                'speed_count': len(speed_benchmarks),
                'memory_count': len(memory_benchmarks),
                'throughput_count': len(throughput_benchmarks),
            }
        }
    
    def log_message(self, format, *args):
        """Suppress logging for cleaner output."""
        pass


def create_handler(data_file: Optional[Path] = None):
    """Create a handler class with the data file bound."""
    def handler(*args, **kwargs):
        return DashboardHandler(*args, data_file=data_file, **kwargs)
    return handler


def serve_dashboard(port: int = 8080, data_file: Optional[Path] = None, open_browser: bool = True):
    """Start the dashboard server."""
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    handler = create_handler(data_file)
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ⚡ GPU Performance Lab Dashboard                                     ║
║                                                                        ║
║   Server running at: {url:<50} ║
║   Data source: {str(data_file or 'benchmark_test_results.json')[:50]:<50} ║
║                                                                        ║
║   📊 Data APIs:                                                        ║
║   • GET /api/data              - Benchmark results                     ║
║   • GET /api/gpu               - Live GPU status                       ║
║   • GET /api/llm-analysis      - LLM insights & explanations           ║
║   • GET /api/profiles          - Available profile data                ║
║                                                                        ║
║   🔬 Deep Profile Comparison (NEW!):                                   ║
║   • GET /api/deep-profile/list        - List comparable profiles       ║
║   • GET /api/deep-profile/compare/:ch - nsys/ncu metrics comparison    ║
║   • GET /api/deep-profile/recommendations - Analysis & recommendations ║
║                                                                        ║
║   🚀 Live Optimization Console (NEW!):                                 ║
║   • POST /api/optimize/start   - Start optimization with streaming     ║
║   • GET /api/optimize/stream/:id - SSE stream for live updates         ║
║   • GET /api/optimize/jobs     - List all optimization jobs            ║
║                                                                        ║
║   Press Ctrl+C to stop                                                 ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
        """)
        
        if open_browser:
            # Open browser after a short delay
            def open_delayed():
                time.sleep(0.5)
                webbrowser.open(url)
            threading.Thread(target=open_delayed, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard server stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="GPU Performance Lab Dashboard Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.dashboard.server
  python -m tools.dashboard.server --port 3000
  python -m tools.dashboard.server --data artifacts/benchmark_test_results.json
        """
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8080,
        help='Port to run the server on (default: 8080)'
    )
    parser.add_argument(
        '--data', '-d',
        type=Path,
        default=None,
        help='Path to benchmark results JSON file'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    args = parser.parse_args()
    serve_dashboard(port=args.port, data_file=args.data, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
