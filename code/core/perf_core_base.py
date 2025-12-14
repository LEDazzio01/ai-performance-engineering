"""
PerformanceCoreBase: shared, non-HTTP performance helpers.

This is the logic side of the old dashboard handler, split out so CLI/MCP
can reuse data loading, profiling artifacts, GPU/system inspection, and
benchmark discovery without depending on the HTTP server.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    load_benchmark_data as load_benchmark_results,
)
from core import profile_artifacts
from core.compile_analysis import load_compile_analysis
from core.discovery import get_bench_roots, discover_all_chapters

CODE_ROOT = Path(__file__).resolve().parents[1]
_HISTORY_CACHE: Dict[str, Any] = {"key": None, "runs": None, "trends": None}


class PerformanceCoreBase:
    """Shared performance logic without HTTP concerns."""

    def __init__(self, data_file: Optional[Path] = None, bench_root: Optional[Path] = None):
        self.data_file = data_file
        self.bench_roots = get_bench_roots(repo_root=CODE_ROOT, bench_root=bench_root)
        self.bench_root = self.bench_roots[0]
        self._analyzer: Optional[PerformanceAnalyzer] = None
        self._make_analyzer()

    def _make_analyzer(self) -> None:
        """Bind an analyzer to the current bench roots and data file."""
        self._analyzer = PerformanceAnalyzer(
            lambda: load_benchmark_results(self.data_file, self.bench_roots)
        )

    def set_bench_root(self, bench_root: Path) -> dict:
        """Dynamically update the benchmark root without restarting the server."""
        new_roots = get_bench_roots(repo_root=CODE_ROOT, bench_root=bench_root)
        self.bench_roots = new_roots
        self.bench_root = new_roots[0]
        self._make_analyzer()
        _HISTORY_CACHE["key"] = None
        _HISTORY_CACHE["runs"] = None
        _HISTORY_CACHE["trends"] = None
        return {"bench_root": str(self.bench_root)}

    @property
    def analyzer(self) -> PerformanceAnalyzer:
        if not hasattr(self, "_analyzer") or self._analyzer is None:
            data_path = getattr(self, "data_file", None)
            self._analyzer = PerformanceAnalyzer(lambda: load_benchmark_results(data_path, self.bench_roots))
        return self._analyzer

    # ------------------------------------------------------------------
    # Benchmark data + exports
    # ------------------------------------------------------------------
    def load_benchmark_data(self) -> dict:
        return load_benchmark_results(self.data_file, self.bench_roots)

    def export_benchmarks_csv(self) -> str:
        data = self.load_benchmark_data()
        return profile_artifacts.export_benchmarks_csv(data)

    def export_detailed_csv(self) -> str:
        data = self.load_benchmark_data()
        return profile_artifacts.export_detailed_csv(data)

    # ------------------------------------------------------------------
    # Profiling artifact helpers
    # ------------------------------------------------------------------
    def get_flame_graph_data(self) -> dict:
        return profile_artifacts.load_flame_graph_data(self.bench_root)

    def get_memory_timeline(self) -> dict:
        return profile_artifacts.load_memory_timeline(self.bench_root)

    def get_cpu_gpu_timeline(self) -> dict:
        return profile_artifacts.load_cpu_gpu_timeline(self.bench_root)

    def get_kernel_breakdown(self) -> dict:
        return profile_artifacts.load_kernel_breakdown(self.get_flame_graph_data())

    def get_hta_analysis(self) -> dict:
        hta_data = profile_artifacts.load_hta_analysis(self.bench_root)
        if not hta_data.get("top_kernels"):
            kernel_data = self.get_kernel_breakdown()
            total_time = kernel_data.get("summary", {}).get("total_time_us", 0)
            if total_time > 0:
                for kernel in kernel_data.get("kernels", [])[:10]:
                    hta_data.setdefault("top_kernels", []).append({
                        "name": kernel.get("name"),
                        "time_us": kernel.get("time_us"),
                        "pct": kernel.get("time_us", 0) / total_time * 100 if total_time else 0,
                    })
            if kernel_data.get("by_type"):
                top_type = max(kernel_data["by_type"].items(), key=lambda x: x[1])
                hta_data.setdefault("recommendations", []).append(
                    f"Optimize {top_type[0]} operations ({top_type[1]/1000:.1f}ms total)"
                )
        return hta_data

    def get_torch_profiler(self) -> dict:
        """Load the latest torch.profiler capture summary."""
        return profile_artifacts.load_torch_profiler(self.bench_root)

    def get_compile_analysis(self) -> dict:
        benchmarks = self.load_benchmark_data().get("benchmarks", [])
        return load_compile_analysis(self.bench_root, benchmarks)

    def get_roofline_data(self) -> dict:
        roofline_data = {
            "has_real_data": False,
            "baseline_points": [],
            "optimized_points": [],
            "hardware_specs": {},
            "benchmark_details": [],
        }

        gpu_info = self.get_gpu_info()
        gpu_name = gpu_info.get("name", "Unknown GPU")
        if "B200" in gpu_name or "B300" in gpu_name:
            roofline_data["hardware_specs"] = {
                "name": gpu_name,
                "memory_bandwidth_gb_s": 8000,
                "peak_tflops": 2500,
            }
        elif "H100" in gpu_name:
            roofline_data["hardware_specs"] = {
                "name": gpu_name,
                "memory_bandwidth_gb_s": 3350,
                "peak_tflops": 120,
            }
        else:
            roofline_data["hardware_specs"] = {
                "name": gpu_name,
                "memory_bandwidth_gb_s": None,
                "peak_tflops": None,
            }

        try:
            data = self.load_benchmark_data().get("benchmarks", [])
            for bench in data:
                if "baseline_time_ms" in bench and "optimized_time_ms" in bench:
                    baseline_ms = bench["baseline_time_ms"]
                    optimized_ms = bench["optimized_time_ms"]
                    speedup = bench.get("speedup", baseline_ms / optimized_ms if optimized_ms else 0)
                    ai_estimate = bench.get("arithmetic_intensity", None)

                    roofline_data["baseline_points"].append({
                        "name": bench.get("name", ""),
                        "intensity": ai_estimate or 0.5,
                        "performance_tflops": bench.get("baseline_tflops", 0),
                    })
                    roofline_data["optimized_points"].append({
                        "name": bench.get("name", ""),
                        "intensity": ai_estimate or 0.5,
                        "performance_tflops": bench.get("optimized_tflops", 0),
                        "speedup": speedup,
                    })

                    roofline_data["benchmark_details"].append({
                        "name": bench.get("name", ""),
                        "chapter": bench.get("chapter", ""),
                        "arithmetic_intensity": ai_estimate,
                        "baseline_gflops": bench.get("baseline_tflops", 0) * 1000,
                        "optimized_gflops": bench.get("optimized_tflops", 0) * 1000,
                        "speedup": speedup,
                    })
            roofline_data["has_real_data"] = len(roofline_data["baseline_points"]) > 0
        except Exception:
            pass

        return roofline_data

    # ------------------------------------------------------------------
    # Benchmark discovery
    # ------------------------------------------------------------------
    def _relative_to_bench_root(self, path: Path) -> str:
        """Return a stable relative path against the configured benchmark roots."""
        for root in self.bench_roots:
            try:
                return str(path.resolve().relative_to(root.resolve()))
            except Exception:
                continue
        return str(path)

    def get_available_benchmarks(self) -> dict:
        available = {
            "chapters": [],
            "labs": [],
            "total_chapters": 0,
            "total_labs": 0,
            "total_benchmarks": 0,
        }

        for dir_path in discover_all_chapters(self.bench_root, bench_roots=self.bench_roots):
            rel = self._relative_to_bench_root(dir_path)
            dir_type = "lab" if rel.startswith("labs/") else "chapter"
            info = self._scan_directory(dir_path, dir_type)
            if not info["benchmarks"]:
                continue
            if dir_type == "lab":
                available["labs"].append(info)
            else:
                available["chapters"].append(info)

        available["total_chapters"] = len(available["chapters"])
        available["total_labs"] = len(available["labs"])
        available["total_benchmarks"] = sum(
            len(ch["benchmarks"]) for ch in available["chapters"]
        ) + sum(len(lab["benchmarks"]) for lab in available["labs"])

        return available

    def _scan_directory(self, directory: Path, dir_type: str) -> dict:
        info = {
            "name": directory.name,
            "path": self._relative_to_bench_root(directory),
            "type": dir_type,
            "benchmarks": [],
            "has_expectations": False,
            "has_profiles": False,
        }

        baseline_files = list(directory.glob("baseline_*.py")) + list(directory.glob("baseline_*.cu"))
        for baseline in baseline_files:
            name = baseline.stem.replace("baseline_", "")
            file_type = "python" if baseline.suffix == ".py" else "cuda"
            optimized_files = list(directory.glob(f"optimized_{name}*.py")) + list(directory.glob(f"optimized_{name}*.cu"))
            benchmark_info = {
                "name": name,
                "type": file_type,
                "baseline_file": baseline.name,
                "optimized_files": [f.name for f in optimized_files],
                "optimization_count": len(optimized_files),
            }
            info["benchmarks"].append(benchmark_info)

        info["has_expectations"] = any(directory.glob("expectations_*.json"))
        rel_path = Path(self._relative_to_bench_root(directory))
        profile_dir = self.bench_root / "benchmark_profiles" / rel_path
        info["has_profiles"] = profile_dir.exists() and any(profile_dir.iterdir()) if profile_dir.exists() else False
        return info

    # ------------------------------------------------------------------
    # Historical performance tracking
    # ------------------------------------------------------------------
    def _list_result_files(self) -> List[Path]:
        """Find all benchmark result files across artifacts and root."""
        paths: List[Path] = []
        if self.data_file:
            candidate = Path(self.data_file)
            if candidate.exists():
                paths.append(candidate)

        for root in self.bench_roots:
            artifacts_dir = root / "artifacts"
            if artifacts_dir.exists():
                paths.extend(sorted(artifacts_dir.rglob("benchmark_test_results.json")))

            root_default = root / "benchmark_test_results.json"
            if root_default.exists():
                paths.append(root_default)

        # Deduplicate while preserving order
        seen = set()
        unique_paths: List[Path] = []
        for path in paths:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_paths.append(path)

        unique_paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
        try:
            limit = int(os.environ.get("DASH_HISTORY_MAX_FILES", "200"))
        except ValueError:
            limit = 200
        if limit > 0 and len(unique_paths) > limit:
            unique_paths = unique_paths[-limit:]
        return unique_paths

    def _summarize_run_file(self, path: Path) -> Optional[dict]:
        """Load a single benchmark result file and compute summary metrics."""
        try:
            data = load_benchmark_results(path)
        except Exception:
            return None

        summary = data.get("summary", {}) or {}
        benchmarks = data.get("benchmarks", [])

        timestamp = data.get("timestamp")
        if not timestamp:
            try:
                timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            except Exception:
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        date = timestamp.split("T")[0] if "T" in timestamp else timestamp.split(" ")[0]

        if not summary.get("avg_speedup"):
            speedups = [b.get("speedup", 0) or 0 for b in benchmarks if b.get("speedup") is not None]
            summary["avg_speedup"] = sum(speedups) / len(speedups) if speedups else 0
            summary["max_speedup"] = max(speedups) if speedups else 0

        benchmark_count = summary.get("total_benchmarks") or len(benchmarks)
        successful = summary.get("successful")
        if successful is None:
            successful = sum(1 for b in benchmarks if str(b.get("status", "")).lower() == "succeeded")
        failed = summary.get("failed")
        if failed is None and benchmark_count is not None and successful is not None:
            failed = max(0, benchmark_count - successful)

        return {
            "date": date,
            "timestamp": timestamp,
            "benchmark_count": benchmark_count,
            "avg_speedup": float(summary.get("avg_speedup", 0) or 0),
            "max_speedup": float(summary.get("max_speedup", 0) or 0),
            "successful": successful if successful is not None else 0,
            "failed": failed if failed is not None else 0,
            "source": str(path),
        }

    def get_history_runs(self) -> dict:
        """Return a chronological list of benchmark runs with summary stats."""
        files = self._list_result_files()
        key = (
            tuple(str(r.resolve()) for r in self.bench_roots),
            tuple((str(p), p.stat().st_mtime) for p in files if p.exists()),
        )

        if _HISTORY_CACHE.get("key") == key and _HISTORY_CACHE.get("runs"):
            return _HISTORY_CACHE["runs"]

        runs: List[dict] = []
        for path in files:
            run = self._summarize_run_file(path)
            if run:
                runs.append(run)

        runs.sort(key=lambda r: r.get("timestamp") or r.get("date") or "", reverse=True)

        result = {
            "total_runs": len(runs),
            "latest": runs[0]["date"] if runs else None,
            "runs": runs,
        }
        _HISTORY_CACHE["key"] = key
        _HISTORY_CACHE["runs"] = result
        _HISTORY_CACHE["trends"] = None
        return result

    def get_historical_runs(self) -> dict:
        """Alias for compatibility with older handlers."""
        return self.get_history_runs()

    def get_performance_trends(self, runs_data: Optional[dict] = None) -> dict:
        """Compute trend data (avg/max speedup) across historical runs."""
        if runs_data is None:
            runs_data = self.get_history_runs()
        if runs_data is _HISTORY_CACHE.get("runs") and _HISTORY_CACHE.get("trends"):
            return _HISTORY_CACHE["trends"]

        runs = runs_data.get("runs", [])
        history = sorted(runs, key=lambda r: r.get("timestamp") or r.get("date") or "")

        trend_points = [
            {
                "date": r.get("date"),
                "avg_speedup": r.get("avg_speedup", 0) or 0,
                "max_speedup": r.get("max_speedup", 0) or 0,
                "benchmark_count": r.get("benchmark_count", 0) or 0,
            }
            for r in history
        ]

        best_ever = {}
        if trend_points:
            best_entry = max(
                trend_points,
                key=lambda x: x.get("max_speedup", 0) or x.get("avg_speedup", 0),
            )
            best_ever = {
                "date": best_entry.get("date"),
                "speedup": best_entry.get("max_speedup") or best_entry.get("avg_speedup") or 0,
            }

        improvements: List[dict] = []
        regressions: List[dict] = []
        prev_avg: Optional[float] = None
        for point in trend_points:
            current = point.get("avg_speedup", 0) or 0
            if prev_avg is not None:
                delta = current - prev_avg
                if delta > 0:
                    improvements.append({"date": point.get("date"), "delta": delta})
                elif delta < 0:
                    regressions.append({"date": point.get("date"), "delta": delta})
            prev_avg = current

        avg_speedup = (
            sum(p.get("avg_speedup", 0) or 0 for p in trend_points) / len(trend_points)
            if trend_points else 0
        )

        result = {
            "by_date": trend_points,
            "history": trend_points,
            "best_ever": best_ever,
            "avg_speedup": avg_speedup,
            "run_count": runs_data.get("total_runs", len(trend_points)),
            "improvements": improvements,
            "regressions": regressions,
        }
        if runs_data is _HISTORY_CACHE.get("runs"):
            _HISTORY_CACHE["trends"] = result
        return result

    def get_history_summary(self) -> dict:
        """Combined summary used by the dashboard history tab."""
        runs = self.get_history_runs()
        trends = self.get_performance_trends(runs)
        run_list = runs.get("runs", [])
        latest_date = run_list[0].get("date") if run_list else None
        return {
            "total": runs.get("total_runs", 0),
            "latest": latest_date,
            "avg_speedup": trends.get("avg_speedup", 0),
            "runs": run_list,
            "trends": trends,
        }

    # ------------------------------------------------------------------
    # GPU + software info
    # ------------------------------------------------------------------
    def get_gpu_info(self) -> dict:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,temperature.memory,power.draw,power.limit,memory.used,memory.total,utilization.gpu,utilization.memory,clocks.current.graphics,clocks.current.memory,fan.speed,persistence_mode,pstate",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                hbm_temp = None
                try:
                    if parts[2] and parts[2] != "[N/A]":
                        hbm_temp = float(parts[2])
                except (ValueError, IndexError):
                    pass
                fan_speed = None
                try:
                    if len(parts) > 11 and parts[11] and parts[11] != "[N/A]":
                        fan_speed = int(float(parts[11]))
                except (ValueError, IndexError):
                    pass
                ecc_mode = None
                try:
                    if len(parts) > 14 and parts[14].strip() not in ["[N/A]", "N/A", ""]:
                        ecc_mode = parts[14].strip() == "Enabled"
                except (ValueError, IndexError):
                    pass

                return {
                    "name": parts[0],
                    "temperature": float(parts[1]),
                    "temperature_hbm": hbm_temp,
                    "power": float(parts[3]),
                    "power_limit": float(parts[4]) if parts[4] != "[N/A]" else None,
                    "memory_used": float(parts[5]),
                    "memory_total": float(parts[6]),
                    "utilization": float(parts[7]),
                    "utilization_memory": float(parts[8]) if parts[8] != "[N/A]" else None,
                    "clock_graphics": int(float(parts[9])) if len(parts) > 9 else None,
                    "clock_memory": int(float(parts[10])) if len(parts) > 10 else None,
                    "fan_speed": fan_speed,
                    "persistence_mode": parts[12].strip() == "Enabled" if len(parts) > 12 else None,
                    "pstate": parts[13].strip() if len(parts) > 13 else None,
                    "ecc_mode": ecc_mode,
                    "live": True,
                }
        except Exception:
            pass
        return {
            "name": "GPU Not Detected",
            "temperature": None,
            "temperature_hbm": None,
            "power": None,
            "power_limit": None,
            "memory_used": None,
            "memory_total": None,
            "utilization": None,
            "utilization_memory": None,
            "clock_graphics": None,
            "clock_memory": None,
            "fan_speed": None,
            "persistence_mode": None,
            "pstate": None,
            "ecc_mode": None,
            "live": False,
            "error": "nvidia-smi failed or no GPU available",
        }

    def get_gpu_topology(self) -> dict:
        topology = {
            "gpu_count": 0,
            "gpus": [],
            "topology_matrix": [],
            "nvlink_available": False,
            "p2p_matrix": [],
        }
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        topology["gpus"].append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "uuid": parts[2] if len(parts) > 2 else "",
                                "pci_bus": parts[3] if len(parts) > 3 else "",
                            }
                        )
                topology["gpu_count"] = len(topology["gpus"])

            result = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                topology["topology_raw"] = result.stdout
                for line in lines:
                    if "GPU" in line and ("NV" in line or "PIX" in line or "PHB" in line or "SYS" in line):
                        topology["nvlink_available"] = "NV" in line
                        parts = line.split()
                        row = []
                        for p in parts[1:]:
                            if p in [
                                "X",
                                "NV1",
                                "NV2",
                                "NV3",
                                "NV4",
                                "NV5",
                                "NV6",
                                "NV7",
                                "NV8",
                                "NV9",
                                "NV10",
                                "NV11",
                                "NV12",
                                "NV18",
                                "PIX",
                                "PHB",
                                "SYS",
                                "NODE",
                            ]:
                                row.append(p)
                        if row:
                            topology["topology_matrix"].append(row)

            try:
                import torch

                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    p2p_matrix: List[List[str]] = []
                    for i in range(min(gpu_count, 8)):
                        row = []
                        for j in range(min(gpu_count, 8)):
                            if i == j:
                                row.append("self")
                            else:
                                try:
                                    can_access = torch.cuda.can_device_access_peer(i, j)
                                    row.append("yes" if can_access else "no")
                                except Exception:
                                    row.append("?")
                        p2p_matrix.append(row)
                    topology["p2p_matrix"] = p2p_matrix
            except Exception:
                pass
        except Exception as e:
            topology["error"] = str(e)

        return topology

    def get_nvlink_status(self) -> dict:
        nvlink = {"available": False, "links_per_gpu": {}, "total_bandwidth_gbs": 0, "link_details": []}
        try:
            result = subprocess.run(["nvidia-smi", "nvlink", "--status"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                nvlink["available"] = True
                nvlink["raw_output"] = result.stdout
                current_gpu = None
                link_count = 0
                for line in result.stdout.split("\n"):
                    if "GPU" in line and ":" in line:
                        if current_gpu is not None:
                            nvlink["links_per_gpu"][current_gpu] = link_count
                        match = re.search(r"GPU (\\d+)", line)
                        if match:
                            current_gpu = int(match.group(1))
                            link_count = 0
                    elif "Link" in line and "GB/s" in line:
                        link_count += 1
                        bw_match = re.search(r"(\\d+)\\s*GB/s", line)
                        if bw_match:
                            nvlink["link_details"].append({"gpu": current_gpu, "bandwidth_gbs": int(bw_match.group(1))})
                if current_gpu is not None:
                    nvlink["links_per_gpu"][current_gpu] = link_count
                nvlink["total_bandwidth_gbs"] = sum(l.get("bandwidth_gbs", 0) for l in nvlink["link_details"])
        except Exception:
            pass
        return nvlink

    def get_software_info(self) -> dict:
        info: Dict[str, Any] = {
            "pytorch": None,
            "cuda_runtime": None,
            "cuda_driver": None,
            "triton": None,
            "cudnn": None,
            "cublas": None,
            "nccl": None,
            "flash_attn": None,
            "transformer_engine": None,
            "xformers": None,
            "python": None,
            "compute_capability": None,
            "architecture": None,
            "gpu_count": None,
            "torch_compile_backend": None,
        }

        try:
            import torch

            info["pytorch"] = torch.__version__
            info["cuda_runtime"] = torch.version.cuda
            if torch.cuda.is_available():
                device = torch.device("cuda")
                props = torch.cuda.get_device_properties(device)
                info["compute_capability"] = f"{props.major}.{props.minor}"
                info["architecture"] = props.name
                info["gpu_count"] = torch.cuda.device_count()
                info["torch_compile_backend"] = os.environ.get("TORCH_COMPILE_BACKEND")
        except Exception:
            pass

        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                lines = result.stdout.splitlines()
                if lines:
                    info["cuda_driver"] = lines[2].split()[2] if len(lines) > 2 else None
        except Exception:
            pass

        try:
            import importlib

            for pkg in ["triton", "transformer_engine", "flash_attn", "xformers"]:
                try:
                    module = importlib.import_module(pkg)
                    info[pkg] = getattr(module, "__version__", None)
                except Exception:
                    continue
        except Exception:
            pass

        try:
            import sys
            info["python"] = sys.version.split()[0]
        except Exception:
            pass

        return info

    def get_dependency_health(self) -> dict:
        project_root = CODE_ROOT
        third_party = project_root / "third_party"
        result = {
            "status": "ok",
            "issues": [],
            "warnings": [],
            "cutlass": {"version": None, "path": None, "sm100_headers": False},
            "transformer_engine": {"version": None, "cutlass_symlink": False, "cutlass_symlink_target": None, "cutlass_sm100_headers": False},
            "nvidia_cutlass_dsl": {"version": None, "path": None},
        }

        # CUTLASS version check
        cutlass_path = third_party / "cutlass"
        version_h = cutlass_path / "include" / "cutlass" / "version.h"
        if version_h.exists():
            result["cutlass"]["path"] = str(cutlass_path)
            try:
                content = version_h.read_text()
                major = minor = patch = 0
                for line in content.splitlines():
                    if "CUTLASS_VERSION_MAJOR" in line:
                        major = int(re.findall(r"\\d+", line)[0])
                    if "CUTLASS_VERSION_MINOR" in line:
                        minor = int(re.findall(r"\\d+", line)[0])
                    if "CUTLASS_VERSION_PATCH" in line:
                        patch = int(re.findall(r"\\d+", line)[0])
                result["cutlass"]["version"] = f"{major}.{minor}.{patch}"
            except Exception:
                pass

            # SM100 headers check
            sm100_header = cutlass_path / "include" / "cutlass" / "arch" / "sm100_smem_selector.h"
            result["cutlass"]["sm100_headers"] = sm100_header.exists()

        # Transformer Engine
        try:
            import transformer_engine

            te_path = Path(transformer_engine.__file__).resolve().parent
            result["transformer_engine"]["version"] = getattr(transformer_engine, "__version__", None)
            cutlass_link = te_path / "csrc" / "cutlass"
            if cutlass_link.exists():
                result["transformer_engine"]["cutlass_symlink"] = cutlass_link.is_symlink()
                try:
                    result["transformer_engine"]["cutlass_symlink_target"] = str(cutlass_link.resolve())
                except Exception:
                    pass
                sm100_header_te = cutlass_link / "include" / "cutlass" / "arch" / "sm100_smem_selector.h"
                result["transformer_engine"]["cutlass_sm100_headers"] = sm100_header_te.exists()
        except Exception:
            result["warnings"].append("transformer_engine not installed")

        # NVIDIA Cutlass DSL (optional)
        cutlass_dsl = third_party / "nvidia-cutlass-dsl"
        if cutlass_dsl.exists():
            result["nvidia_cutlass_dsl"]["path"] = str(cutlass_dsl)
            version_file = cutlass_dsl / "VERSION.txt"
            if version_file.exists():
                result["nvidia_cutlass_dsl"]["version"] = version_file.read_text().strip()

        return result

    def check_dependency_updates(self) -> dict:
        updates = {"outdated": [], "errors": []}
        try:
            result = subprocess.run(
                [os.environ.get("PYTHON_BIN", "python"), "-m", "pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0:
                try:
                    updates["outdated"] = json.loads(result.stdout)
                except Exception:
                    updates["errors"].append("Failed to parse pip output")
            else:
                updates["errors"].append(result.stderr.strip())
        except Exception as e:
            updates["errors"].append(str(e))
        return updates

    def get_full_system_context(self) -> dict:
        context = {
            "gpu_info": self.get_gpu_info(),
            "software_info": self.get_software_info(),
            "dependency_health": self.get_dependency_health(),
            "gpu_topology": self.get_gpu_topology(),
        }
        try:
            import torch

            context["cuda_available"] = torch.cuda.is_available()
        except Exception:
            context["cuda_available"] = False
        return context

    # ------------------------------------------------------------------
    # Advanced analysis + helper methods
    # ------------------------------------------------------------------
    def list_benchmark_targets(self) -> dict:
        """List all available benchmark targets in chapter:example format."""
        targets = []

        for dir_path in discover_all_chapters(self.bench_root, bench_roots=self.bench_roots):
            chapter = self._relative_to_bench_root(dir_path)
            for baseline in dir_path.glob("baseline_*.py"):
                name = baseline.stem.replace("baseline_", "")
                targets.append(f"{chapter}:{name}")
            for baseline in dir_path.glob("baseline_*.cu"):
                name = baseline.stem.replace("baseline_", "")
                targets.append(f"{chapter}:{name}")

        unique = sorted(set(targets))
        return {"targets": unique, "count": len(unique)}

    def get_bank_conflicts(self, stride: int = 1, element_size: int = 4) -> dict:
        """
        Estimate shared-memory bank conflicts for a warp given an access pattern.

        We assume a typical 32-bank layout with 4-byte stride per bank. The
        computation maps the first warp's accesses and reports how many threads
        collide on each bank.
        """
        stride_bytes = max(1, stride) * max(1, element_size)
        banks = [((t * stride_bytes) // 4) % 32 for t in range(32)]
        counts = Counter(banks)
        unique_banks = len(counts)
        worst_conflict = max(counts.values())
        conflict_ratio = 1 - unique_banks / 32

        severity = "low"
        if worst_conflict > 4:
            severity = "medium"
        if worst_conflict > 8:
            severity = "high"

        recommendations = [
            "Pad shared-memory leading dimension by +1 to break stride aliasing",
            "Use 8-byte banking mode for fp64/int64 data" if element_size >= 8 else "Keep element size at 4 bytes where possible",
            "Restructure accesses so consecutive threads hit consecutive addresses",
        ]
        if stride_bytes % 128 == 0:
            recommendations.append("Stride maps every thread to the same bank; add padding or adjust tile shape")

        return {
            "stride_elements": stride,
            "element_size_bytes": element_size,
            "bank_mapping": banks,
            "bank_histogram": dict(sorted(counts.items())),
            "unique_banks_touched": unique_banks,
            "worst_conflicts_per_bank": worst_conflict,
            "conflict_ratio": round(conflict_ratio, 3),
            "severity": severity,
            "recommendations": recommendations,
        }

    def get_warp_divergence(self, code: str = "") -> dict:
        """Lightweight static analysis for warp divergence risks."""
        snippet = code or "// Provide kernel code to analyze branching patterns"
        lowered = snippet.lower()
        branch_markers = ["if", "else", "switch", "? :", "while", "for", "return"]
        branch_count = sum(lowered.count(tok.replace(" ", "")) for tok in branch_markers)
        loop_count = lowered.count("for") + lowered.count("while")
        uses_predication = "__shfl" in snippet or "__ballot" in snippet

        risk_score = branch_count + loop_count * 0.5
        severity = "low"
        if risk_score > 6:
            severity = "high"
        elif risk_score > 3:
            severity = "medium"

        recommendations = [
            "Favor predication (`?:`) over divergent branches where possible",
            "Hoist invariant branches out of the warp loop",
            "Group threads with similar control flow (data reordering) to reduce divergence",
        ]
        if not uses_predication:
            recommendations.append("Use warp intrinsics (__shfl_sync, __ballot_sync) to avoid branches")

        return {
            "input_size": len(snippet),
            "branch_markers": branch_markers,
            "branch_count": branch_count,
            "loop_count": loop_count,
            "predication_detected": uses_predication,
            "divergence_severity": severity,
            "recommendations": recommendations,
        }

    def get_memory_access_patterns(self, stride: int = 1, element_size: int = 4) -> dict:
        """
        Analyze memory coalescing for a single warp.

        We map 32 thread accesses into 128-byte segments (Ampere+/Hopper cache
        line size) and report how many memory transactions are required.
        """
        stride_bytes = max(1, stride) * max(1, element_size)
        addresses = [t * stride_bytes for t in range(32)]
        segments = [addr // 128 for addr in addresses]
        transactions = len(set(segments))
        efficiency = 100 / transactions if transactions else 100

        contiguous = transactions == 1
        misaligned = (addresses[0] % 128) != 0

        recommendations = []
        if not contiguous:
            recommendations.append("Realign data so a warp touches a single 128B segment")
            recommendations.append("Use structure-of-arrays layout for stride-based access")
        if misaligned:
            recommendations.append("Align starting address to 128 bytes to maximize coalescing")
        if stride_bytes > 128:
            recommendations.append("Large stride causes scattered loads; consider tiling or transpose")

        return {
            "stride_elements": stride,
            "element_size_bytes": element_size,
            "addresses": addresses,
            "segments": segments,
            "transactions": transactions,
            "coalescing_efficiency_pct": round(efficiency, 1),
            "contiguous": contiguous,
            "misaligned": misaligned,
            "recommendations": recommendations or ["Pattern already coalesced"],
        }

    def get_container_limits(self) -> dict:
        """Inspect container/cgroup limits with concrete values."""
        in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv") or bool(
            os.environ.get("KUBERNETES_SERVICE_HOST")
        )
        container_type = (
            "docker" if os.path.exists("/.dockerenv") else
            "podman" if os.path.exists("/run/.containerenv") else
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else
            "baremetal"
        )

        cgroup_v2 = os.path.exists("/sys/fs/cgroup/cgroup.controllers")
        cpu_limit = None
        mem_limit_gb = None
        errors: List[str] = []

        try:
            if cgroup_v2:
                cpu_max_path = "/sys/fs/cgroup/cpu.max"
                mem_max_path = "/sys/fs/cgroup/memory.max"
                if os.path.exists(cpu_max_path):
                    cpu_limit = Path(cpu_max_path).read_text().strip()
                if os.path.exists(mem_max_path):
                    raw = Path(mem_max_path).read_text().strip()
                    if raw != "max":
                        mem_limit_gb = int(raw) / (1024 ** 3)
            else:
                quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
                period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
                if Path(quota_path).exists() and Path(period_path).exists():
                    quota = int(Path(quota_path).read_text().strip())
                    period = int(Path(period_path).read_text().strip())
                    if quota > 0 and period > 0:
                        cpu_limit = round(quota / period, 2)
                mem_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
                if Path(mem_path).exists():
                    raw = Path(mem_path).read_text().strip()
                    if raw.isdigit():
                        mem_limit_gb = int(raw) / (1024 ** 3)
        except Exception as exc:
            errors.append(str(exc))

        recommendations = []
        if in_container and cpu_limit and cpu_limit != "max":
            recommendations.append("Increase CPU quota for dataloader/compilation steps if throttling is observed")
        if in_container and mem_limit_gb and mem_limit_gb < 32:
            recommendations.append("Memory limit is low; increase container memory to avoid OOM during compilation")

        return {
            "in_container": in_container,
            "container_type": container_type,
            "cgroup_version": "v2" if cgroup_v2 else "v1",
            "cpu_limit": cpu_limit,
            "memory_limit_gb": mem_limit_gb,
            "errors": errors,
            "recommendations": recommendations,
        }

    def get_cpu_memory_analysis(self) -> dict:
        """Inspect CPU topology, caches, memory, NUMA, and TLB hints."""
        cpu_info: Dict[str, Any] = {"logical_cpus": os.cpu_count(), "model": None}
        caches: List[Dict[str, Any]] = []
        memory: Dict[str, Any] = {}
        numa_nodes: List[Dict[str, Any]] = []
        tlb_info: List[str] = []
        errors: List[str] = []

        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                raw = handle.read()
            for line in raw.splitlines():
                if "model name" in line:
                    cpu_info["model"] = line.split(":", 1)[1].strip()
                    break
        except Exception as exc:
            errors.append(f"cpuinfo: {exc}")

        try:
            cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
            if cache_root.exists():
                for idx in cache_root.iterdir():
                    try:
                        level = (idx / "level").read_text().strip()
                        ctype = (idx / "type").read_text().strip()
                        size = (idx / "size").read_text().strip()
                        caches.append({"level": f"L{level} {ctype}", "size": size})
                    except Exception:
                        continue
        except Exception as exc:
            errors.append(f"cache: {exc}")

        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()[:20]
            for line in lines:
                if ":" in line:
                    key, val = line.split(":", 1)
                    if any(k in key for k in ["MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal"]):
                        memory[key.strip()] = val.strip()
        except Exception as exc:
            errors.append(f"meminfo: {exc}")

        try:
            result = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "node" in line and "cpus" in line:
                        numa_nodes.append({"raw": line.strip()})
        except Exception:
            # numactl optional
            pass

        try:
            result = subprocess.run(["cpuid", "-1"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "TLB" in line or "tlb" in line:
                        tlb_info.append(line.strip())
        except Exception:
            pass

        recommendations = [
            "Use numactl --localalloc for NUMA-aware allocation",
            "Enable huge pages for large batches: echo 1024 > /proc/sys/vm/nr_hugepages",
            "Pin dataloader workers to the local NUMA node for each GPU",
        ]

        return {
            "cpu": cpu_info,
            "cache_hierarchy": caches,
            "memory": memory,
            "numa": numa_nodes,
            "tlb": tlb_info,
            "recommendations": recommendations,
            "errors": errors,
        }

    def get_system_parameters(self) -> dict:
        """Inspect kernel parameters that commonly affect performance."""
        params = [
            ("/proc/sys/vm/swappiness", "Swappiness", "Lower favors RAM over swap", "10"),
            ("/proc/sys/vm/dirty_ratio", "Dirty Ratio", "Max % of RAM for dirty pages", "20"),
            ("/proc/sys/vm/dirty_background_ratio", "Dirty Background", "Start background writeback", "5"),
            ("/proc/sys/kernel/sched_migration_cost_ns", "Sched Migration Cost", "Task migration threshold", "500000"),
            ("/proc/sys/kernel/numa_balancing", "NUMA Balancing", "Automatic NUMA page migration", "1"),
            ("/proc/sys/net/core/rmem_max", "Net RX Buffer Max", "Max receive buffer size", "16777216"),
            ("/proc/sys/net/core/wmem_max", "Net TX Buffer Max", "Max send buffer size", "16777216"),
            ("/proc/sys/fs/file-max", "File Descriptors Max", "System-wide FD limit", "1000000"),
            ("/proc/sys/kernel/perf_event_paranoid", "Perf Paranoid", "Performance counter access", "1"),
        ]

        readings = []
        errors: List[str] = []
        for path, name, desc, recommended in params:
            current = None
            try:
                if Path(path).exists():
                    current = Path(path).read_text().strip()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
            readings.append(
                {
                    "name": name,
                    "path": path,
                    "current": current,
                    "recommended": recommended,
                    "description": desc,
                    "needs_tuning": current is not None and str(current) != str(recommended),
                }
            )

        quick_tune = [
            "sudo sysctl -w vm.swappiness=10",
            "sudo sysctl -w vm.dirty_ratio=20",
            "sudo sysctl -w kernel.numa_balancing=0  # if manually pinning NUMA",
            "export CUDA_DEVICE_MAX_CONNECTIONS=8",
            "export NCCL_NVLS_ENABLE=1",
        ]

        return {
            "parameters": readings,
            "errors": errors,
            "quick_tune_commands": quick_tune,
        }

    def get_comm_overlap_analysis(self, model: str = "llama-3.1-70b") -> dict:
        """Estimate communication/computation overlap opportunities."""
        try:
            from core.optimization.parallelism_planner.distributed_training import CommunicationOverlapAnalyzer
            from core.optimization.parallelism_planner.model_analyzer import ModelAnalyzer

            analyzer = ModelAnalyzer()
            model_name = model or "llama-3.1-70b"
            if model_name.lower() == "default":
                model_name = "llama-3.1-70b"

            fallback_warning = None
            try:
                arch = analyzer.analyze(model_name)
            except Exception as exc:
                # Avoid hard failures when HF auth/config is unavailable
                fallback_warning = f"Falling back to llama-3.1-70b preset: {exc}"
                arch = analyzer.analyze("llama-3.1-70b")
                model_name = "llama-3.1-70b"

            params_b = getattr(arch, "total_params_billion", None) or getattr(arch, "active_params_billion", None)
            params_b = float(params_b) if params_b else 70.0

            topo = self.get_gpu_topology()
            gpu_count = topo.get("gpu_count") or 1
            tp_size = max(1, min(gpu_count, math.ceil(params_b / 35)))
            pp_size = 2 if params_b > 40 and gpu_count >= tp_size * 2 else 1
            dp_size = max(1, gpu_count // max(tp_size * pp_size, 1))
            batch_size = 8
            seq_length = getattr(arch, "max_position_embeddings", 4096) or 4096

            overlap = CommunicationOverlapAnalyzer().analyze(
                params_b,
                tp_size=tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                batch_size=batch_size,
                seq_length=seq_length,
            )
            result = {
                "success": True,
                "model": model_name,
                "inputs": {
                    "params_b": params_b,
                    "tp": tp_size,
                    "pp": pp_size,
                    "dp": dp_size,
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "gpus": gpu_count,
                },
                "overlap_analysis": overlap,
            }
            if fallback_warning:
                result["warning"] = fallback_warning
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def generate_slurm_script(self, model: str, nodes: int, gpus: int, framework: str) -> dict:
        """Generate a SLURM job script using the cluster config helper."""
        try:
            from core.optimization.parallelism_planner.extras import JobScriptGenerator

            generator = JobScriptGenerator()
            launch_cmd = (
                f"torchrun --nnodes={nodes} --nproc_per_node={gpus} "
                f"train.py --model {model} --framework {framework}"
            )
            script = generator.generate_slurm(
                job_name=f"{model}-{framework}",
                num_nodes=nodes,
                gpus_per_node=gpus,
                time_hours=24,
                partition="gpu",
                script="train.py",
                launch_command=launch_cmd,
            )
            return {"success": True, "slurm_script": script}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_nccl_recommendations(self, nodes: int, gpus: int, diagnose: bool) -> dict:
        """Get NCCL tuning recommendations."""
        try:
            from dataclasses import asdict

            from core.optimization.parallelism_planner.distributed_training import NCCLTuningAdvisor, NCCLConfig

            advisor = NCCLTuningAdvisor()
            if diagnose:
                result = advisor.diagnose_issues()
                return {"success": True, "recommendations": result}

            config = advisor.get_optimal_config(num_nodes=nodes, gpus_per_node=gpus)
            if isinstance(config, NCCLConfig):
                result = asdict(config)
            else:
                result = config
            return {"success": True, "recommendations": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_parallelism_recommendations(self, params: Optional[dict] = None) -> dict:
        """Recommend TP/PP/DP settings using the Parallelism Advisor."""
        params = params or {}
        model = params.get("model", "llama-3.1-70b")
        batch_size = int(params.get("batch_size", 1))
        seq_length = int(params.get("seq_length", 2048))
        goal = params.get("goal", "throughput")
        is_training = bool(params.get("is_training", False))

        try:
            from core.optimization.parallelism_planner.advisor import ParallelismAdvisor

            advisor = ParallelismAdvisor(auto_detect_topology=True)
            result = advisor.recommend(
                model=model,
                batch_size=batch_size,
                seq_length=seq_length,
                goal=goal,
                is_training=is_training,
            )
            best = result.best_strategy.to_dict() if result.best_strategy else None
            recs = [rec.to_dict() for rec in result.recommendations]
            return {
                "success": True,
                "model": model,
                "goal": goal,
                "is_training": is_training,
                "recommendations": recs,
                "best": best,
                "topology": result.topology.to_dict() if result.topology else None,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_cloud_cost_estimate(self, params: dict) -> dict:
        """Estimate cloud GPU costs for running models."""
        model_params = params.get("params", 7e9)
        batch_size = params.get("batch_size", 32)
        tokens_per_request = params.get("tokens", 512)
        requests_per_day = params.get("requests_per_day", 10000)
        precision = params.get("precision", "fp16")

        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}.get(precision, 2)
        model_mem_gb = (model_params * bytes_per_param) / (1024 ** 3)

        gpu_pricing = [
            {"name": "NVIDIA T4", "vram": 16, "price": 0.35, "tflops": 65},
            {"name": "NVIDIA A10G", "vram": 24, "price": 1.00, "tflops": 125},
            {"name": "NVIDIA L4", "vram": 24, "price": 0.80, "tflops": 121},
            {"name": "NVIDIA A100 40GB", "vram": 40, "price": 3.50, "tflops": 312},
            {"name": "NVIDIA A100 80GB", "vram": 80, "price": 5.00, "tflops": 312},
            {"name": "NVIDIA H100 80GB", "vram": 80, "price": 8.50, "tflops": 990},
            {"name": "NVIDIA H100 SXM", "vram": 80, "price": 12.00, "tflops": 1980},
        ]

        estimates = []
        for gpu in gpu_pricing:
            if model_mem_gb > gpu["vram"] * 0.85:
                num_gpus = int(model_mem_gb / (gpu["vram"] * 0.7)) + 1
                if num_gpus > 8:
                    continue
            else:
                num_gpus = 1

            base_tokens_per_sec = (gpu["tflops"] / model_params * 1e9) * 50
            tokens_per_sec = base_tokens_per_sec * batch_size * 0.7

            total_tokens = requests_per_day * tokens_per_request
            hours_needed = (total_tokens / tokens_per_sec) / 3600 if tokens_per_sec else 0

            hourly_cost = gpu["price"] * num_gpus
            daily_cost = hourly_cost * max(hours_needed, 1)
            monthly_cost = daily_cost * 30
            monthly_24_7 = hourly_cost * 24 * 30

            estimates.append({
                "gpu": gpu["name"],
                "num_gpus": num_gpus,
                "vram_total": gpu["vram"] * num_gpus,
                "hourly_cost": round(hourly_cost, 2),
                "estimated_tokens_per_sec": round(tokens_per_sec, 0),
                "hours_for_daily_load": round(hours_needed, 2),
                "daily_cost": round(daily_cost, 2),
                "monthly_cost": round(monthly_cost, 2),
                "monthly_24_7_cost": round(monthly_24_7, 2),
            })

        return {
            "model_params": self._format_params(int(model_params)),
            "precision": precision.upper(),
            "batch_size": batch_size,
            "tokens_per_request": tokens_per_request,
            "requests_per_day": requests_per_day,
            "estimates": estimates,
        }

    def get_energy_analysis(self) -> dict:
        """Energy efficiency analysis combining measured data if available."""
        from core.analysis import power_efficiency_analyzer as pea

        artifacts_dir = self.bench_root / "artifacts"
        power_file = None
        throughput_file = None

        if artifacts_dir.exists():
            for candidate in artifacts_dir.rglob("*.json"):
                name = candidate.name.lower()
                if "power" in name and power_file is None:
                    power_file = candidate
                if "throughput" in name and throughput_file is None:
                    throughput_file = candidate
                if power_file and throughput_file:
                    break

        if not power_file or not throughput_file:
            return {"success": False, "error": "Missing power/throughput artifacts for energy analysis"}

        try:
            power_data = pea.load_power_metrics(power_file)
            throughput_data = pea.load_throughput_metrics(throughput_file)
            measured = pea.calculate_power_efficiency(power_data, throughput_data)
            return {
                "success": True,
                "source": "measured",
                "analysis": measured,
                "power_file": str(power_file),
                "throughput_file": str(throughput_file),
                "errors": [],
            }
        except Exception as exc:
            return {"success": False, "error": f"measurement: {exc}"}

    def get_quantization_comparison(self, params: dict) -> dict:
        """Compare batch size and memory needs across precisions."""
        gpu_info = self.get_gpu_info()
        vram_total_gb = (gpu_info.get("memory_total", 0) or 0) / 1024
        vram_free_gb = vram_total_gb - ((gpu_info.get("memory_used", 0) or 0) / 1024)

        model_params = params.get("params", 7e9)
        model_name = params.get("name", "Model")

        precisions = ["fp32", "fp16", "bf16", "int8", "int4"]
        comparison = []

        for precision in precisions:
            batch_info = self._calculate_batch_for_params(int(model_params), vram_free_gb, precision)
            comparison.append({
                "precision": precision.upper(),
                "weight_memory_gb": batch_info["weight_memory_gb"],
                "can_run": batch_info["can_run"],
                "inference_batch": batch_info["inference"]["recommended_batch_size"],
                "inference_max": batch_info["inference"]["max_batch_size"],
                "training_batch": batch_info["training"]["recommended_batch_size"],
                "training_max": batch_info["training"]["max_batch_size"],
            })

        return {
            "model_name": model_name,
            "params": model_params,
            "params_display": self._format_params(int(model_params)),
            "gpu": gpu_info.get("name", "Unknown"),
            "vram_free_gb": round(vram_free_gb, 1),
            "comparison": comparison,
        }

    def get_vllm_config(self, model: str, target: str, compare: bool) -> dict:
        """Generate vLLM configuration or compare inference engines."""
        try:
            from core.optimization.parallelism_planner.distributed_training import VLLMConfigGenerator

            generator = VLLMConfigGenerator()
            if compare:
                result = generator.compare_engines(model)
                return {"success": True, "engine_comparison": result}
            result = generator.generate(model, target=target)
            return {"success": True, "vllm_config": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_optimization_roi(self) -> dict:
        """Compute ROI for optimization techniques based on real benchmark data."""
        from core.optimization_reports import compute_roi

        data = self.load_benchmark_data()
        benchmarks = self._flatten_benchmarks(data)
        cost_data = self.analyzer.get_cost_analysis()
        return compute_roi(benchmarks, cost_data)

    def get_all_optimizations(self) -> dict:
        """List all available optimization techniques."""
        from core import optimization_stack

        return optimization_stack.get_all_optimizations()

    def predict_scaling(self, params: dict) -> dict:
        """Predict throughput scaling when increasing GPU count."""
        model_size = float(params.get("model_size", 7.0))
        target_gpus = int(params.get("gpus", 1))
        gpu_info = self.get_gpu_info()
        topo = self.get_gpu_topology()
        nvlink = topo.get("nvlink_available", False)

        gpu_name = (gpu_info.get("name") or "H100").lower()
        if "b200" in gpu_name:
            tflops = 2250
        elif "h100" in gpu_name or "h200" in gpu_name:
            tflops = 1979
        elif "a100" in gpu_name:
            tflops = 312
        elif "l40" in gpu_name:
            tflops = 181
        else:
            tflops = 100

        base_throughput = (tflops * 1e12 / (model_size * 1e9)) * 0.3
        efficiency = []
        for idx in range(1, target_gpus + 1):
            factor = 0.90 if nvlink else 0.82
            efficiency.append(factor ** max(idx - 1, 0))
        avg_eff = sum(efficiency) / len(efficiency) if efficiency else 1.0
        predicted_throughput = base_throughput * target_gpus * avg_eff

        return {
            "model_size_b": model_size,
            "gpu": gpu_info.get("name", "Unknown"),
            "target_gpus": target_gpus,
            "nvlink": nvlink,
            "base_tokens_per_sec_est": round(base_throughput, 2),
            "predicted_tokens_per_sec": round(predicted_throughput, 2),
            "scaling_efficiency_pct": round(avg_eff * 100, 1),
            "assumptions": [
                "Based on FP16 TFLOPS and simple communication overhead model",
                "NVLink improves efficiency vs PCIe",
            ],
        }

    def get_hardware_capabilities(self) -> dict:
        """Return probed hardware capability records."""
        try:
            from core.harness.hardware_capabilities import detect_capabilities, all_capability_records, format_capability_report

            cap = detect_capabilities()
            records = all_capability_records()
            return {
                "success": cap is not None,
                "device": cap.to_dict() if cap else None,
                "available_records": records,
                "report": format_capability_report(cap),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_full_system_analysis(self) -> dict:
        """Run combined system analysis with hardware + optimization context."""
        from core import optimization_stack

        return {
            "cpu_memory": self.get_cpu_memory_analysis(),
            "system_params": self.get_system_parameters(),
            "container": self.get_container_limits(),
            "optimizations_available": len(self.get_all_optimizations().get("optimizations", [])),
            "playbooks_available": optimization_stack.get_optimization_playbooks().get("count", 0),
            "recommendations": self._generate_comprehensive_recommendations(),
        }

    def get_data_loading_analysis(self) -> dict:
        """Provide dataloader recommendations; optimized for Grace-Blackwell if available."""
        try:
            from ch04.gb200_grace_numa_optimization import (
                optimize_data_loading_for_grace,
                detect_grace_cpu,
                setup_grace_affinity,
            )
            import torch

            gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
            loader_kwargs = optimize_data_loading_for_grace(gpu_id=gpu_id, verbose=False)
            cpu_info = detect_grace_cpu()
            cpu_affinity, numa_node = setup_grace_affinity(gpu_id, loader_kwargs["num_workers"], verbose=False)

            return {
                "success": True,
                "gpu_id": gpu_id,
                "dataloader_kwargs": loader_kwargs,
                "cpu": cpu_info,
                "numa_node": numa_node,
                "cpu_affinity": cpu_affinity,
                "notes": [
                    "Pinned memory and larger prefetch improve CPU→GPU transfer on GB200/GB300",
                    "Persistent workers avoid process churn for long training runs",
                ],
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Compose a concise list of actionable tuning tips."""
        recs: List[str] = []
        recs.extend(self.get_cpu_memory_analysis().get("recommendations", []))
        recs.extend(self.get_system_parameters().get("quick_tune_commands", []))
        recs.extend(self.get_container_limits().get("recommendations", []))

        sw_info = self.get_software_info()
        cc = sw_info.get("compute_capability")
        if cc:
            try:
                cc_val = float(cc)
            except Exception:
                cc_val = 0
            if cc_val >= 8.9:
                recs.append("FP8 supported; enable Transformer Engine for 2x throughput")
            if cc_val >= 9.0:
                recs.append("Use TMA/WGMMA kernels on Hopper-class GPUs")
            if cc_val >= 10.0:
                recs.append("Blackwell detected; enable FP4 and DSMEM optimizations")
        if sw_info.get("torch_compile_backend"):
            recs.append(f"torch.compile backend set to {sw_info['torch_compile_backend']}")

        return recs[:10]

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _format_params(self, params: int) -> str:
        """Format parameter count for display."""
        if params >= 1e12:
            return f"{params/1e12:.1f}T"
        if params >= 1e9:
            return f"{params/1e9:.1f}B"
        if params >= 1e6:
            return f"{params/1e6:.0f}M"
        if params >= 1e3:
            return f"{params/1e3:.0f}K"
        return str(params)

    def _format_number(self, num: int) -> str:
        """Format large numbers for display."""
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        if num >= 1e6:
            return f"{num/1e6:.1f}M"
        if num >= 1e3:
            return f"{num/1e3:.1f}K"
        return str(num)

    def _calculate_batch_for_params(self, params: int, vram_free_gb: float, precision: str = "fp16") -> dict:
        """Calculate batch size recommendations for a model with given param count."""
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
        }
        bytes_per_param = precision_bytes.get(precision, 2)

        weight_mem_gb = (params * bytes_per_param) / (1024 ** 3)
        inference_mem_gb = weight_mem_gb * 1.2
        training_mem_gb = weight_mem_gb * 3.5

        available_inference = max(0, vram_free_gb - inference_mem_gb - 1)
        available_training = max(0, vram_free_gb - training_mem_gb - 2)

        if params > 50e9:
            mem_per_sample_mb = 2000
        elif params > 10e9:
            mem_per_sample_mb = 800
        elif params > 3e9:
            mem_per_sample_mb = 400
        elif params > 1e9:
            mem_per_sample_mb = 200
        else:
            mem_per_sample_mb = 100

        max_batch_inference = int(available_inference * 1024 / mem_per_sample_mb) if available_inference > 0 else 0
        max_batch_training = int(available_training * 1024 / (mem_per_sample_mb * 2)) if available_training > 0 else 0

        def round_to_power_of_2(n: int) -> int:
            if n <= 0:
                return 0
            result = 1
            for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                if bs <= n:
                    result = bs
            return result

        recommended_inference = round_to_power_of_2(max_batch_inference)
        recommended_training = round_to_power_of_2(max_batch_training)

        util_inference = (recommended_inference * mem_per_sample_mb / 1024) / vram_free_gb * 100 if vram_free_gb > 0 else 0
        util_training = (recommended_training * mem_per_sample_mb * 2 / 1024) / vram_free_gb * 100 if vram_free_gb > 0 else 0

        can_run = weight_mem_gb < vram_free_gb

        return {
            "can_run": can_run,
            "weight_memory_gb": round(weight_mem_gb, 2),
            "precision": precision,
            "inference": {
                "recommended_batch_size": recommended_inference,
                "max_batch_size": max(1, max_batch_inference) if can_run else 0,
                "utilization_pct": round(min(util_inference, 100), 1),
            },
            "training": {
                "recommended_batch_size": recommended_training,
                "max_batch_size": max(0, max_batch_training),
                "utilization_pct": round(min(util_training, 100), 1),
            },
            "memory_per_sample_mb": mem_per_sample_mb,
            "suggestions": self._get_optimization_suggestions(params, vram_free_gb, precision, can_run),
        }

    def _get_optimization_suggestions(self, params: int, vram_gb: float, precision: str, can_run: bool) -> list:
        """Get optimization suggestions for running the model."""
        suggestions = []

        if not can_run:
            suggestions.append({
                "type": "critical",
                "text": "Model too large for available VRAM",
                "solutions": [
                    "Use quantization (INT8, INT4)",
                    "Use model parallelism across multiple GPUs",
                    "Try a smaller model variant",
                ],
            })

        if precision == "fp32":
            suggestions.append({
                "type": "optimization",
                "text": "Switch to FP16/BF16 for 2x memory savings",
                "benefit": "Double your batch size or fit larger models",
            })

        if params > 7e9 and precision in ["fp32", "fp16", "bf16"]:
            suggestions.append({
                "type": "optimization",
                "text": "Consider INT8 quantization for 4x memory savings",
                "benefit": "Minimal accuracy loss, major memory reduction",
            })

        if params > 20e9:
            suggestions.append({
                "type": "advanced",
                "text": "Use Flash Attention 2 for efficient attention",
                "benefit": "Reduce memory usage and improve speed",
            })
            suggestions.append({
                "type": "advanced",
                "text": "Enable gradient checkpointing for training",
                "benefit": "Trade compute for memory to train larger batches",
            })

        return suggestions

    def get_cost_savings_header(self, ops_per_day: int = 1_000_000) -> dict:
        """
        Calculate aggregate $ savings for the header display.
        
        This is the FRONT AND CENTER metric that translates performance
        gains into business value using public cloud GPU pricing.
        
        Args:
            ops_per_day: Assumed operations per day (default: 1M for enterprise scale)
        
        Returns:
            Dictionary with total savings and breakdown
        """
        from core.costs import GPU_PRICING, detect_gpu_pricing
        
        data = self.load_benchmark_data()
        benchmarks = self._flatten_benchmarks(data)
        gpu_info = self.get_gpu_info()
        
        # Detect GPU and get hourly rate
        gpu_name = gpu_info.get("name", "B200")
        hourly_rate = detect_gpu_pricing(gpu_name)
        
        # Find which GPU type we matched
        detected_gpu = "B200"  # default
        for gpu_type in GPU_PRICING:
            if gpu_type.lower() in gpu_name.lower():
                detected_gpu = gpu_type
                break
        
        total_baseline_time_ms = 0.0
        total_optimized_time_ms = 0.0
        total_time_saved_ms = 0.0
        successful_optimizations = 0
        savings_breakdown = []
        
        for b in benchmarks:
            if b.get("status") != "succeeded":
                continue
            
            speedup = b.get("speedup", 1.0)
            baseline_ms = b.get("baseline_time_ms", 0)
            optimized_ms = b.get("optimized_time_ms", baseline_ms)
            
            if baseline_ms <= 0 or optimized_ms <= 0 or speedup <= 1.0:
                continue
            
            time_saved_ms = baseline_ms - optimized_ms
            if time_saved_ms <= 0:
                continue
            
            successful_optimizations += 1
            total_baseline_time_ms += baseline_ms
            total_optimized_time_ms += optimized_ms
            total_time_saved_ms += time_saved_ms
            
            # Calculate savings for this benchmark
            # Time saved percentage = (baseline - optimized) / baseline
            time_saved_pct = (time_saved_ms / baseline_ms) * 100
            
            # Ops per hour at baseline rate
            baseline_ops_per_hour = 3_600_000 / baseline_ms if baseline_ms > 0 else 0
            optimized_ops_per_hour = 3_600_000 / optimized_ms if optimized_ms > 0 else 0
            
            # Cost per million ops
            baseline_cost_per_m = (hourly_rate / baseline_ops_per_hour) * 1_000_000 if baseline_ops_per_hour > 0 else 0
            optimized_cost_per_m = (hourly_rate / optimized_ops_per_hour) * 1_000_000 if optimized_ops_per_hour > 0 else 0
            savings_per_m = baseline_cost_per_m - optimized_cost_per_m
            
            # Daily savings based on ops_per_day
            daily_savings = (savings_per_m / 1_000_000) * ops_per_day
            monthly_savings = daily_savings * 30
            yearly_savings = daily_savings * 365
            
            savings_breakdown.append({
                "name": f"{b.get('chapter', 'unknown')}:{b.get('name', 'unknown')}",
                "speedup": round(speedup, 2),
                "time_saved_pct": round(time_saved_pct, 1),
                "monthly_savings_usd": round(monthly_savings, 2),
                "yearly_savings_usd": round(yearly_savings, 2),
            })
        
        # Sort by monthly savings (highest first)
        savings_breakdown.sort(key=lambda x: x["monthly_savings_usd"], reverse=True)
        
        # Calculate aggregate savings
        total_monthly_savings = sum(s["monthly_savings_usd"] for s in savings_breakdown)
        total_yearly_savings = sum(s["yearly_savings_usd"] for s in savings_breakdown)
        
        # Average time saved percentage across all optimizations
        avg_time_saved_pct = 0.0
        if total_baseline_time_ms > 0:
            avg_time_saved_pct = (total_time_saved_ms / total_baseline_time_ms) * 100
        
        # Average speedup
        avg_speedup = 0.0
        if successful_optimizations > 0:
            avg_speedup = sum(s["speedup"] for s in savings_breakdown) / successful_optimizations
        
        return {
            "total_monthly_savings_usd": round(total_monthly_savings, 2),
            "total_yearly_savings_usd": round(total_yearly_savings, 2),
            "avg_speedup": round(avg_speedup, 2),
            "avg_time_saved_pct": round(avg_time_saved_pct, 1),
            "successful_optimizations": successful_optimizations,
            "gpu": {
                "name": gpu_name,
                "type": detected_gpu,
                "hourly_rate_usd": hourly_rate,
            },
            "assumptions": {
                "ops_per_day": ops_per_day,
                "ops_per_month": ops_per_day * 30,
                "cloud_provider": "Public Cloud (avg)",
                "pricing_source": "2024 cloud GPU pricing",
            },
            "top_savers": savings_breakdown[:5],  # Top 5 for header tooltip
            "all_savers": savings_breakdown,
            "pricing_table": GPU_PRICING,
        }

    def get_cost_calculator(self) -> dict:
        """Get cost calculator data - wrapper for compatibility."""
        return self.analyzer.get_cost_analysis()

    def _flatten_benchmarks(self, raw_data: dict) -> List[Dict[str, Any]]:
        """Normalize benchmark structures between flat and chapter-grouped formats."""
        if "benchmarks" in raw_data:
            return raw_data.get("benchmarks", [])
        flattened: List[Dict[str, Any]] = []
        for chapter in raw_data.get("results", []):
            flattened.extend(chapter.get("benchmarks", []))
        return flattened

    # ------------------------------------------------------------------
    # Deep Profile Comparison (baseline vs optimized nsys/ncu)
    # ------------------------------------------------------------------
    def list_deep_profile_pairs(self) -> dict:
        """List all chapters/directories with baseline + optimized profile pairs."""
        from core.discovery import discover_all_chapters
        
        pairs = []
        
        # Check artifacts directory
        artifacts_dir = self.bench_root / "artifacts"
        if artifacts_dir.exists():
            for subdir in artifacts_dir.iterdir():
                if subdir.is_dir():
                    baseline_nsys = list(subdir.glob("*baseline*.nsys-rep"))
                    optimized_nsys = list(subdir.glob("*optimized*.nsys-rep"))
                    baseline_ncu = list(subdir.glob("*baseline*.ncu-rep"))
                    optimized_ncu = list(subdir.glob("*optimized*.ncu-rep"))
                    
                    if (baseline_nsys and optimized_nsys) or (baseline_ncu and optimized_ncu):
                        pairs.append({
                            "chapter": subdir.name,
                            "name": subdir.name,
                            "path": str(subdir),
                            "type": "artifact",
                            "has_nsys": bool(baseline_nsys and optimized_nsys),
                            "has_ncu": bool(baseline_ncu and optimized_ncu),
                            "baseline_nsys": [f.name for f in baseline_nsys],
                            "optimized_nsys": [f.name for f in optimized_nsys],
                            "baseline_ncu": [f.name for f in baseline_ncu],
                            "optimized_ncu": [f.name for f in optimized_ncu],
                        })
        
        # Check benchmark_profiles directory
        profiles_dir = self.bench_root / "benchmark_profiles"
        if profiles_dir.exists():
            for subdir in profiles_dir.iterdir():
                if subdir.is_dir():
                    baseline_nsys = list(subdir.glob("*baseline*.nsys-rep"))
                    optimized_nsys = list(subdir.glob("*optimized*.nsys-rep"))
                    
                    if baseline_nsys and optimized_nsys:
                        pairs.append({
                            "chapter": subdir.name,
                            "name": subdir.name,
                            "path": str(subdir),
                            "type": "profile",
                            "has_nsys": True,
                            "has_ncu": False,
                        })
        
        # Check chapter directories
        for dir_path in discover_all_chapters(self.bench_root, bench_roots=self.bench_roots):
            baseline_nsys = list(dir_path.glob("*baseline*.nsys-rep"))
            optimized_nsys = list(dir_path.glob("*optimized*.nsys-rep"))
            
            if baseline_nsys and optimized_nsys:
                rel = self._relative_to_bench_root(dir_path)
                pairs.append({
                    "chapter": dir_path.name,
                    "name": rel,
                    "path": str(dir_path),
                    "type": "chapter",
                    "has_nsys": True,
                    "has_ncu": bool(list(dir_path.glob("*baseline*.ncu-rep")) and list(dir_path.glob("*optimized*.ncu-rep"))),
                })
        
        return {"pairs": pairs, "count": len(pairs)}

    def compare_profiles(self, chapter: str) -> dict:
        """Compare baseline vs optimized profiles for a chapter.
        
        Integrates file-level comparison with metric-level analysis to provide:
        - Raw nsys/ncu comparisons
        - Structured metric diff (what improved, regressed, unchanged)
        - Bottleneck shift detection
        - Prioritized recommendations
        """
        from core import profile_insights
        
        # Find the chapter directory
        chapter_dir = self._find_profile_directory(chapter)
        
        if not chapter_dir:
            return {"error": f"Chapter not found: {chapter}", "chapter": chapter}
        
        # Get nsys comparison
        nsys_comparison = profile_insights.compare_nsys_files(chapter_dir)
        
        # Get ncu comparison
        ncu_comparison = profile_insights.compare_ncu_files(chapter_dir)
        
        # Generate recommendations
        result = {
            "chapter": chapter,
            "path": str(chapter_dir),
            "nsys_comparison": nsys_comparison,
            "ncu_comparison": ncu_comparison,
        }
        
        recommendations = profile_insights.generate_recommendations_from_profiles(result)
        result["recommendations"] = recommendations
        
        # NEW: Integrate metric-level analysis using _diff_metrics
        metric_analysis = self._analyze_metric_diff(ncu_comparison, nsys_comparison)
        if metric_analysis:
            result["metric_analysis"] = metric_analysis
        
        return result
    
    def _analyze_metric_diff(
        self,
        ncu_comparison: Optional[dict],
        nsys_comparison: Optional[dict],
    ) -> Optional[dict]:
        """Internal: Apply metric-level diff analysis to profile comparison.
        
        Extracts baseline/optimized metrics and runs structured analysis
        to identify what improved, regressed, and any bottleneck shifts.
        """
        try:
            from core.analysis import _diff_metrics, ProfileComparison
        except ImportError:
            return None  # Analysis module not available
        
        if not ncu_comparison or "metrics" not in ncu_comparison:
            return None
        
        # Convert ncu_comparison metrics list to dicts
        baseline_metrics: dict = {}
        optimized_metrics: dict = {}
        
        for m in ncu_comparison.get("metrics", []):
            name = m.get("name", "")
            if not name:
                continue
            try:
                baseline_metrics[name] = float(str(m.get("baseline", 0)).replace(",", ""))
            except (ValueError, TypeError):
                pass
            try:
                optimized_metrics[name] = float(str(m.get("optimized", 0)).replace(",", ""))
            except (ValueError, TypeError):
                pass
        
        if not baseline_metrics or not optimized_metrics:
            return None
        
        # Try to extract timing from metrics or nsys comparison
        baseline_time_us = self._extract_kernel_time(baseline_metrics, nsys_comparison, "baseline")
        optimized_time_us = self._extract_kernel_time(optimized_metrics, nsys_comparison, "optimized")
        
        # Run metric diff analysis
        try:
            comparison: ProfileComparison = _diff_metrics(
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                baseline_time_us=baseline_time_us,
                optimized_time_us=optimized_time_us,
            )
            
            return {
                "speedup": comparison.speedup,
                "baseline_time_us": comparison.baseline_time_us,
                "optimized_time_us": comparison.optimized_time_us,
                "bottleneck_shift": comparison.bottleneck_shift,
                "key_improvements": comparison.key_improvements,
                "remaining_issues": comparison.remaining_issues,
                "improved_count": len(comparison.improved_metrics),
                "regressed_count": len(comparison.regressed_metrics),
                "top_improvements": [
                    {
                        "metric": d.metric_name,
                        "baseline": d.baseline_value,
                        "optimized": d.optimized_value,
                        "change_pct": d.relative_delta_pct,
                    }
                    for d in comparison.improved_metrics[:5]
                ],
                "regressions": [
                    {
                        "metric": d.metric_name,
                        "baseline": d.baseline_value,
                        "optimized": d.optimized_value,
                        "change_pct": d.relative_delta_pct,
                    }
                    for d in comparison.regressed_metrics[:5]
                ],
            }
        except Exception:
            return None
    
    def _extract_kernel_time(
        self,
        metrics: dict,
        nsys_comparison: Optional[dict],
        version: str,
    ) -> float:
        """Extract kernel execution time from metrics or nsys data."""
        # Try common kernel duration metric names
        time_keys = [
            "gpu__time_duration.sum",
            "gpu__time_active.sum",
            "sm__cycles_elapsed.avg",
            "Duration",
            "Kernel Duration",
        ]
        
        for key in time_keys:
            if key in metrics and metrics[key] > 0:
                # Convert to microseconds if needed (cycles need conversion)
                if "cycles" in key.lower():
                    # Rough estimate: assume 1.5GHz clock
                    return metrics[key] / 1500.0
                return metrics[key]
        
        # Fallback: try to get from nsys comparison
        if nsys_comparison and "metrics" in nsys_comparison:
            for m in nsys_comparison["metrics"]:
                name = m.get("name", "").lower()
                if "duration" in name or "time" in name:
                    try:
                        val = float(str(m.get(version, 0)).replace(",", ""))
                        if val > 0:
                            return val
                    except (ValueError, TypeError):
                        pass
        
        # Default fallback
        return 1000.0 if version == "baseline" else 500.0

    def get_profile_recommendations(self) -> dict:
        """Get general profiling recommendations based on all available profiles."""
        recommendations = [
            {
                "title": "Profile Both Versions",
                "description": "Run nsys and ncu on both baseline and optimized code to enable comparison",
                "impact": "Essential for understanding optimization impact",
            },
            {
                "title": "Use NVTX Markers",
                "description": "Add torch.cuda.nvtx.range() markers around key operations for clearer profiling",
                "impact": "Better visibility into performance hotspots",
            },
            {
                "title": "Check Stream Synchronization",
                "description": "Reduce cudaDeviceSynchronize() calls in favor of stream-specific synchronization",
                "impact": "Can significantly improve overlap and throughput",
            },
            {
                "title": "Analyze Kernel Occupancy",
                "description": "Use ncu to check SM occupancy and identify register/shared memory bottlenecks",
                "impact": "Higher occupancy often means better GPU utilization",
            },
        ]
        
        # Check what profiles are available
        pairs = self.list_deep_profile_pairs()
        if pairs.get("count", 0) > 0:
            recommendations.insert(0, {
                "title": f"{pairs['count']} Profile Pairs Available",
                "description": "Select a chapter to view detailed baseline vs optimized comparison",
                "impact": "Ready for analysis",
            })
        
        return {"recommendations": recommendations, "profile_count": pairs.get("count", 0)}

    def _find_profile_directory(self, chapter: str) -> Optional[Path]:
        """Find the directory containing profiles for a chapter."""
        from core.discovery import discover_all_chapters
        
        # Try artifacts directory first
        artifacts_dir = self.bench_root / "artifacts" / chapter
        if artifacts_dir.exists():
            return artifacts_dir
        
        # Try benchmark_profiles directory
        profiles_dir = self.bench_root / "benchmark_profiles" / chapter
        if profiles_dir.exists():
            return profiles_dir
        
        # Search chapter directories
        for dir_path in discover_all_chapters(self.bench_root, bench_roots=self.bench_roots):
            rel = self._relative_to_bench_root(dir_path)
            if chapter in rel or rel.endswith(chapter) or dir_path.name == chapter:
                return dir_path
        
        return None
