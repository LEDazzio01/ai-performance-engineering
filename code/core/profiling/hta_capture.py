"""HTA capture automation: run nsys with HTA-friendly flags and produce an analysis report."""

from __future__ import annotations

import subprocess
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.profiling.hta_integration import HTAAnalyzer
from core.profiling.nsight_automation import NsightAutomation
from core.utils.logger import get_logger

logger = get_logger(__name__)


class HTACaptureAutomation:
    """Run an Nsight Systems capture and immediately analyze it with HTA."""

    def __init__(self, output_root: Path = Path("artifacts/hta")):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.last_error: Optional[str] = None
        self.last_run: Dict[str, Any] = {}

    def capture(
        self,
        command: List[str],
        output_name: str = "hta_capture",
        preset: str = "full",
        force_lineinfo: bool = True,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Capture a trace with nsys and run HTA over the exported JSON trace."""
        self.last_error = None
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{output_name}_{ts}"
        output_dir = self.output_root
        auto = NsightAutomation(output_dir)
        if not auto.nsys_available:
            self.last_error = "nsys is not installed or not on PATH"
            return {"success": False, "error": self.last_error}

        logger.info("Starting HTA capture with preset=%s, force_lineinfo=%s", preset, force_lineinfo)
        rep_path = auto.profile_nsys(
            command=command,
            output_name=base_name,
            trace_cuda=True,
            trace_nvtx=True,
            trace_osrt=True,
            full_timeline=True,
            trace_forks=True,
            preset=preset or "full",
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
        )
        self.last_run = getattr(auto, "last_run", {}) or {}
        if rep_path is None:
            self.last_error = auto.last_error or "nsys capture failed"
            return {
                "success": False,
                "error": self.last_error,
                "nsys_available": auto.nsys_available,
                "run_details": self.last_run,
            }

        # Export to Chrome trace JSON for HTA consumption
        trace_base = rep_path.with_suffix("")  # drop .nsys-rep
        export_output = trace_base.with_suffix(".json")
        try:
            if export_output.exists():
                export_output.unlink()
        except Exception:
            pass
        export_cmd = ["nsys", "export", "--type=json", f"--output={export_output}", "--force-overwrite=true", str(rep_path)]
        logger.info("Exporting nsys trace for HTA: %s", " ".join(export_cmd))
        export_res = subprocess.run(
            export_cmd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        self.last_run["export"] = {
            "cmd": export_cmd,
            "returncode": export_res.returncode,
            "stdout": export_res.stdout,
            "stderr": export_res.stderr,
        }
        trace_path = export_output if export_output.exists() else trace_base.with_suffix(".json")
        if not trace_path.exists() and trace_base.exists():
            trace_path = trace_base
        if not trace_path.exists() or export_res.returncode != 0:
            self.last_error = "nsys export did not produce a JSON trace"
            return {
                "success": False,
                "error": self.last_error,
                "nsys_output": str(rep_path),
                "export": self.last_run.get("export"),
                "returncode": export_res.returncode,
            }

        analyzer = HTAAnalyzer(output_dir=output_dir)
        report = analyzer.analyze_trace(trace_path)
        hta_report_path = output_dir / f"{base_name}_hta_report.json"
        try:
            analyzer.export_report(report, hta_report_path, format="json")
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = f"Failed to export HTA report: {exc}"
            return {"success": False, "error": self.last_error, "trace_path": str(trace_path)}

        # Optional: collect kernel stats directly from nsys for richer top_kernels
        kernel_stats: List[Dict[str, Any]] = []
        try:
            stats_output = output_dir / f"{base_name}_stats"
            stats_cmd = [
                "nsys",
                "stats",
                "-r",
                "cuda_gpu_kern_sum",
                "--format",
                "json",
                "--output",
                str(stats_output),
                str(rep_path),
            ]
            stats_res = subprocess.run(
                stats_cmd,
                text=True,
                capture_output=True,
                timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            )
            self.last_run["kernel_stats"] = {
                "cmd": stats_cmd,
                "returncode": stats_res.returncode,
                "stdout": stats_res.stdout,
                "stderr": stats_res.stderr,
            }
            stats_path = output_dir / f"{stats_output.name}_cuda_gpu_kern_sum.json"
            if stats_path.exists():
                rows = json.loads(stats_path.read_text())
                if isinstance(rows, list):
                    rows_sorted = sorted(rows, key=lambda r: r.get("Total Time (ns)", 0), reverse=True)
                    for row in rows_sorted[:20]:
                        kernel_stats.append(
                            {
                                "name": row.get("Name"),
                                "time_us": (row.get("Total Time (ns)", 0) or 0) / 1000.0,
                                "pct": row.get("Time (%)", 0.0),
                                "count": row.get("Instances", 0),
                                "avg_ns": row.get("Avg (ns)", 0),
                            }
                        )
        except Exception as exc:
            self.last_run["kernel_stats_error"] = str(exc)

        payload = report.to_dict()
        if kernel_stats:
            payload["top_kernels"] = kernel_stats[:10]
            payload.setdefault("kernel_analysis", {})["top_kernels"] = kernel_stats[:20]
        payload.update(
            {
                "success": True,
                "nsys_output": str(rep_path),
                "trace_path": str(trace_path),
                "hta_report": str(hta_report_path),
                "preset": preset or "full",
                "force_lineinfo": bool(force_lineinfo),
                "run_details": self.last_run,
            }
        )
        # Persist enriched payload so downstream loaders see top_kernels
        try:
            Path(hta_report_path).write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # pragma: no cover - best-effort
            self.last_run["hta_report_write_error"] = str(exc)
        return payload
