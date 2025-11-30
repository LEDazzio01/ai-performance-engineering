"""Automation helpers for running the PyTorch profiler from CLI/MCP/API/UI.

This wraps the `core.scripts.profiling.pytorch_profiler_runner` module so that
callers can trigger captures with consistent defaults (NVTX range + lineinfo)
and retrieve lightweight summaries for dashboards.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.utils.logger import get_logger

logger = get_logger(__name__)


class TorchProfilerAutomation:
    """Run torch.profiler captures for an arbitrary Python script."""

    def __init__(self, output_root: Path = Path("artifacts/torch-profiles")):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.last_error: Optional[str] = None
        self.last_run: Dict[str, Any] = {}

    def _build_env(self, force_lineinfo: bool = True) -> Dict[str, str]:
        """Mirror Nsight env wiring so source mapping stays consistent."""
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[1]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else str(repo_root)
        if force_lineinfo:
            def _append_flag(key: str, flag: str) -> None:
                current = env.get(key, "").strip()
                if flag not in current.split():
                    env[key] = f"{flag} {current}".strip()
            _append_flag("NVCC_PREPEND_FLAGS", "-lineinfo")
            _append_flag("TORCH_NVCC_FLAGS", "-lineinfo")
        return env

    def profile(
        self,
        script: Path,
        output_name: Optional[str] = None,
        mode: str = "full",
        script_args: Optional[List[str]] = None,
        force_lineinfo: bool = True,
        timeout_seconds: Optional[int] = None,
        nvtx_label: str = "aisp_torch_profile",
        use_nvtx: bool = True,
    ) -> Dict[str, Any]:
        """Run torch.profiler and return a summary dict."""
        self.last_error = None
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_name = output_name or script.stem or "torch_profile"
        capture_dir = self.output_root / f"{safe_name}_{ts}"
        capture_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "core.scripts.profiling.pytorch_profiler_runner",
            str(script),
            "--output-dir",
            str(capture_dir),
            "--profile-mode",
            mode,
            "--nvtx-label",
            nvtx_label,
        ]
        if not use_nvtx:
            cmd.append("--no-nvtx")
        if not force_lineinfo:
            cmd.append("--no-force-lineinfo")
        if script_args:
            cmd.append("--script-args")
            cmd.extend(script_args)

        logger.info("Running torch profiler: %s", " ".join(cmd))
        self.last_run = {"cmd": cmd, "capture_dir": str(capture_dir), "mode": mode}
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
                env=self._build_env(force_lineinfo=force_lineinfo),
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - runtime path
            self.last_error = f"torch profiler timed out after {timeout_seconds}s"
            self.last_run.update({"timeout_hit": True, "stdout": exc.stdout or "", "stderr": exc.stderr or ""})
            return {
                "success": False,
                "error": self.last_error,
                "timeout_seconds": timeout_seconds,
                "capture_dir": str(capture_dir),
            }

        self.last_run.update({"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode})
        if proc.returncode != 0:
            self.last_error = proc.stderr or proc.stdout or f"torch profiler exited with {proc.returncode}"
            return {
                "success": False,
                "error": self.last_error,
                "capture_dir": str(capture_dir),
                "returncode": proc.returncode,
            }

        # Collect artifacts
        trace_path = capture_dir / "trace.json"
        if not trace_path.exists():
            # Fallback to mode-specific trace
            alt_trace = capture_dir / f"chrome_trace_{mode}.json"
            trace_path = alt_trace if alt_trace.exists() else trace_path
        metadata_path = capture_dir / "metadata.json"
        summary_path = capture_dir / "torch_profile_summary.json"

        def _load(path: Path) -> Optional[Any]:
            try:
                if path.exists():
                    return json.loads(path.read_text())
            except Exception:
                return None
            return None

        result = {
            "success": True,
            "capture_dir": str(capture_dir),
            "trace_path": str(trace_path) if trace_path.exists() else None,
            "metadata": _load(metadata_path),
            "summary": _load(summary_path),
            "mode": mode,
            "nvtx_label": nvtx_label,
            "force_lineinfo": bool(force_lineinfo),
            "timeout_seconds": timeout_seconds,
        }
        return result
