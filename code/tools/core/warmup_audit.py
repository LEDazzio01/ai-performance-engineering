"""
Shared warmup audit runner.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def run_warmup_audit(code_root: Path, check_recommended: bool = False) -> Dict[str, Any]:
    """Run the warmup audit script and return parsed results."""
    audit_script = code_root / "scripts" / "audit_warmup_settings.py"
    if not audit_script.exists():
        return {
            "success": False,
            "error": "Audit script not found",
            "issues": [],
            "total_scanned": 0,
        }

    cmd = [sys.executable, str(audit_script), "--json"]
    if check_recommended:
        cmd.append("--check-recommended")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(code_root),
        )
        try:
            data = json.loads(result.stdout)
            return {
                "success": True,
                "issues": data.get("issues", []),
                "total_scanned": data.get("total_scanned", 0),
                "passing": data.get("passing", 0),
                "exit_code": result.returncode,
                "check_recommended": check_recommended,
            }
        except json.JSONDecodeError:
            if result.returncode == 0:
                return {
                    "success": True,
                    "issues": [],
                    "total_scanned": 0,
                    "passing": 0,
                    "exit_code": 0,
                    "message": "Audit passed - all benchmarks have sufficient warmup",
                    "check_recommended": check_recommended,
                }
            return {
                "success": False,
                "error": result.stderr or result.stdout or "Unknown error",
                "exit_code": result.returncode,
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Audit timed out", "issues": [], "total_scanned": 0}
    except Exception as e:
        return {"success": False, "error": str(e), "issues": [], "total_scanned": 0}

