"""
Shared NCU deep-dive helpers to parse NCU artifacts and provide summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


def load_ncu_deepdive(code_root: Path) -> Dict[str, Any]:
    """Load NCU metrics and synthesize basic analysis."""
    profile_dirs = [
        code_root / "benchmark_profiles",
        code_root / "artifacts" / "profiles",
    ]

    ncu_data: Dict[str, Any] = {
        "available": False,
        "metrics": [],
        "occupancy_analysis": {},
        "memory_analysis": {},
        "warp_stalls": {},
        "recommendations": [],
        "source_samples": [],
        "disassembly": [],
    }

    ncu_files: List[Path] = []
    for profile_dir in profile_dirs:
        if profile_dir.exists():
            ncu_files.extend(profile_dir.glob("**/*.ncu-rep"))
            ncu_files.extend(profile_dir.glob("**/*ncu*.csv"))

    if ncu_files:
        ncu_data["available"] = True
        ncu_data["files_found"] = len(ncu_files)
        try:
            latest_ncu = sorted(ncu_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
            ncu_data["latest_file"] = latest_ncu.name
            if latest_ncu.suffix == ".csv":
                with open(latest_ncu) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ncu_data["metrics"].append(dict(row))
            else:
                # Best-effort source extraction for the latest .ncu-rep
                try:
                    from tools.core.profile_insights import _extract_ncu_sources, _extract_ncu_disassembly
                    ncu_data["source_samples"] = _extract_ncu_sources(latest_ncu)
                    ncu_data["disassembly"] = _extract_ncu_disassembly(latest_ncu)
                except Exception:
                    ncu_data["source_samples"] = []
                    ncu_data["disassembly"] = []
        except Exception as e:
            ncu_data["parse_error"] = str(e)

    return ncu_data
