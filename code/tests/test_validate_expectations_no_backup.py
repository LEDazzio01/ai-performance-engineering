from __future__ import annotations

import json
from pathlib import Path

from core.benchmark.validate_expectations import fix_file


def test_fix_file_respects_backup_flag(tmp_path: Path) -> None:
    path = tmp_path / "expectations_b200.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hardware_key": "b200",
                "examples": {
                    "example": {
                        "metadata": {"optimization_goal": "speed", "best_optimization_speedup": 0.0},
                        "metrics": {
                            "baseline_time_ms": 10.0,
                            "best_optimized_time_ms": 5.0,
                            "best_speedup": 0.0,
                        },
                    }
                },
            }
        )
        + "\n"
    )

    modified = fix_file(path, backup=False)
    assert modified is True
    assert (tmp_path / "expectations_b200.json.bak").exists() is False


def test_fix_file_creates_backup_by_default(tmp_path: Path) -> None:
    path = tmp_path / "expectations_b200.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hardware_key": "b200",
                "examples": {
                    "example": {
                        "metadata": {"optimization_goal": "speed", "best_optimization_speedup": 0.0},
                        "metrics": {
                            "baseline_time_ms": 10.0,
                            "best_optimized_time_ms": 5.0,
                            "best_speedup": 0.0,
                        },
                    }
                },
            }
        )
        + "\n"
    )

    modified = fix_file(path, backup=True)
    assert modified is True
    assert (tmp_path / "expectations_b200.json.bak").exists() is True
