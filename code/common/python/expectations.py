"""Per-chapter expectation tracking for benchmark results."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPECTATION_FILENAME_TEMPLATE = "expectations_{hardware_key}.json"

# Tolerances to avoid flagging noise as regressions.
# GPU benchmarks typically have 10-30% variance between runs due to thermal
# throttling, memory state, kernel JIT, and first-call overhead.
RELATIVE_TOLERANCE = 0.25  # 25%
ABSOLUTE_TOLERANCE = 1e-5

# Metric direction hints. Extend as new metrics are tracked.
METRIC_DIRECTIONS: Dict[str, str] = {
    "best_speedup": "higher",
    "best_optimized_speedup": "higher",
    "baseline_time_ms": "lower",
    "best_optimized_time_ms": "lower",
    "baseline_throughput.requests_per_s": "higher",
    "baseline_throughput.tokens_per_s": "higher",
    "baseline_throughput.samples_per_s": "higher",
    "baseline_throughput.goodput": "higher",
    "baseline_throughput.latency_ms": "lower",
    "best_optimized_throughput.requests_per_s": "higher",
    "best_optimized_throughput.tokens_per_s": "higher",
    "best_optimized_throughput.samples_per_s": "higher",
    "best_optimized_throughput.goodput": "higher",
    "best_optimized_throughput.latency_ms": "lower",
    "baseline_p75_ms": "lower",
    "baseline_p90_ms": "lower",
    "best_optimized_p75_ms": "lower",
    "best_optimized_p90_ms": "lower",
    "baseline_custom.scenario_total_phase_ms": "lower",
    "best_optimized_custom.scenario_total_phase_ms": "lower",
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown"


def detect_expectation_key() -> str:
    """Return a hardware/environment key for selecting expectation files."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).name
            gpu_slug = _slugify(re.sub(r"^nvidia_", "", _slugify(name)))
            count = torch.cuda.device_count()
            prefix = f"{count}x_" if count > 1 else ""
            return _slugify(f"{prefix}{gpu_slug}")
    except RuntimeError:
        pass  # CUDA not initialized

    return "unknown"


def _tolerance(expected: float) -> float:
    return max(abs(expected) * RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def _format_delta(observed: Optional[float], expected: Optional[float]) -> Dict[str, Optional[float]]:
    if observed is None or expected is None:
        return {"delta": None, "delta_pct": None}
    delta = observed - expected
    if expected == 0:
        delta_pct = math.inf if delta > 0 else -math.inf if delta < 0 else 0.0
    else:
        delta_pct = (delta / expected) * 100.0
    return {"delta": delta, "delta_pct": delta_pct}


def _compare_metric(metric: str, direction: Optional[str], observed: Optional[float], expected: Optional[float]) -> Dict[str, Any]:
    comparison = {
        "metric": metric,
        "direction": direction,
        "expected": expected,
        "observed": observed,
        "status": "not_tracked",
    }
    comparison.update(_format_delta(observed, expected))

    if observed is None:
        comparison["status"] = "missing"
        return comparison
    if direction is None:
        comparison["status"] = "recorded"
        return comparison
    if expected is None:
        comparison["status"] = "new"
        return comparison

    tol = _tolerance(expected)
    if direction == "higher":
        if observed > expected + tol:
            comparison["status"] = "improved"
        elif observed < expected - tol:
            comparison["status"] = "regressed"
        else:
            comparison["status"] = "met"
    else:  # lower is better
        if observed < expected - tol:
            comparison["status"] = "improved"
        elif observed > expected + tol:
            comparison["status"] = "regressed"
        else:
            comparison["status"] = "met"
    return comparison


@dataclass
class ExpectationEvaluation:
    example_key: str
    hardware_key: str
    expectation_exists: bool
    regressed: bool
    comparisons: List[Dict[str, Any]]
    regressions: List[Dict[str, Any]]
    improvements: List[Dict[str, Any]]
    updated_metrics: List[str] = field(default_factory=list)
    expectation_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_key": self.example_key,
            "hardware_key": self.hardware_key,
            "expectation_exists": self.expectation_exists,
            "regressed": self.regressed,
            "comparisons": self.comparisons,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "updated_metrics": self.updated_metrics,
            "expectation_path": str(self.expectation_path) if self.expectation_path else None,
        }


class ExpectationsStore:
    """Maintain expectation files per chapter and hardware target."""

    def __init__(self, chapter_dir: Path, hardware_key: str, *, accept_regressions: bool = False) -> None:
        self.chapter_dir = chapter_dir
        self.hardware_key = hardware_key
        self.path = chapter_dir / EXPECTATION_FILENAME_TEMPLATE.format(hardware_key=hardware_key)
        self._data = self._load()
        self._changed = False
        self._accept_regressions = accept_regressions

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
        data.setdefault("hardware_key", self.hardware_key)
        data.setdefault("examples", {})
        return data

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def evaluate(self, example_key: str, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None) -> Optional[ExpectationEvaluation]:
        """Compare metrics against expectations and update stored bests if improved."""
        if not metrics:
            return None

        metadata = metadata or {}
        examples = self._data.setdefault("examples", {})
        entry = examples.get(example_key)
        expectation_exists = bool(entry and entry.get("metrics"))

        if entry is None:
            entry = {
                "example": metadata.get("example", example_key),
                "type": metadata.get("type", "python"),
                "metrics": {},
                "metadata": {},
            }
            examples[example_key] = entry

        stored_metrics: Dict[str, float] = entry.setdefault("metrics", {})
        comparisons: List[Dict[str, Any]] = []
        regressions: List[Dict[str, Any]] = []
        improvements: List[Dict[str, Any]] = []
        updated_metrics: List[str] = []

        for metric_name in sorted(metrics.keys()):
            if metric_name not in METRIC_DIRECTIONS:
                continue
            observed = metrics[metric_name]
            direction = METRIC_DIRECTIONS.get(metric_name)
            expected = stored_metrics.get(metric_name)
            comp = _compare_metric(metric_name, direction, observed, expected)
            comparisons.append(comp)
            if comp["status"] == "regressed":
                if self._accept_regressions:
                    stored_metrics[metric_name] = observed
                    comp["status"] = "updated"
                    improvements.append(comp)
                    updated_metrics.append(metric_name)
                else:
                    regressions.append(comp)
            elif comp["status"] in {"improved", "new"}:
                stored_metrics[metric_name] = observed
                improvements.append(comp)
                updated_metrics.append(metric_name)

        # Metrics tracked previously but not emitted in this run
        for metric_name in sorted(stored_metrics.keys()):
            if metric_name not in metrics:
                direction = METRIC_DIRECTIONS.get(metric_name)
                comp = {
                    "metric": metric_name,
                    "direction": direction,
                    "expected": stored_metrics.get(metric_name),
                    "observed": None,
                    "status": "not_reported",
                    "delta": None,
                    "delta_pct": None,
                }
                comparisons.append(comp)

        regressed = bool(regressions)

        if not regressed and updated_metrics:
            self._update_metadata(entry, metadata)

        if not expectation_exists and not regressed and not updated_metrics:
            # First run with metrics equal to defaults: still treat as update.
            self._update_metadata(entry, metadata)

        if updated_metrics:
            self._changed = True

        return ExpectationEvaluation(
            example_key=example_key,
            hardware_key=self.hardware_key,
            expectation_exists=expectation_exists,
            regressed=regressed,
            comparisons=sorted(comparisons, key=lambda c: c["metric"]),
            regressions=regressions,
            improvements=improvements,
            updated_metrics=updated_metrics,
            expectation_path=self.path if self.path else None,
        )

    def _update_metadata(self, entry: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        entry["example"] = metadata.get("example", entry.get("example"))
        entry["type"] = metadata.get("type", entry.get("type", "python"))
        meta = entry.setdefault("metadata", {})
        if metadata.get("best_optimization"):
            meta["best_optimization"] = metadata["best_optimization"]
        if metadata.get("best_optimization_file"):
            meta["best_optimization_file"] = metadata["best_optimization_file"]
        if metadata.get("best_optimization_speedup") is not None:
            meta["best_optimization_speedup"] = metadata["best_optimization_speedup"]
        if metadata.get("best_optimization_time_ms") is not None:
            meta["best_optimization_time_ms"] = metadata["best_optimization_time_ms"]
        if metadata.get("git_commit"):
            meta["git_commit"] = metadata["git_commit"]
        meta["updated_at"] = datetime.now().isoformat()

    def save(self) -> None:
        if not self._changed:
            return
        serialized = json.dumps(self._data, indent=2, sort_keys=True)
        self.path.write_text(serialized + "\n")
        self._changed = False
