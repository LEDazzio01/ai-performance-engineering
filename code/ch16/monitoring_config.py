"""Shared metric/threshold configuration for vLLM monitoring bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Any, Mapping

import yaml


@dataclass
class MetricNames:
    """Prometheus metric names for vLLM v1 surfaces."""

    ttft_hist: str = "vllm:time_to_first_token_seconds_bucket"
    prefill_hist: str = "vllm:request_prefill_time_seconds_bucket"
    decode_hist: str = "vllm:request_decode_time_seconds_bucket"
    e2e_hist: str = "vllm:e2e_request_latency_seconds_bucket"
    inter_token_hist: str = "vllm:time_per_output_token_seconds_bucket"
    request_success: str = "vllm:request_success_total"
    num_running: str = "vllm:num_requests_running"
    num_waiting: str = "vllm:num_requests_waiting"
    gpu_cache_usage: str = "vllm:gpu_cache_usage_perc"
    cudagraph_mode_info: str = "vllm:cudagraph_mode_info"
    enginecore_errors_total: str = "vllm_enginecore_errors_total"
    scheduler_errors_total: str = "vllm_scheduler_errors_total"


@dataclass
class AlertThresholds:
    """Default alert thresholds; override per SLA."""

    ttft_p90_warn: float = 0.6
    ttft_p99_crit: float = 1.5
    prefill_p90_warn: float = 1.5
    decode_p90_warn: float = 10.0
    inter_token_p90_warn: float = 0.15
    kv_warn: float = 90.0
    kv_crit: float = 98.0
    stalled_finished_rate: float = 0.05
    stalled_active_floor: float = 50.0


def load_monitoring_overrides(config_path: Optional[Path]) -> Tuple[MetricNames, AlertThresholds]:
    """Load metric name/threshold overrides from YAML if provided."""
    metrics = MetricNames()
    thresholds = AlertThresholds()

    if not config_path:
        return metrics, thresholds

    resolved = config_path.expanduser()
    if not resolved.exists():
        return metrics, thresholds

    raw = _safe_load(resolved)
    metric_overrides = raw.get("metrics", {}) if isinstance(raw, Mapping) else {}
    threshold_overrides = raw.get("thresholds", {}) if isinstance(raw, Mapping) else {}

    _update_dataclass(metrics, metric_overrides)
    _update_dataclass(thresholds, threshold_overrides)
    return metrics, thresholds


def _safe_load(path: Path) -> Any:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _update_dataclass(obj: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
