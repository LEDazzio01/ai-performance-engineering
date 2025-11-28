"""Baseline vLLM monitoring bundle focused on TTFT and KV cache health."""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from ch18.monitoring_bundle import MonitoringBundle, write_bundle  # noqa: E402
from ch18.monitoring_config import MetricNames, AlertThresholds, load_monitoring_overrides  # noqa: E402


def build_baseline_bundle(metrics: MetricNames, thresholds: AlertThresholds) -> MonitoringBundle:
    """Minimal metrics: TTFT, prefill/decode split, queue drain, KV cache."""
    scrape_config = textwrap.dedent(
        """
        scrape_configs:
          - job_name: vllm
            metrics_path: /metrics
            scrape_interval: 5s
            static_configs:
              - targets: ['localhost:8001']
                labels:
                  service: vllm
                  cluster: dev
        """
    ).strip()

    m = metrics
    recording_rules = textwrap.dedent(
        f"""
        groups:
          - name: vllm-baseline-recording
            interval: 15s
            rules:
              - record: vllm:ttft_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance) (rate({m.ttft_hist}[5m]))
                  )

              - record: vllm:prefill_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance) (rate({m.prefill_hist}[5m]))
                  )

              - record: vllm:decode_seconds:p90
                expr: |
                  histogram_quantile(
                    0.90,
                    sum by (le, instance) (rate({m.decode_hist}[5m]))
                  )

              - record: vllm:req_rate:rps
                expr: |
                  sum by (instance) (rate({m.request_success}[1m]))

              - record: vllm:kv_cache_usage_percent
                expr: |
                  100 * avg by (instance) ({m.gpu_cache_usage})

              - record: vllm:finished_to_active_ratio
                expr: |
                  (
                    sum by (instance) (rate({m.request_success}[1m]))
                  )
                  /
                  (
                    sum by (instance) ({m.num_running}) + 1
                  )
        """
    ).strip()

    t = thresholds
    alert_rules = textwrap.dedent(
        f"""
        groups:
          - name: vllm-baseline-alerts
            rules:
              - alert: VLLMHighTTFTP90
                expr: vllm:ttft_seconds:p90 > {t.ttft_p90_warn}
                for: 5m
                labels:
                  severity: warning
                annotations:
                  summary: "TTFT p90 high on {{ $labels.instance }}"
                  description: |
                    Time to first token p90 is {{ $value }}s.
                    Check prefill batch sizing, scheduler load, or KV cache pressure.

              - alert: VLLMHighPrefillLatency
                expr: vllm:prefill_seconds:p90 > {max(t.prefill_p90_warn, 2.0)}
                for: 10m
                labels:
                  severity: warning
                annotations:
                  summary: "Prefill latency p90 high on {{ $labels.instance }}"
                  description: |
                    Prefill p90 exceeds {max(t.prefill_p90_warn, 2.0)}s, typically due to large prompts or oversized batches.

              - alert: VLLMHighKVCacheUsage
                expr: vllm:kv_cache_usage_percent > {max(t.kv_warn, 95.0)}
                for: 2m
                labels:
                  severity: critical
                annotations:
                  summary: "KV cache nearly full on {{ $labels.instance }}"
                  description: |
                    KV cache usage above {{ $value }}%. Expect stalled requests or allocation failures.

              - alert: VLLMStalledQueue
                expr: |
                  (sum by (instance) ({m.num_running}) > 0)
                  and
                  (sum by (instance) (rate({m.request_success}[3m])) < {t.stalled_finished_rate})
                for: 5m
                labels:
                  severity: critical
                annotations:
                  summary: "Active requests but almost no completions on {{ $labels.instance }}"
                  description: |
                    Requests are running but completions are near zero.
                    Indicates scheduler blockage, cache exhaustion, or CUDA stalls.
        """
    ).strip()

    grafana_dashboard = {
        "title": "vLLM Baseline – TTFT and KV cache",
        "uid": "vllm-baseline-health",
        "schemaVersion": 38,
        "version": 1,
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "type": "timeseries",
                "title": "TTFT p90",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": "vllm:ttft_seconds:p90{instance=~\"$instance\"}",
                        "legendFormat": "p90 {{instance}}",
                    }
                ],
                "gridPos": {"h": 7, "w": 12, "x": 0, "y": 0},
            },
            {
                "id": 2,
                "type": "timeseries",
                "title": "Prefill / decode p90",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": "vllm:prefill_seconds:p90{instance=~\"$instance\"}",
                        "legendFormat": "prefill p90 {{instance}}",
                    },
                    {
                        "refId": "B",
                        "expr": "vllm:decode_seconds:p90{instance=~\"$instance\"}",
                        "legendFormat": "decode p90 {{instance}}",
                    },
                ],
                "gridPos": {"h": 7, "w": 12, "x": 12, "y": 0},
            },
            {
                "id": 3,
                "type": "timeseries",
                "title": "KV cache usage (%)",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": "vllm:kv_cache_usage_percent{instance=~\"$instance\"}",
                        "legendFormat": "{{instance}}",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "orange", "value": t.kv_warn},
                                {"color": "red", "value": max(t.kv_crit, t.kv_warn)},
                            ],
                        },
                    },
                    "overrides": [],
                },
                "gridPos": {"h": 7, "w": 12, "x": 0, "y": 7},
            },
            {
                "id": 4,
                "type": "stat",
                "title": "Finished / active ratio (instant)",
                "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": "vllm:finished_to_active_ratio{instance=~\"$instance\"}",
                        "instant": True,
                        "legendFormat": "ratio {{instance}}",
                    }
                ],
                "gridPos": {"h": 7, "w": 12, "x": 12, "y": 7},
            },
        ],
        "templating": {
            "list": [
                {
                    "name": "instance",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "PROM_DS_UID"},
                    "query": f"label_values({m.gpu_cache_usage}, instance)",
                    "refresh": 1,
                    "multi": True,
                    "includeAll": True,
                }
            ]
        },
    }

    return MonitoringBundle(
        name="baseline_vllm_monitoring",
        scrape_config=scrape_config,
        recording_rules=recording_rules,
        alerting_rules=alert_rules,
        grafana_dashboard=grafana_dashboard,
    )


class BaselineVLLMMonitoringBenchmark(BaseBenchmark):
    """Benchmark wrapper so aisp bench can emit the bundle."""

    def __init__(self, outdir: Optional[Path] = None, config_path: Optional[Path] = None):
        self._device_override = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        super().__init__()
        self._outdir = Path(outdir or Path.cwd() / "artifacts" / "vllm_monitoring_baseline")
        default_cfg = Path(__file__).resolve().parent / "configs" / "vllm_monitoring.yaml"
        self._config_path = Path(config_path) if config_path else default_cfg
        self._metrics: Optional[MetricNames] = None
        self._thresholds: Optional[AlertThresholds] = None
        self._paths: List[Path] = []
        self._written = False

    def _resolve_device(self):  # type: ignore[override]
        return self._device_override

    def setup(self) -> None:
        self._metrics, self._thresholds = load_monitoring_overrides(self._config_path)

    def benchmark_fn(self) -> None:
        if self._written:
            return
        if self._metrics is None or self._thresholds is None:
            raise RuntimeError("Monitoring config not loaded")
        bundle = build_baseline_bundle(self._metrics, self._thresholds)
        self._paths = write_bundle(bundle, self._outdir)
        self._written = True

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self):
        if not self._written:
            return None
        total_bytes = sum(p.stat().st_size for p in self._paths if p.exists())
        return {
            "bundle.files": float(len(self._paths)),
            "bundle.bytes": float(total_bytes),
            "ttft_p90_warn": float(self._thresholds.ttft_p90_warn if self._thresholds else 0.0),
            "kv_warn": float(self._thresholds.kv_warn if self._thresholds else 0.0),
        }


def get_benchmark() -> BaseBenchmark:
    return BaselineVLLMMonitoringBenchmark()


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit baseline vLLM monitoring bundle.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd() / "artifacts" / "vllm_monitoring_baseline",
        help="Directory to write Prometheus and Grafana assets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "vllm_monitoring.yaml",
        help="YAML with metric name and alert-threshold overrides.",
    )
    args = parser.parse_args()

    metrics, thresholds = load_monitoring_overrides(args.config)
    bundle = build_baseline_bundle(metrics, thresholds)
    paths = write_bundle(bundle, args.outdir)
    print(f"Wrote baseline monitoring bundle to {args.outdir}")
    print(f"Metric names: {asdict(metrics)}")
    print(f"Thresholds: {asdict(thresholds)}")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
