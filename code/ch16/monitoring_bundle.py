"""Shared helpers for emitting vLLM monitoring bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass
class MonitoringBundle:
    name: str
    scrape_config: str
    recording_rules: str
    alerting_rules: str
    grafana_dashboard: Mapping[str, object]


def write_bundle(bundle: MonitoringBundle, outdir: Path) -> Sequence[Path]:
    """
    Write the bundle files into outdir and return the written paths.

    Files are prefixed with the bundle name to make it easy to keep multiple
    variants side by side.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    paths = [
        _write_text(outdir / f"{bundle.name}_prometheus_scrape.yaml", bundle.scrape_config),
        _write_text(outdir / f"{bundle.name}_prometheus_recording_rules.yaml", bundle.recording_rules),
        _write_text(outdir / f"{bundle.name}_prometheus_alerts.yaml", bundle.alerting_rules),
        _write_json(outdir / f"{bundle.name}_grafana_dashboard.json", bundle.grafana_dashboard),
    ]
    return paths


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2))
    return path
