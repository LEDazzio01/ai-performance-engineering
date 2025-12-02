"""Lightweight GPU telemetry helpers (temperature, power, utilization).

These helpers rely on ``nvidia-smi`` when available, with graceful fallbacks
when the binary is missing or the GPU is inaccessible (e.g., in CI sandboxes).
The intent is to provide best-effort diagnostics without introducing hard
dependencies on NVML bindings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
import time
from typing import Dict, List, Optional

import torch

try:
    import pynvml  # type: ignore
    _NVML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False

_NVML_INITIALIZED: Optional[bool] = None
_TELEMETRY_CACHE: Dict[int, tuple[float, Dict[str, Optional[float]]]] = {}
# Cache telemetry briefly to avoid repeatedly shelling out to nvidia-smi when NVML is unavailable.
# Kept fixed to avoid environment-dependent behavior inside tight benchmark loops.
_TELEMETRY_TTL_SEC = 1.0


def _ensure_nvml_initialized() -> bool:
    """Initialize NVML once per process."""
    global _NVML_INITIALIZED
    if _NVML_INITIALIZED is not None:
        return _NVML_INITIALIZED
    if not _NVML_AVAILABLE:
        _NVML_INITIALIZED = False
        return False
    try:
        pynvml.nvmlInit()  # type: ignore[attr-defined]
    except Exception:
        _NVML_INITIALIZED = False
        return False
    _NVML_INITIALIZED = True
    return True


_NVIDIA_SMI = shutil.which("nvidia-smi")

_QUERY_FIELD_LIST: List[str] = [
    "temperature.gpu",
    "temperature.memory",
    "power.draw",
    "fan.speed",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.graphics",
    "clocks.current.memory",
    # NVLink counters are sourced via NVML per-link queries, not nvidia-smi.
]
_QUERY_FIELDS = ",".join(_QUERY_FIELD_LIST)

_NVLINK_COUNTER_CONFIGURED: Dict[int, bool] = {}


def _resolve_physical_gpu_index(logical_index: int) -> int:
    """Map a logical CUDA device index to the physical GPU index.

    When CUDA_VISIBLE_DEVICES is set, PyTorch reports logical indices starting
    at zero. ``nvidia-smi`` still uses the physical indices, so we map the first
    visible logical index to the corresponding physical GPU if possible.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return logical_index
    candidates: list[int] = []
    for token in visible.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            candidates.append(int(token))
        except ValueError:
            # Could be MIG identifiers – fall back to logical index mapping.
            return logical_index
    if logical_index < len(candidates):
        return candidates[logical_index]
    return logical_index


def query_gpu_telemetry(device_index: Optional[int] = None) -> Dict[str, Optional[float]]:
    """Return a snapshot of GPU telemetry (temperature, power, utilization, errors).

    Args:
        device_index: Logical CUDA device index. Defaults to the current device.

    Returns:
        Dict with temperature/power/utilization/error data (values may be None when unavailable).
    """
    metrics: Dict[str, Optional[float]] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_index": None,
        "temperature_gpu_c": None,
        "temperature_memory_c": None,
        "power_draw_w": None,
        "fan_speed_pct": None,
        "utilization_gpu_pct": None,
        "utilization_memory_pct": None,
        "graphics_clock_mhz": None,
        "memory_clock_mhz": None,
        "nvlink_tx_gbps": None,
        "nvlink_rx_gbps": None,
        "nvlink_tx_bytes_total": None,
        "nvlink_rx_bytes_total": None,
        "nvlink_link_count": None,
        "nvlink_status": "unknown",
        # ECC and error metrics
        "ecc_errors_corrected": None,
        "ecc_errors_uncorrected": None,
        "retired_pages_sbe": None,
        "retired_pages_dbe": None,
        # PCIe metrics
        "pcie_tx_bytes": None,
        "pcie_rx_bytes": None,
        "pcie_replay_counter": None,
        "pcie_generation": None,
        "pcie_link_width": None,
        # Performance state
        "performance_state": None,
        "throttle_reasons": None,
    }

    if not torch.cuda.is_available():
        return metrics

    logical_index = device_index if device_index is not None else torch.cuda.current_device()
    metrics["gpu_index"] = logical_index

    cache_key = int(logical_index)
    cached = _TELEMETRY_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and _TELEMETRY_TTL_SEC > 0 and (now - cached[0]) < _TELEMETRY_TTL_SEC:
        return dict(cached[1])

    nvml_metrics = _query_via_nvml(logical_index)
    if nvml_metrics is not None:
        metrics.update(nvml_metrics)
        # NVLink counters come solely from NVML; if missing, mark status.
        if metrics["nvlink_tx_bytes_total"] is None or metrics["nvlink_rx_bytes_total"] is None:
            metrics["nvlink_status"] = "nvlink_counters_missing"
        metrics_copy = dict(metrics)
        if _TELEMETRY_TTL_SEC > 0:
            _TELEMETRY_CACHE[cache_key] = (now, metrics_copy)
        return metrics_copy

    smi_metrics = _query_via_nvidia_smi(logical_index)
    if smi_metrics is not None:
        metrics.update(smi_metrics)
    metrics["nvlink_status"] = "nvml_unavailable"
    metrics_copy = dict(metrics)
    if _TELEMETRY_TTL_SEC > 0:
        _TELEMETRY_CACHE[cache_key] = (now, metrics_copy)
    return metrics_copy


def _query_via_nvml(logical_index: int) -> Optional[Dict[str, Optional[float]]]:
    if not _ensure_nvml_initialized():
        return None
    physical_index = _resolve_physical_gpu_index(logical_index)
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)  # type: ignore[attr-defined]
    except Exception:
        return None

    def safe(callable_obj):
        try:
            return callable_obj()
        except Exception:
            return None

    temp_gpu = safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))  # type: ignore[attr-defined]
    if hasattr(pynvml, "NVML_TEMPERATURE_MEMORY"):
        temp_mem = safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY))  # type: ignore[attr-defined]
    else:  # pragma: no cover - legacy GPUs
        temp_mem = None
    power_draw = safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))  # type: ignore[attr-defined]
    fan_speed = safe(lambda: pynvml.nvmlDeviceGetFanSpeed(handle))  # type: ignore[attr-defined]
    utilization = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))  # type: ignore[attr-defined]
    graphics_clock = safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))  # type: ignore[attr-defined]
    memory_clock = safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))  # type: ignore[attr-defined]

    metrics: Dict[str, Optional[float]] = {
        "temperature_gpu_c": float(temp_gpu) if temp_gpu is not None else None,
        "temperature_memory_c": float(temp_mem) if temp_mem is not None else None,
        "power_draw_w": float(power_draw) / 1000.0 if power_draw is not None else None,
        "fan_speed_pct": float(fan_speed) if fan_speed is not None else None,
        "utilization_gpu_pct": float(utilization.gpu) if getattr(utilization, "gpu", None) is not None else None,  # type: ignore[attr-defined]
        "utilization_memory_pct": float(utilization.memory) if getattr(utilization, "memory", None) is not None else None,  # type: ignore[attr-defined]
        "graphics_clock_mhz": float(graphics_clock) if graphics_clock is not None else None,
        "memory_clock_mhz": float(memory_clock) if memory_clock is not None else None,
        "nvlink_tx_gbps": None,
        "nvlink_rx_gbps": None,
        "nvlink_tx_bytes_total": None,
        "nvlink_rx_bytes_total": None,
    }
    
    # ECC error metrics
    ecc_corrected = safe(lambda: pynvml.nvmlDeviceGetTotalEccErrors(  # type: ignore[attr-defined]
        handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_VOLATILE_ECC))  # type: ignore[attr-defined]
    ecc_uncorrected = safe(lambda: pynvml.nvmlDeviceGetTotalEccErrors(  # type: ignore[attr-defined]
        handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC))  # type: ignore[attr-defined]
    metrics["ecc_errors_corrected"] = float(ecc_corrected) if ecc_corrected is not None else None
    metrics["ecc_errors_uncorrected"] = float(ecc_uncorrected) if ecc_uncorrected is not None else None
    
    # Retired pages
    retired_sbe = safe(lambda: pynvml.nvmlDeviceGetRetiredPages(  # type: ignore[attr-defined]
        handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS))  # type: ignore[attr-defined]
    retired_dbe = safe(lambda: pynvml.nvmlDeviceGetRetiredPages(  # type: ignore[attr-defined]
        handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR))  # type: ignore[attr-defined]
    metrics["retired_pages_sbe"] = float(len(retired_sbe)) if retired_sbe is not None else None
    metrics["retired_pages_dbe"] = float(len(retired_dbe)) if retired_dbe is not None else None
    
    # PCIe metrics
    pcie_tx = safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES))  # type: ignore[attr-defined]
    pcie_rx = safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES))  # type: ignore[attr-defined]
    metrics["pcie_tx_bytes"] = float(pcie_tx * 1024) if pcie_tx is not None else None  # KB/s to bytes/s
    metrics["pcie_rx_bytes"] = float(pcie_rx * 1024) if pcie_rx is not None else None
    
    # PCIe link info
    pcie_gen = safe(lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))  # type: ignore[attr-defined]
    pcie_width = safe(lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))  # type: ignore[attr-defined]
    metrics["pcie_generation"] = float(pcie_gen) if pcie_gen is not None else None
    metrics["pcie_link_width"] = float(pcie_width) if pcie_width is not None else None
    
    # PCIe replay counter (link quality indicator)
    pcie_replay = safe(lambda: pynvml.nvmlDeviceGetPcieReplayCounter(handle))  # type: ignore[attr-defined]
    metrics["pcie_replay_counter"] = float(pcie_replay) if pcie_replay is not None else None
    
    # Performance state
    pstate = safe(lambda: pynvml.nvmlDeviceGetPerformanceState(handle))  # type: ignore[attr-defined]
    metrics["performance_state"] = float(pstate) if pstate is not None else None
    
    # Throttle reasons (bitmask)
    throttle = safe(lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))  # type: ignore[attr-defined]
    metrics["throttle_reasons"] = float(throttle) if throttle is not None else None

    nvlink_totals = _query_nvlink_counters_via_nvml(handle, logical_index)
    if nvlink_totals:
        metrics.update(nvlink_totals)
    return metrics
def _query_via_nvidia_smi(logical_index: int) -> Optional[Dict[str, Optional[float]]]:
    if _NVIDIA_SMI is None:
        return None

    physical_index = _resolve_physical_gpu_index(logical_index)
    cmd = [
        _NVIDIA_SMI,
        f"--query-gpu={_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
        f"--id={physical_index}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    parts = [p.strip() for p in result.stdout.strip().split(",")]
    if len(parts) != len(_QUERY_FIELD_LIST):
        return None

    def _to_float(value: str) -> Optional[float]:
        if not value or value.lower() == "n/a":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    values = [_to_float(p) for p in parts]
    metrics = {
        "temperature_gpu_c": values[0],
        "temperature_memory_c": values[1],
        "power_draw_w": values[2],
        "fan_speed_pct": values[3],
        "utilization_gpu_pct": values[4],
        "utilization_memory_pct": values[5],
        "graphics_clock_mhz": values[6],
        "memory_clock_mhz": values[7],
        "nvlink_tx_bytes_total": values[8] if len(values) > 8 else None,
        "nvlink_rx_bytes_total": values[9] if len(values) > 9 else None,
    }
    return metrics


def _extract_field_value(entry) -> Optional[float]:
    if entry is None:
        return None
    if entry.nvmlReturn != getattr(pynvml, "NVML_SUCCESS", 0):
        return None
    value_type = entry.valueType
    val = entry.value
    if value_type == getattr(pynvml, "NVML_VALUE_TYPE_DOUBLE", 0):
        return float(val.dVal)
    if value_type == getattr(pynvml, "NVML_VALUE_TYPE_UNSIGNED_INT", 1):
        return float(val.uiVal)
    if value_type == getattr(pynvml, "NVML_VALUE_TYPE_UNSIGNED_LONG", 2):
        return float(val.ulVal)
    if value_type == getattr(pynvml, "NVML_VALUE_TYPE_UNSIGNED_LONG_LONG", 3):
        return float(val.ullVal)
    if value_type == getattr(pynvml, "NVML_VALUE_TYPE_SIGNED_LONG_LONG", 4):
        return float(val.sllVal)
    return None


def _query_nvlink_counters_via_nvml(handle, logical_index: int) -> Optional[Dict[str, Optional[float]]]:
    if not _NVML_AVAILABLE:
        return None
    try:
        max_links = getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 12)
    except Exception:
        max_links = 12

    status = "ok"

    if logical_index not in _NVLINK_COUNTER_CONFIGURED:
        control_cls = getattr(pynvml, "nvmlNvLinkUtilizationControl_t", None)
        unit_bytes = getattr(pynvml, "NVML_NVLINK_COUNTER_UNIT_BYTES", None)
        pkt_all = getattr(pynvml, "NVML_NVLINK_COUNTER_PKTFILTER_ALL", None)
        if control_cls and unit_bytes is not None and pkt_all is not None:
            ctrl = control_cls()
            ctrl.units = unit_bytes
            ctrl.pktfilter = pkt_all
            for link in range(max_links):
                try:
                    pynvml.nvmlDeviceSetNvLinkUtilizationControl(handle, link, 0, ctrl, 0)  # type: ignore[attr-defined]
                except Exception:
                    continue
        _NVLINK_COUNTER_CONFIGURED[logical_index] = True

    total_rx = 0.0
    total_tx = 0.0
    active_links = 0
    for link in range(max_links):
        try:
            state = pynvml.nvmlDeviceGetNvLinkState(handle, link)  # type: ignore[attr-defined]
        except Exception:
            continue
        if not state:
            continue
        try:
            rx, tx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, link, 0)  # type: ignore[attr-defined]
        except Exception:
            continue
        total_rx += float(rx)
        total_tx += float(tx)
        active_links += 1

    if active_links == 0:
        status = "nvlink_disabled"
    if total_rx == 0.0 and total_tx == 0.0:
        status = "nvlink_counters_missing"

    return {
        "nvlink_tx_bytes_total": total_tx if total_tx > 0 else None,
        "nvlink_rx_bytes_total": total_rx if total_rx > 0 else None,
        "nvlink_link_count": float(active_links),
        "nvlink_status": status,
    }


def format_gpu_telemetry(metrics: Dict[str, Optional[float]]) -> str:
    """Return a human-readable summary string for telemetry."""
    if not metrics:
        return "GPU telemetry unavailable"

    parts = []
    temp = metrics.get("temperature_gpu_c")
    if temp is not None:
        parts.append(f"temp={temp:.1f}°C")
    power = metrics.get("power_draw_w")
    if power is not None:
        parts.append(f"power={power:.1f}W")
    util = metrics.get("utilization_gpu_pct")
    if util is not None:
        parts.append(f"util={util:.0f}%")
    mem_util = metrics.get("utilization_memory_pct")
    if mem_util is not None:
        parts.append(f"mem_util={mem_util:.0f}%")
    clock = metrics.get("graphics_clock_mhz")
    if clock is not None:
        parts.append(f"clock={clock:.0f}MHz")
    fan = metrics.get("fan_speed_pct")
    if fan is not None:
        parts.append(f"fan={fan:.0f}%")
    
    # PCIe metrics
    pcie_gen = metrics.get("pcie_generation")
    pcie_width = metrics.get("pcie_link_width")
    if pcie_gen is not None and pcie_width is not None:
        parts.append(f"pcie=Gen{int(pcie_gen)}x{int(pcie_width)}")
    
    # Error indicators (only show if non-zero)
    ecc_corrected = metrics.get("ecc_errors_corrected")
    if ecc_corrected is not None and ecc_corrected > 0:
        parts.append(f"ecc_corr={int(ecc_corrected)}")
    ecc_uncorrected = metrics.get("ecc_errors_uncorrected")
    if ecc_uncorrected is not None and ecc_uncorrected > 0:
        parts.append(f"ecc_uncorr={int(ecc_uncorrected)}⚠")
    
    # Throttle indicator
    throttle = metrics.get("throttle_reasons")
    if throttle is not None and throttle > 0:
        parts.append(f"throttled⚠")
    
    if not parts:
        return "GPU telemetry unavailable"
    return ", ".join(parts)


def get_throttle_reason_names(throttle_bitmask: int) -> List[str]:
    """Decode throttle reasons bitmask into human-readable names."""
    if not _NVML_AVAILABLE or throttle_bitmask == 0:
        return []
    
    reasons = []
    # NVML throttle reason constants
    reason_map = {
        getattr(pynvml, "nvmlClocksThrottleReasonGpuIdle", 0x0000000000000001): "idle",
        getattr(pynvml, "nvmlClocksThrottleReasonApplicationsClocksSetting", 0x0000000000000002): "app_clocks",
        getattr(pynvml, "nvmlClocksThrottleReasonSwPowerCap", 0x0000000000000004): "sw_power_cap",
        getattr(pynvml, "nvmlClocksThrottleReasonHwSlowdown", 0x0000000000000008): "hw_slowdown",
        getattr(pynvml, "nvmlClocksThrottleReasonSyncBoost", 0x0000000000000010): "sync_boost",
        getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0x0000000000000020): "sw_thermal",
        getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0x0000000000000040): "hw_thermal",
        getattr(pynvml, "nvmlClocksThrottleReasonHwPowerBrakeSlowdown", 0x0000000000000080): "power_brake",
        getattr(pynvml, "nvmlClocksThrottleReasonDisplayClockSetting", 0x0000000000000100): "display_clocks",
    }
    
    for mask, name in reason_map.items():
        if mask and (throttle_bitmask & mask):
            reasons.append(name)
    
    return reasons
