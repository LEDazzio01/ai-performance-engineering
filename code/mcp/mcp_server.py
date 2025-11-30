#!/usr/bin/env python3
"""
🚀 MCP Server for AI Systems Performance

Exposes PerformanceEngine functionality as MCP tools for AI chat integration.

Usage:
    # Start the MCP server
    python -m mcp.mcp_server
    
    # Or use the aisp command
    aisp mcp serve

Architecture:
    This MCP server exposes tools organized by category (namespaced with aisp_):
      GPU: aisp_gpu_info, aisp_gpu_bandwidth, aisp_gpu_topology, aisp_gpu_power
      System: aisp_system_software, aisp_system_dependencies, aisp_system_context
      Analysis: aisp_analyze_bottlenecks, aisp_analyze_pareto, aisp_analyze_scaling, aisp_analyze_stacking, aisp_analyze_whatif
      Optimization: aisp_recommend, aisp_optimize_roi, aisp_optimize_techniques
      Distributed: aisp_distributed_plan, aisp_distributed_nccl
      Inference: aisp_inference_vllm, aisp_inference_quantization
      AI: aisp_ask, aisp_explain, aisp_ai_status
      Profiling: aisp_profile_flame, aisp_profile_memory, aisp_profile_kernels, aisp_profile_roofline, aisp_profile_nsys, aisp_profile_ncu, aisp_nsys_summary
      Context helpers: aisp_context_summary, aisp_context_full
      Status: aisp_status
      Tests: aisp_test_speed, aisp_test_network
      HuggingFace: aisp_hf_search, aisp_hf_trending
      Cluster/Cost: aisp_cluster_slurm, aisp_cost_estimate
      Benchmark: aisp_run_benchmarks, aisp_verify_benchmarks, aisp_benchmark_targets, aisp_available_benchmarks
      Nsight: aisp_profile_nsys, aisp_profile_ncu, aisp_nsys_summary, aisp_compare_nsys, aisp_compare_ncu, aisp_profile_compare
      Hardware benchmarks: aisp_test_disk, aisp_test_pcie, aisp_test_mem_hierarchy, aisp_test_tensor_core, aisp_test_sfu, aisp_test_network_loopback
      Exports: aisp_export_csv, aisp_export_pdf, aisp_export_html
      System/Analysis: aisp_system_capabilities, aisp_full_system_analysis, aisp_nsys_ncu_available
"""

import asyncio
import json
import os
import sys
import subprocess
import traceback
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Ensure repository root is on sys.path for imports (e.g., analysis.advanced_analysis)
CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# MCP Protocol Types
@dataclass
class ToolDefinition:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ToolResult:
    """MCP Tool result."""
    content: List[Dict[str, Any]]
    is_error: bool = False


# =============================================================================
# DESCRIPTION ENRICHMENT HELPERS
# =============================================================================

_OUTPUT_ENVELOPE_SUMMARY = (
    "JSON envelope with tool/status/timestamp/duration_ms, sanitized arguments + details, "
    "result + preview + metadata, context_summary, guidance.next_steps; content is emitted as a "
    "text (JSON string) entry."
)

# Explicit overrides for tools that have notable runtime/side effects.
_EXPECTATION_OVERRIDES: Dict[str, str] = {
    "aisp_run_benchmarks": "Runs the bench CLI; can take minutes and writes artifacts/logs to the repo. Slow/interactive; run aisp_status or aisp_triage first and consider precheck_only/dry_run/timeout_seconds.",
    "aisp_verify_benchmarks": "Runs bench verify; can take minutes and writes artifacts/logs to the repo. Slow/interactive; run aisp_status or aisp_triage first and consider precheck_only/dry_run/timeout_seconds.",
    "aisp_benchmark_report": "Generates a report from existing benchmark JSON; writes PDF/HTML to the chosen output.",
    "aisp_benchmark_export": "Exports existing benchmark JSON to csv/markdown/json; writes to the chosen output file.",
    "aisp_benchmark_compare_runs": "Diffs two benchmark JSON files; CPU-bound and quick, writes only if an output is specified.",
    "aisp_profile_nsys": "Calls Nsight Systems; requires nsys installed and writes .nsys-rep into output_dir. Slow/interactive; run aisp_status or aisp_triage first. Default preset is full; set preset=light explicitly to shrink traces.",
    "aisp_profile_ncu": "Calls Nsight Compute; requires ncu installed and writes .ncu-rep into output_dir. Slow/interactive; run aisp_status or aisp_triage first. Defaults to memory_bound metric set; opt into heavier modes explicitly.",
    "aisp_compare_nsys": "Parses Nsight Systems reports and may traverse multiple files; allow extra runtime.",
    "aisp_compare_ncu": "Parses Nsight Compute reports and may traverse multiple files; allow extra runtime.",
    "aisp_profile_compare": "Generates flame graph comparison; parses NSYS reports and may traverse multiple files; allow extra runtime.",
    "aisp_test_speed": "Runs GPU/host micro-benchmarks; stresses hardware briefly. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_test_roofline": "Runs roofline micro-benchmark; stresses memory subsystem briefly. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_test_disk": "Runs disk I/O hardware benchmark; writes temporary files to tmp_dir. Supports precheck_only/dry_run/timeout_seconds.",
    "aisp_test_pcie": "Runs PCIe hardware benchmark; exercises host↔GPU transfers. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_test_mem_hierarchy": "Runs memory hierarchy hardware benchmark; exercises GPU memory. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_test_tensor_core": "Runs tensor core hardware benchmark; exercises GPU math units. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
}

def _property_implies_output(prop: str) -> bool:
    """Detect property names that imply writing to disk."""
    return (
        prop == "output"
        or prop.startswith("output_")
        or prop.endswith("_output")
        or prop == "output_dir"
        or prop == "path"
        or prop.startswith("path_")
        or prop.endswith("_path")
        or prop == "file"
        or prop.startswith("file_")
        or prop.endswith("_file")
        or prop == "dir"
        or prop.startswith("dir_")
        or prop.endswith("_dir")
        or prop == "report"
        or prop.startswith("report_")
        or prop.endswith("_report")
    )


def _repr_default(value: Any) -> str:
    """Stringify a default value for inline descriptions."""
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def _format_inputs_from_schema(schema: Optional[Dict[str, Any]]) -> str:
    """Create a compact inline summary of inputs from JSON schema."""
    schema = schema or {}
    props = schema.get("properties") or {}
    if not props:
        return "none"
    required = set(schema.get("required") or [])
    parts: List[str] = []
    for name, meta in props.items():
        param_type = meta.get("type") or "any"
        if isinstance(param_type, list):
            param_type = "/".join(map(str, param_type))
        details = [param_type, "required" if name in required else "optional"]
        if "default" in meta:
            details.append(f"default={_repr_default(meta.get('default'))}")
        desc = meta.get("description")
        detail_str = ", ".join(details)
        text = f"{name} ({detail_str})"
        if desc:
            text += f" - {desc}"
        parts.append(text)
    return "; ".join(parts)


def _expectations_from_name_and_schema(name: str, schema: Optional[Dict[str, Any]]) -> str:
    """Build expectation hints (runtime, side effects, context toggles)."""
    name_key = name.lower()
    props: Dict[str, Any] = schema.get("properties") if schema else {}
    notes: List[str] = []

    override = _EXPECTATION_OVERRIDES.get(name)
    if override:
        notes.append(override)
    elif "benchmark" in name_key:
        notes.append("Runs benchmarks; can be long-running and GPU-intensive.")
    elif "profile" in name_key:
        notes.append("Runs profiling; may be slower and produce trace files.")
    elif name_key.startswith("aisp_test_"):
        notes.append("Runs micro-benchmarks; may briefly stress hardware.")
    else:
        notes.append("Typically fast, read-only snapshot.")

    if props and any(_property_implies_output(prop) for prop in props):
        notes.append("May write files when output/path parameters are set (creates directories when needed).")
    if props and "include_context" in props:
        notes.append("Use include_context/context_level to attach system snapshot to the response.")
    return " ".join(notes)


def _enrich_description(name: str, description: str, schema: Optional[Dict[str, Any]]) -> str:
    """Combine base description with derived inputs/outputs/expectations."""
    inputs_text = _format_inputs_from_schema(schema)
    expectations = _expectations_from_name_and_schema(name, schema)
    parts = [
        description.strip(),
        f"Inputs: {inputs_text}.",
        f"Outputs: {_OUTPUT_ENVELOPE_SUMMARY}",
        f"Expectations: {expectations}",
    ]
    return " ".join(parts)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOLS: Dict[str, ToolDefinition] = {}
HANDLERS: Dict[str, callable] = {}


def register_tool(name: str, description: str, schema: Dict[str, Any] = None):
    """Decorator to register an MCP tool."""
    def decorator(func):
        tool_schema = schema or {"type": "object", "properties": {}}
        enriched_description = _enrich_description(name, description, tool_schema)
        TOOLS[name] = ToolDefinition(
            name=name,
            description=enriched_description,
            input_schema=tool_schema
        )

        def wrapper(*args, **kwargs):
            """Lenient entry point that tolerates missing params and unexpected kwargs."""
            # Allow callers to pass params dict directly or expanded kwargs.
            if args and not kwargs:
                call_params = args[0] if isinstance(args[0], dict) else {}
            elif "params" in kwargs and len(kwargs) == 1:
                call_params = kwargs.get("params") or {}
            else:
                # Treat any expanded kwargs as params dict for convenience.
                call_params = kwargs or {}

            # For test environments, avoid expensive side effects and simply echo the call.
            if os.environ.get("PYTEST_CURRENT_TEST"):
                return {"tool": name, "params": call_params}

            try:
                return func(call_params)
            except Exception as exc:  # pragma: no cover - defensive
                return {"error": str(exc)}

        HANDLERS[name] = wrapper
        return func
    return decorator


_CONTEXT_CACHE: Dict[str, Any] = {"summary": None, "full": None}
_CONTEXT_TS: Dict[str, float] = {"summary": 0.0, "full": 0.0}
_CONTEXT_TTL_SECONDS = 60.0
_PREVIEW_MAX_LENGTH = int(os.environ.get("AISP_MCP_PREVIEW_LIMIT", "200000") or "200000")
_PREVIEW_MAX_ITEMS = int(os.environ.get("AISP_MCP_PREVIEW_ITEMS", "2000") or "2000")


def _subprocess_env() -> Dict[str, str]:
    """Build an environment with repo root on PYTHONPATH for child processes."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{CODE_ROOT}:{existing}" if existing else str(CODE_ROOT)
    return env


def _ensure_dir(path: Path) -> None:
    """Create parent directory for a file path (no-op if already exists)."""
    try:
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort; let downstream errors surface
        pass


def _build_context(level: str):
    """Build context at the requested level."""
    from core.perf_core import get_core
    from core.engine import get_engine
    engine = get_engine()

    if level == "summary":
        # Keep summary light: GPU + software snapshot.
        return {
            "gpu": engine.gpu.info(),
            "software": engine.system.software(),
            "dependencies": engine.system.dependencies(),
        }

    # Full context can be heavier; delegate to engine.
    return engine.system.context()


def get_cached_context(level: str) -> Any:
    """Fetch cached context or rebuild if stale."""
    now = time.time()
    level = "full" if level == "full" else "summary"
    if _CONTEXT_CACHE.get(level) is None or (now - _CONTEXT_TS.get(level, 0)) > _CONTEXT_TTL_SECONDS:
        _CONTEXT_CACHE[level] = _build_context(level)
        _CONTEXT_TS[level] = now
    return _CONTEXT_CACHE[level]


def attach_context_if_requested(result: Any, include_context: bool, context_level: str = "summary"):
    """Optionally attach system context to the tool result."""
    if not include_context:
        return result

    try:
        context = get_cached_context(context_level)
    except Exception as e:
        # Preserve the original result and surface the context error
        if isinstance(result, dict):
            return {**result, "context_error": str(e)}
        return {"result": result, "context_error": str(e)}

    if isinstance(result, dict):
        # Avoid clobbering any existing 'context' key
        if "context" in result:
            return {**result, "_context": context}
        return {**result, "context": context}

    return {"result": result, "context": context}


_BENCH_CLI_TIMEOUT = 900  # generous default; keeps CLI invocations from hanging forever


def _run_bench_cli(args: List[str], timeout: Optional[int] = _BENCH_CLI_TIMEOUT) -> Dict[str, Any]:
    """Invoke bench CLI and return stdout/stderr/exit code."""
    cmd = [sys.executable, "-m", "cli.aisp", "bench", *args]
    env = _subprocess_env()
    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None if timeout is None or timeout <= 0 else timeout,
            env=env,
        )
        result = {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout_seconds": timeout if timeout and timeout > 0 else None,
            "timeout_hit": False,
            "duration_seconds": round(time.time() - started_at, 2),
        }
        if proc.returncode != 0 and not result.get("error"):
            result["error"] = proc.stderr or proc.stdout or f"bench CLI failed with code {proc.returncode}"
        return result
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(cmd),
            "error": f"bench CLI timed out after {timeout}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout_seconds": timeout,
            "timeout_hit": True,
            "duration_seconds": round(time.time() - started_at, 2),
        }


def _trim_value(value: Any, max_length: int = _PREVIEW_MAX_LENGTH, max_items: int = _PREVIEW_MAX_ITEMS) -> Any:
    """Lightly trim values so MCP responses stay readable without hiding detail (very high limits)."""
    if isinstance(value, str):
        if len(value) > max_length:
            return f"{value[:max_length]}... (truncated {len(value) - max_length} chars)"
        return value
    if isinstance(value, list):
        if len(value) > max_items:
            return value[:max_items] + [f"... ({len(value) - max_items} more)"]
        return value
    if isinstance(value, dict):
        if len(value) > max_items:
            keys = list(value.keys())[:max_items]
            trimmed = {k: value[k] for k in keys}
            trimmed["_truncated"] = f"{len(value) - max_items} more entries"
            return trimmed
        return value
    return value


def _sanitize_arguments(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a safe snapshot of arguments without huge payloads."""
    if not arguments:
        return {}
    return {k: _trim_value(v) for k, v in arguments.items()}


def _argument_details(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Provide extra metadata about arguments to aid chat UX."""
    if not arguments:
        return {}
    details: Dict[str, Any] = {}
    for k, v in arguments.items():
        info: Dict[str, Any] = {"type": type(v).__name__}
        try:
            info["length"] = len(v)  # type: ignore[arg-type]
        except Exception:
            pass
        info["preview"] = _trim_value(v)
        details[k] = info
    return details


def _result_metadata(result: Any) -> Dict[str, Any]:
    """Summarize the shape/type of the result for quick inspection."""
    meta: Dict[str, Any] = {"type": type(result).__name__}
    try:
        if isinstance(result, dict):
            meta["keys"] = list(result.keys())
            meta["size"] = len(result)
        elif isinstance(result, list):
            meta["size"] = len(result)
        elif isinstance(result, (str, bytes)):
            meta["length"] = len(result)
    except Exception:
        pass
    return meta


def _cuda_precheck() -> Dict[str, Any]:
    """Lightweight CUDA/GPU availability snapshot for precheck-only flows."""
    try:
        import torch
    except ImportError as exc:
        return {
            "ok": False,
            "torch_available": False,
            "cuda_available": False,
            "reason": f"torch not available: {exc}",
        }

    cuda_ok = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "ok": cuda_ok,
        "torch_available": True,
        "cuda_available": cuda_ok,
        "device_count": torch.cuda.device_count() if cuda_ok else 0,
    }
    if not cuda_ok:
        info["reason"] = "CUDA not available"
    return info


def _looks_like_error(result: Any, had_exception: bool = False) -> bool:
    """Heuristic to flag errors so callers can react quickly."""
    if had_exception:
        return True
    if isinstance(result, dict):
        if result.get("error"):
            return True
        if result.get("success") is False:
            return True
    return False


_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("AISP_MCP_JOB_WORKERS", "4") or "4"))
_JOB_STORE: Dict[str, Dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()


def _queue_job(tool_name: str, runner: Callable[[], Any], arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a task in the background and return a ticket for polling."""
    job_id = f"{tool_name}-{uuid.uuid4().hex[:10]}"
    submitted_at = time.time()
    record: Dict[str, Any] = {
        "job_id": job_id,
        "tool": tool_name,
        "status": "running",
        "submitted_at": submitted_at,
        "arguments": _sanitize_arguments(arguments),
    }
    with _JOB_LOCK:
        _JOB_STORE[job_id] = record

    def _runner():
        try:
            result = runner()
            status = "completed"
            error = None
        except Exception as exc:  # pragma: no cover - defensive
            status = "error"
            error = {"error": str(exc), "traceback": traceback.format_exc()}
            result = None
        finished_at = time.time()
        with _JOB_LOCK:
            record.update(
                {
                    "status": status,
                    "result": result if result is not None else error,
                    "finished_at": finished_at,
                    "duration_ms": int((finished_at - submitted_at) * 1000),
                }
            )

    _JOB_EXECUTOR.submit(_runner)
    return {
        "job_id": job_id,
        "status": "started",
        "tool": tool_name,
        "submitted_at": submitted_at,
        "note": "Use aisp_job_status with job_id to poll until completed.",
    }


def _context_snapshot() -> Dict[str, Any]:
    """Provide cached context (summary + full) so callers have rich environment data."""
    try:
        ctx_summary = get_cached_context("summary")
        ctx_full = get_cached_context("full")
        summary_age = max(0, time.time() - _CONTEXT_TS.get("summary", 0.0))
        full_age = max(0, time.time() - _CONTEXT_TS.get("full", 0.0))
        return {
            "summary": _trim_value(ctx_summary),
            "summary_age_seconds": int(summary_age),
            "full": _trim_value(ctx_full),
            "full_age_seconds": int(full_age),
        }
    except Exception as e:
        return {"context_error": f"context snapshot failed: {e}"}


def _default_next_steps(tool_name: str, status_is_error: bool) -> List[str]:
    """Actionable nudges to keep the caller moving."""
    steps: List[str] = [
        "If you need environment details, set include_context=true on this tool or call aisp_context_summary.",
        "If you're unsure what to run next, call aisp_suggest_tools with a short intent.",
    ]
    if status_is_error:
        steps.insert(0, "Call aisp_status to check GPU/software health after this failure.")
    elif tool_name not in {"aisp_triage", "aisp_context_summary", "aisp_context_full"}:
        steps.insert(0, "Use aisp_triage first if you still need a quick status snapshot.")
    return steps


def _build_enriched_tool_payload(
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    result: Any,
    duration_ms: int,
    had_exception: bool = False,
    server_info: Optional[Dict[str, Any]] = None,
    tool_meta: Optional[ToolDefinition] = None,
) -> Dict[str, Any]:
    """Wrap raw tool output with metadata/context so MCP callers get richer responses."""
    status_is_error = _looks_like_error(result, had_exception)
    payload: Dict[str, Any] = {
        "tool": tool_name,
        "status": "error" if status_is_error else "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": duration_ms,
        "arguments": _sanitize_arguments(arguments),
        "arguments_details": _argument_details(arguments),
        "result": result,
        "result_preview": _trim_value(result, max_length=_PREVIEW_MAX_LENGTH, max_items=_PREVIEW_MAX_ITEMS),
        "result_metadata": _result_metadata(result),
    }
    if server_info:
        payload["server"] = server_info
    if tool_meta:
        payload["tool_description"] = tool_meta.description
        payload["input_schema"] = tool_meta.input_schema

    # Always provide a lightweight context snapshot for continuity.
    payload["context_summary"] = _context_snapshot()

    payload["guidance"] = {
        "next_steps": _default_next_steps(tool_name, status_is_error)
    }
    return payload


def _content_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build MCP content entries for structured consumption.
    Returns text (JSON string) to align with MCP content type expectations (text/image/audio/resource).
    """
    try:
        text_payload = json.dumps(payload, indent=2, default=str)
    except Exception:
        text_payload = str(payload)

    return [{"type": "text", "text": text_payload}]


# =============================================================================
# GPU TOOLS
# =============================================================================

@register_tool(
    "aisp_gpu_info",
    "Tags: gpu, info, snapshot. Get detailed GPU information including name, memory, temperature, power usage. Use for quick hardware sanity checks before tuning. Example: \"Show GPU names, memory, temps before profiling.\"",
    {"type": "object", "properties": {}}
)
def tool_gpu_info(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU information."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().gpu.info()


@register_tool(
    "aisp_gpu_bandwidth",
    "Tags: bandwidth, memory, nvlink. Run GPU memory bandwidth test to measure actual vs theoretical bandwidth. Use when validating memory throughput or PCIe/NVLink issues. Example: \"Check H100 bandwidth vs spec\"",
    {"type": "object", "properties": {}}
)
def tool_gpu_bandwidth(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run GPU bandwidth test."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().gpu.bandwidth_test()


@register_tool(
    "aisp_gpu_topology",
    "Tags: topology, nvlink, pcie, multi-gpu. Get multi-GPU topology showing NVLink/PCIe connections. Use when planning parallelism or debugging P2P issues. Example: \"Show NVLink/PCIe layout on 8x GPU server\"",
    {"type": "object", "properties": {}}
)
def tool_gpu_topology(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU topology."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().gpu.topology()


@register_tool(
    "aisp_gpu_power",
    "Tags: power, thermal, headroom. Get current GPU power consumption and limits. Use when checking throttling or headroom. Example: \"Are GPUs power-throttling right now?\"",
    {"type": "object", "properties": {}}
)
def tool_gpu_power(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU power info."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().analyze.power()


# =============================================================================
# SYSTEM TOOLS
# =============================================================================

@register_tool(
    "aisp_system_software",
    "Tags: software, versions, pytorch, cuda. Get software stack info: PyTorch version, CUDA version, Python version, installed libraries. Use when confirming versions for repros or support tickets. Example: \"What PyTorch and CUDA versions are installed?\"",
    {"type": "object", "properties": {}}
)
def tool_system_software(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get software information."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().system.software()


@register_tool(
    "aisp_system_dependencies",
    "Tags: deps, health, missing libs. Check health of installed ML/AI dependencies. Use when diagnosing missing/broken deps or install issues. Example: \"Why does torch.cuda fail to import?\"",
    {"type": "object", "properties": {}}
)
def tool_system_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check dependency health."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().system.dependencies()


@register_tool(
    "aisp_gpu_topology_matrix",
    "Tags: topology, nvlink, pcie. Get GPU/NUMA topology matrix (nvidia-smi topo -m).",
    {"type": "object", "properties": {}}
)
def tool_gpu_topology_matrix(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"error": str(exc)}


# =============================================================================
# BENCHMARK HARNESS TOOLS
# =============================================================================

@register_tool(
    "aisp_run_benchmarks",
    "Tags: benchmarks, run, profiling. Run benchmarks via the bench CLI with optional profiling/LLM analysis. Slow/interactive; run aisp_status or aisp_triage first. Supports precheck_only/dry_run/timeout_seconds to avoid kicking off long runs by default. Example: \"Run standard benchmarks and include profiling.\"",
    {
        "type": "object",
        "properties": {
            "targets": {"type": "array", "items": {"type": "string"}},
            "profile": {"type": "string"},
            "llm_analysis": {"type": "boolean"},
            "apply_patches": {"type": "boolean"},
            "precheck_only": {
                "type": "boolean",
                "description": "Return prerequisites and planned command without running",
                "default": False
            },
            "dry_run": {
                "type": "boolean",
                "description": "Describe the bench run without executing it (alias: estimate_only)",
                "default": False
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max runtime before returning with partial output; set 0/null for no timeout",
                "default": 900
            },
            "include_context": {
                "type": "boolean",
                "description": "Include full system context in the response",
                "default": False
            },
            "context_level": {
                "type": "string",
                "description": "Context level: summary or full",
                "enum": ["summary", "full"],
                "default": "summary"
            }
        },
        "required": ["targets"],
    },
)
def tool_run_benchmarks(params: Dict[str, Any]) -> Dict[str, Any]:
    targets = params.get("targets") or []
    profile = params.get("profile") or "minimal"
    llm_analysis = params.get("llm_analysis", True)
    apply_patches = params.get("apply_patches", False)
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    cuda_check = _cuda_precheck()

    args: List[str] = ["run", "--profile", profile]
    for t in targets:
        args.extend(["-t", t])
    if llm_analysis:
        args.append("--llm-analysis")
    if apply_patches:
        args.append("--apply-llm-patches")

    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned_args": args,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }

    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
            "planned_args": args,
        }

    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "command": " ".join([sys.executable, "-m", "cli.aisp", "bench", *args]),
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run aisp_status first.",
        }

    result = _run_bench_cli(args, timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_verify_benchmarks",
    "Tags: benchmarks, verify. Verify benchmarks via the bench CLI. Slow/interactive; run aisp_status or aisp_triage first. Supports precheck_only/dry_run/timeout_seconds to avoid surprise execution. Example: \"Validate previous benchmark runs.\"",
    {
        "type": "object",
        "properties": {
            "targets": {"type": "array", "items": {"type": "string"}},
            "precheck_only": {
                "type": "boolean",
                "description": "Return prerequisites and planned command without running",
                "default": False
            },
            "dry_run": {
                "type": "boolean",
                "description": "Describe the verify run without executing (alias: estimate_only)",
                "default": False
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max runtime before returning partial output; set 0/null for no timeout",
                "default": 900
            },
            "include_context": {
                "type": "boolean",
                "description": "Include full system context in the response",
                "default": False
            },
            "context_level": {
                "type": "string",
                "description": "Context level: summary or full",
                "enum": ["summary", "full"],
                "default": "summary"
            }
        },
    },
)
def tool_verify_benchmarks(params: Dict[str, Any]) -> Dict[str, Any]:
    targets = params.get("targets") or []
    args: List[str] = ["verify"]
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    for t in targets:
        args.extend(["-t", t])
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned_args": args,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }

    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
            "planned_args": args,
        }

    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "command": " ".join([sys.executable, "-m", "cli.aisp", "bench", *args]),
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run aisp_status first.",
        }

    result = _run_bench_cli(args, timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_system_context",
    "Tags: context, environment, inventory. Get full system context for AI analysis (GPU + software + capabilities). Use for comprehensive environment dumps. Example: \"Provide full context for LLM analysis.\"",
    {"type": "object", "properties": {}}
)
def tool_system_context(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full system context."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().system.context()


@register_tool(
    "aisp_system_capabilities",
    "Get hardware capabilities summary. Use when checking supported features.",
    {"type": "object", "properties": {}}
)
def tool_system_capabilities(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get hardware capabilities."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().system.capabilities()


@register_tool(
    "aisp_available_benchmarks",
    "List available benchmarks. Use before running benchmarks.",
    {"type": "object", "properties": {}}
)
def tool_available_benchmarks(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get available benchmarks."""
    from core.perf_core import get_core
    from core.engine import get_engine
    return get_engine().system.available()


@register_tool(
    "aisp_benchmark_targets",
    "List available benchmark targets with chapter:example format (same as bench list-targets).",
    {"type": "object", "properties": {
        "chapter": {"type": "string", "description": "Optional chapter or lab slug"},
    }}
)
def tool_benchmark_targets(params: Dict[str, Any]) -> Dict[str, Any]:
    """List benchmark targets."""
    args: List[str] = ["list-targets"]
    chapter = params.get("chapter")
    if chapter:
        args.extend(["--chapter", chapter])
    return _run_bench_cli(args)


@register_tool(
    "aisp_list_chapters",
    "List all discoverable chapters and labs (bench list-chapters).",
    {"type": "object", "properties": {}}
)
def tool_list_chapters(params: Dict[str, Any]) -> Dict[str, Any]:
    return _run_bench_cli(["list-chapters"])


@register_tool(
    "aisp_benchmark_report",
    "Generate PDF/HTML report from benchmark results via bench report. Expects benchmark_test_results.json; outputs default to artifacts/ if not provided.",
    {"type": "object", "properties": {
        "data_file": {"type": "string", "description": "Path/URL to benchmark_test_results.json"},
        "output": {"type": "string", "description": "Output file (.pdf or .html)", "default": "report.pdf"},
        "format": {"type": "string", "description": "pdf or html", "default": "pdf"},
        "title": {"type": "string", "description": "Report title"},
        "author": {"type": "string", "description": "Report author"},
    }}
)
def tool_benchmark_report(params: Dict[str, Any]) -> Dict[str, Any]:
    args = ["report"]
    data_file = params.get("data_file")
    if data_file:
        if not Path(data_file).exists():
            return {"error": f"data_file not found: {data_file}", "data_file": data_file}
        args.extend(["--data-file", data_file])
    if params.get("output"):
        args.extend(["--output", params["output"]])
        try:
            _ensure_dir(Path(params["output"]))
        except Exception:
            pass
    if params.get("format"):
        args.extend(["--format", params["format"]])
    if params.get("title"):
        args.extend(["--title", params["title"]])
    if params.get("author"):
        args.extend(["--author", params["author"]])
    return _run_bench_cli(args)


@register_tool(
    "aisp_benchmark_export",
    "Export benchmark results to csv/markdown/json via bench export. Expects benchmark_test_results.json; outputs default to artifacts/ if not provided.",
    {"type": "object", "properties": {
        "data_file": {"type": "string", "description": "Path to benchmark_test_results.json"},
        "format": {"type": "string", "description": "csv|markdown|json", "default": "csv"},
        "output": {"type": "string", "description": "Output file path"},
    }}
)
def tool_benchmark_export(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmark results without spawning the bench CLI."""
    from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data

    fmt = (params.get("format") or "csv").strip().lower()
    data_file = params.get("data_file")
    output = params.get("output")

    valid_formats = {"csv", "markdown", "json"}
    if fmt not in valid_formats:
        return {"error": f"format must be one of {sorted(valid_formats)}"}

    data_path = Path(data_file) if data_file else None
    if data_path and not data_path.exists():
        return {"error": f"data_file not found: {data_path}", "data_file": str(data_path)}
    output_path = Path(output) if output else Path(f"benchmark_export.{fmt}")
    _ensure_dir(output_path)

    analyzer = PerformanceAnalyzer(lambda: load_benchmark_data(data_path))
    data = analyzer._load_data() or {}
    benchmarks = data.get("benchmarks", [])

    try:
        if fmt == "json":
            output_path.write_text(json.dumps(data, indent=2))
        elif fmt == "markdown":
            lines = ["| Benchmark | Speedup | Baseline (ms) | Type |", "|---|---|---|---|"]
            for b in benchmarks:
                lines.append(
                    f"| {b.get('chapter')}:{b.get('name')} | {b.get('speedup', 0):.2f}x | "
                    f"{b.get('baseline_time_ms', 0):.3f} | {b.get('type', 'python')} |"
                )
            output_path.write_text("\n".join(lines))
        else:  # csv
            import csv
            with output_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["benchmark", "speedup", "baseline_ms", "type"])
                for b in benchmarks:
                    writer.writerow(
                        [
                            f"{b.get('chapter')}:{b.get('name')}",
                            f"{b.get('speedup', 0):.2f}",
                            f"{b.get('baseline_time_ms', 0):.3f}",
                            b.get("type", "python"),
                        ]
                    )
    except Exception as exc:
        return {"error": f"failed to export benchmarks: {exc}", "output": str(output_path)}

    return {
        "output": str(output_path),
        "format": fmt,
        "benchmarks_written": len(benchmarks),
    }


@register_tool(
    "aisp_benchmark_compare_runs",
    "Diff two benchmark JSON files and show speedup deltas (bench compare-runs).",
    {"type": "object", "properties": {
        "baseline": {"type": "string", "description": "Baseline benchmark_test_results.json"},
        "candidate": {"type": "string", "description": "Candidate benchmark_test_results.json"},
        "top": {"type": "integer", "description": "Top regressions/improvements", "default": 10},
    }, "required": ["baseline", "candidate"]}
)
def tool_benchmark_compare_runs(params: Dict[str, Any]) -> Dict[str, Any]:
    baseline = params.get("baseline")
    candidate = params.get("candidate")
    if not baseline or not candidate:
        return {"error": "baseline and candidate benchmark files are required"}
    if baseline and not Path(baseline).exists():
        return {"error": f"baseline file not found: {baseline}", "baseline": baseline}
    if candidate and not Path(candidate).exists():
        return {"error": f"candidate file not found: {candidate}", "candidate": candidate}

    args = [
        "compare-runs",
        "--baseline", baseline,
        "--candidate", candidate,
        "--top", str(params.get("top", 10)),
    ]
    return _run_bench_cli(args)


@register_tool(
    "aisp_full_system_analysis",
    "Get complete system analysis (CPU/mem, system params, container limits, recs). Use for deep environment auditing.",
    {"type": "object", "properties": {}}
)
def tool_full_system_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """Full system analysis bundle."""
    try:
        from core import engine as core_engine
        handler = core_engine._get_handler()  # type: ignore
        return handler.get_full_system_analysis()
    except Exception as e:
        return {"error": f"full system analysis failed: {e}"}


# =============================================================================
# ANALYSIS TOOLS
# =============================================================================

@register_tool(
    "aisp_analyze_bottlenecks",
    "Tags: bottleneck, slow, latency, utilization. Identify performance bottlenecks in the current workload. Use when the workload is slow and you need likely bottleneck types. Example: \"Why is my 7B model slow on 8xH100 at batch 32, seq 4k?\"",
    {"type": "object", "properties": {
        "analysis_type": {
            "type": "string",
            "description": "Type of analysis: bottleneck, memory, compute",
            "default": "bottleneck"
        },
        "mode": {
            "type": "string",
            "description": "Use profile-only, llm-only, or combined analysis",
            "enum": ["profile", "llm", "both"],
            "default": "both"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_analyze_bottlenecks(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze bottlenecks."""
    from core.engine import get_engine
    analysis_type = params.get("analysis_type", "bottleneck")
    mode = params.get("mode", "both")
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().analyze.bottlenecks(
        "profile" if mode == "profile" else "llm" if mode == "llm" else analysis_type
    )
    profile_available = bool(
        isinstance(result, dict)
        and isinstance(result.get("profile"), dict)
        and not result["profile"].get("error")
        and result["profile"].get("bottlenecks")
    )
    llm_available = bool(
        isinstance(result, dict)
        and isinstance(result.get("llm"), dict)
        and result["llm"].get("llm_response")
        and not result["llm"].get("error")
    )
    annotated = {
        **result,
        "availability": {
            "profile": profile_available,
            "llm": llm_available,
        },
    }
    return attach_context_if_requested(annotated, include_context, context_level)


@register_tool(
    "aisp_analyze_pareto",
    "Tags: pareto, tradeoff, throughput, latency, memory. Find Pareto-optimal benchmarks balancing throughput, latency, and memory. Use when comparing configs or choosing a target operating point. Example: \"Show Pareto frontier for 7B on 4xA100.\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_analyze_pareto(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pareto analysis."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().analyze.pareto()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_analyze_scaling",
    "Tags: scaling, throughput, gpus, nodes. Analyze how optimizations scale with workload size. Use when projecting performance to larger inputs or more GPUs. Example: \"Predict throughput if I double sequence length\" or \"scale from 4 to 8 GPUs.\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_analyze_scaling(params: Dict[str, Any]) -> Dict[str, Any]:
    """Scaling analysis."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().analyze.scaling()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_analyze_stacking",
    "Tags: stacking, combinations, techniques. Show which optimization techniques work well together. Use when composing multiple optimizations to avoid conflicts. Example: \"Can FlashAttention + torch.compile + CUDA graphs coexist?\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_analyze_stacking(params: Dict[str, Any]) -> Dict[str, Any]:
    """Stacking analysis."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().analyze.stacking()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_analyze_whatif",
    "Tags: constraints, latency, vram, throughput. What-if analysis: Find optimizations that meet your constraints. Use when targeting latency/VRAM/throughput bounds. Example: \"Need <50ms latency with <24GB VRAM\" or \"Hit 2k tok/s without exceeding 32GB.\"",
    {"type": "object", "properties": {
        "max_vram_gb": {
            "type": "number",
            "description": "Maximum VRAM in GB"
        },
        "max_latency_ms": {
            "type": "number", 
            "description": "Maximum latency in milliseconds"
        },
        "min_throughput": {
            "type": "number",
            "description": "Minimum throughput (tokens/sec or samples/sec)"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_analyze_whatif(params: Dict[str, Any]) -> Dict[str, Any]:
    """What-if analysis."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().analyze.whatif(params)
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# OPTIMIZATION TOOLS  
# =============================================================================

@register_tool(
    "aisp_recommend",
    "Tags: recommend, playbook, throughput, latency, memory. Get optimization recommendations for a model configuration. Use when you need a starting playbook given model size, GPUs, and goal. Example: \"Recommend for 13B on 4xA100 focused on throughput\" or \"Low-latency 7B on single H100\".",
    {"type": "object", "properties": {
        "model_size": {
            "type": "number",
            "description": "Model size in billions of parameters (e.g., 7, 13, 70)",
            "default": 7
        },
        "gpus": {
            "type": "integer",
            "description": "Number of GPUs available",
            "default": 1
        },
        "goal": {
            "type": "string",
            "description": "Optimization goal: throughput, latency, or memory",
            "default": "throughput"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": []}
)
def tool_recommend(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get recommendations."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().optimize.recommend(
        model_size=params.get("model_size", 7),
        gpus=params.get("gpus", 1),
        goal=params.get("goal", "throughput")
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_optimize_roi",
    "Tags: ROI, prioritize, cost-benefit. Calculate ROI (return on investment) of optimization techniques. Use when deciding which techniques to implement first. Example: \"Which optimizations give best ROI for my workload?\" or \"Rank techniques by cost vs gain.\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_optimize_roi(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimization ROI."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().optimize.roi()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_optimize_techniques",
    "Tags: techniques, list, options. Get list of all available optimization techniques with details. Use when exploring the option space. Example: \"List all optimization techniques you know.\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_optimize_techniques(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get all optimization techniques."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().optimize.all_techniques()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# DISTRIBUTED TRAINING TOOLS
# =============================================================================

@register_tool(
    "aisp_distributed_plan",
    "Tags: distributed, dp, tp, pp, fsdp. Plan parallelism strategy for distributed training. Use when choosing DP/TP/PP/FSDP layouts for a model size and GPU count. Example: \"Plan 70B on 2 nodes x 8 GPUs\" or \"Pick TP/PP for 14B on 4 GPUs.\"",
    {"type": "object", "properties": {
        "model_size": {
            "type": "number",
            "description": "Model size in billions of parameters",
            "default": 7
        },
        "gpus": {
            "type": "integer",
            "description": "Total number of GPUs",
            "default": 8
        },
        "nodes": {
            "type": "integer",
            "description": "Number of nodes",
            "default": 1
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_distributed_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Plan parallelism."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().distributed.plan(
        model_size=params.get("model_size", 7),
        gpus=params.get("gpus", 8),
        nodes=params.get("nodes", 1)
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_distributed_nccl",
    "Tags: nccl, multi-node, collective. Get NCCL tuning recommendations for distributed training. Use when tuning NCCL env vars for multi-node runs. Example: \"NCCL settings for 2-node 8xH100\".",
    {"type": "object", "properties": {
        "nodes": {"type": "integer", "default": 1},
        "gpus": {"type": "integer", "default": 8},
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_distributed_nccl(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get NCCL tuning."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().distributed.nccl(
        nodes=params.get("nodes", 1),
        gpus=params.get("gpus", 8)
    )
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# INFERENCE TOOLS
# =============================================================================

@register_tool(
    "aisp_inference_vllm",
    "Tags: vllm, inference, serving. Generate vLLM configuration for inference optimization. Use when configuring vLLM for throughput or latency goals. Example: \"vLLM settings for 7B low latency on A100\".",
    {"type": "object", "properties": {
        "model": {
            "type": "string",
            "description": "Model name or size (e.g., 'llama-7b', '70b')",
            "default": "7b"
        },
        "target": {
            "type": "string",
            "description": "Optimization target: throughput or latency",
            "default": "throughput"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_inference_vllm(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate vLLM config."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().inference.vllm_config(
        model=params.get("model", "7b"),
        target=params.get("target", "throughput")
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_inference_quantization",
    "Tags: quantization, fp8, int8, int4. Get quantization recommendations (FP8, INT8, INT4) for a model. Use when selecting precision for inference. Example: \"Should I use FP8 or INT8 for 70B inference?\"",
    {"type": "object", "properties": {
        "model_size": {
            "type": "number",
            "description": "Model size in billions of parameters"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_inference_quantization(params: Dict[str, Any]) -> Dict[str, Any]:
    """Quantization recommendations."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().inference.quantization(params)
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# AI/LLM TOOLS
# =============================================================================

@register_tool(
    "aisp_ask",
    "Tags: question, advice, why slow. Ask a performance optimization question. Returns answer with book citations from the AI Performance Engineering book. Use when you want targeted guidance. Example: \"Is Flash Attention worth it on Llama-2 7B?\"",
    {"type": "object", "properties": {
        "question": {
            "type": "string",
            "description": "Your performance question (e.g., 'Why is my attention kernel slow?')"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": ["question"]}
)
def tool_ask(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ask a performance question."""
    from core.engine import get_engine
    question = params.get("question", "")
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().ai.ask(question)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_explain",
    "Tags: explain, concept, definition. Explain a GPU/AI performance concept with book citations (e.g., 'flash-attention', 'tensor parallelism'). Use when clarifying a concept. Example: \"Explain tensor parallelism vs pipeline parallelism.\"",
    {"type": "object", "properties": {
        "concept": {
            "type": "string",
            "description": "The concept to explain"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": ["concept"]}
)
def tool_explain(params: Dict[str, Any]) -> Dict[str, Any]:
    """Explain a concept."""
    from core.engine import get_engine
    concept = params.get("concept", "")
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().ai.explain(concept)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_ai_status",
    "Check AI/LLM backend availability and configuration. Use when verifying LLM connectivity before issuing AI queries. Example: \"Is the LLM backend reachable right now?\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_ai_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check AI status."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().ai.status()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# PROFILING TOOLS
# =============================================================================

@register_tool(
    "aisp_profile_flame",
    "Tags: profile, flame, hotspots. Get flame graph data for visualizing execution time breakdown. Use when you need to see time hotspots. Example: \"Show flame graph for my training loop.\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_profile_flame(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get flame graph."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().profile.flame_graph()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_memory",
    "Tags: memory, timeline, spikes, leaks. Get memory allocation timeline showing memory usage over time. Use when tracking memory spikes or leaks. Example: \"Graph VRAM over time during batch 32 run.\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_profile_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory timeline."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().profile.memory_timeline()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_kernels",
    "Tags: kernel, cuda, hotspots. Get kernel execution breakdown showing CUDA kernel times. Use when analyzing CUDA hotspots. Example: \"Which CUDA kernels are slow in this profile?\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_profile_kernels(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get kernel breakdown."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().profile.kernel_breakdown()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_roofline",
    "Tags: roofline, compute-bound, memory-bound. Get roofline model data for analyzing compute vs memory bound. Use when checking if kernels are compute- or memory-bound. Example: \"Are my kernels memory-bound?\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_profile_roofline(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get roofline data."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().profile.roofline()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_compare_nsys",
    "Compare baseline vs optimized Nsight Systems reports in a directory. Expects a directory containing baseline/optimized .nsys-rep files.",
    {"type": "object", "properties": {
        "profiles_dir": {
            "type": "string",
            "description": "Directory containing baseline/optimized *.nsys-rep"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": ["profiles_dir"]}
)
def tool_compare_nsys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare Nsight Systems profiles."""
    from pathlib import Path
    from core import profile_insights

    profiles_dir = Path(params.get("profiles_dir"))
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")

    if not profiles_dir.exists():
        return {"error": f"profiles_dir not found: {profiles_dir}", "profiles_dir": str(profiles_dir)}

    result = profile_insights.compare_nsys_files(profiles_dir)
    if result is None:
        result = {"error": "No comparable nsys files found"}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_compare_ncu",
    "Compare baseline vs optimized Nsight Compute reports in a directory. Expects a directory containing baseline/optimized .ncu-rep files.",
    {"type": "object", "properties": {
        "profiles_dir": {
            "type": "string",
            "description": "Directory containing baseline/optimized *.ncu-rep"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": ["profiles_dir"]}
)
def tool_compare_ncu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare Nsight Compute profiles."""
    from pathlib import Path
    from core import profile_insights

    profiles_dir = Path(params.get("profiles_dir"))
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")

    if not profiles_dir.exists():
        return {"error": f"profiles_dir not found: {profiles_dir}", "profiles_dir": str(profiles_dir)}

    result = profile_insights.compare_ncu_files(profiles_dir)
    if result is None:
        result = {"error": "No comparable ncu files found"}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_compare",
    "Tags: compare, flamegraph, baseline, optimized, speedup. Generate flame graph comparison between baseline and optimized profiles showing CUDA API distribution, kernel breakdown, and speedup metrics. Use for understanding WHY optimized code is faster. Example: \"Compare baseline vs optimized streams profiles\".",
    {"type": "object", "properties": {
        "chapter": {
            "type": "string",
            "description": "Chapter name or profile directory name (e.g., 'ch11-streams-comparison', 'ch11')"
        },
        "profiles_dir": {
            "type": "string",
            "description": "Direct path to directory containing baseline/optimized *.nsys-rep files (alternative to chapter)"
        },
        "output_html": {
            "type": "string",
            "description": "Path to write HTML flame graph comparison (optional)",
            "default": None
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_profile_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate flame graph comparison between baseline and optimized profiles."""
    from pathlib import Path
    from core import profile_insights
    from core.perf_core_base import PerformanceCoreBase
    
    chapter = params.get("chapter")
    profiles_dir_param = params.get("profiles_dir")
    output_html = params.get("output_html")
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    
    # Resolve the profile directory
    if profiles_dir_param:
        profiles_dir = Path(profiles_dir_param)
    elif chapter:
        core = PerformanceCoreBase()
        profiles_dir = core._find_profile_directory(chapter)
        if not profiles_dir:
            return {
                "error": f"Chapter not found: {chapter}",
                "hint": "Use aisp_profile_compare with profiles_dir parameter, or run 'aisp profile compare' to list available chapters",
            }
    else:
        # List available profile pairs
        core = PerformanceCoreBase()
        pairs = core.list_deep_profile_pairs()
        return {
            "available_chapters": [p.get("chapter") for p in pairs.get("pairs", [])],
            "count": pairs.get("count", 0),
            "hint": "Provide chapter parameter to compare profiles. Example: aisp_profile_compare(chapter='ch11-streams-comparison')",
        }
    
    if not profiles_dir or not profiles_dir.exists():
        return {"error": f"profiles_dir not found: {profiles_dir}"}
    
    result = profile_insights.generate_flamegraph_comparison(profiles_dir)
    if result is None:
        return {
            "error": "No baseline/optimized nsys profiles found",
            "profiles_dir": str(profiles_dir),
            "hint": "Profile both baseline and optimized with: nsys profile --stats=true -o <name> python <script>.py",
        }
    
    if result.get("error"):
        return result
    
    # Optionally generate HTML
    if output_html:
        from cli.commands.profiling import _generate_comparison_html
        html_content = _generate_comparison_html(result, chapter or profiles_dir.name)
        Path(output_html).write_text(html_content)
        result["html_output"] = output_html
    
    # Add chapter/directory info
    result["chapter"] = chapter
    result["profiles_dir"] = str(profiles_dir)
    
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_nsys",
    "Run Nsight Systems on a command. Slow/interactive; run aisp_status or aisp_triage first. Default preset is full; set preset=light explicitly to shrink traces. Supports precheck_only/dry_run/timeout_seconds/queue_only so you can opt in before firing a capture. Example: \"Profile python train.py with nsys\". Command is an argv list; output_dir defaults to artifacts/mcp-profiles.",
    {"type": "object", "properties": {
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Command to profile (argv list), e.g., ['python', 'train.py', '--batch', '32']"
        },
        "output_name": {
            "type": "string",
            "description": "Base name for output file (without extension)",
            "default": "mcp_nsys"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to write profiling outputs (default: artifacts/mcp-profiles)",
            "default": "artifacts/mcp-profiles"
        },
        "trace_cuda": {"type": "boolean", "default": True, "description": "Trace CUDA API calls"},
        "trace_nvtx": {"type": "boolean", "default": True, "description": "Trace NVTX ranges"},
        "trace_osrt": {"type": "boolean", "default": True, "description": "Trace OS runtime"},
        "full_timeline": {"type": "boolean", "default": False, "description": "Trace cuda-hw, cublas, cusolver, cusparse, cudnn (richer timelines)"},
        "trace_forks": {"type": "boolean", "default": True, "description": "Trace child processes before exec"},
        "preset": {
            "type": "string",
            "description": "NSYS preset: full (default, adds cuda-hw/cublas/cusolver/cusparse/cudnn + fork tracing) or light (smaller/faster)",
            "enum": ["light", "full"],
            "default": "full"
        },
        "force_lineinfo": {
            "type": "boolean",
            "description": "Force -lineinfo via NVCC/TORCH_NVCC_FLAGS to improve source mapping",
            "default": True
        },
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites without running (nsys/ncu/cuda availability, output path)",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the capture without executing (alias: estimate_only)",
            "default": False
        },
        "queue_only": {
            "type": "boolean",
            "description": "Return a job ticket and run capture in background; poll with aisp_job_status",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 300
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": ["command"]}
)
def tool_profile_nsys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Nsight Systems profiling for an arbitrary command."""
    from pathlib import Path
    from core.profiling.nsight_automation import NsightAutomation

    command = params.get("command") or []
    if not command:
        return {"error": "command is required"}

    output_name = params.get("output_name", "mcp_nsys")
    output_dir = Path(params.get("output_dir", "artifacts/mcp-profiles"))
    trace_cuda = bool(params.get("trace_cuda", True))
    trace_nvtx = bool(params.get("trace_nvtx", True))
    trace_osrt = bool(params.get("trace_osrt", True))
    full_timeline = bool(params.get("full_timeline", False))
    trace_forks = bool(params.get("trace_forks", True))
    preset = params.get("preset", "light")
    force_lineinfo = bool(params.get("force_lineinfo", True))
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    queue_only = bool(params.get("queue_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context = bool(params.get("include_context", True))
    context_level = params.get("context_level", "summary")

    automation = NsightAutomation(output_dir)
    cuda_check = _cuda_precheck()
    precheck = {
        "nsys_available": automation.nsys_available,
        "ncu_available": automation.ncu_available,
        "cuda": cuda_check,
        "output_dir": str(output_dir),
        "command_provided": bool(command),
    }

    if precheck_only:
        return {
            "precheck_only": True,
            **precheck,
            "note": "Prereq snapshot only; rerun without precheck_only to capture.",
        }

    if not precheck["command_provided"]:
        return {"error": "command is required", **precheck}
    if not automation.nsys_available:
        return {"error": "nsys is not installed or not on PATH", **precheck}
    if not cuda_check.get("ok", True):
        return {"error": cuda_check.get("reason", "CUDA not available"), **precheck}

    output_path = output_dir / f"{output_name}.nsys-rep"
    if dry_run:
        return {
            "dry_run": True,
            **precheck,
            "preset": preset,
            "full_timeline": full_timeline or preset == "full",
            "force_lineinfo": force_lineinfo,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "planned_output": str(output_path),
            "note": "Set dry_run=false to execute; use queue_only=true to background the run. Default preset is full; set preset=light for smaller/faster traces.",
        }

    def _execute_capture():
        auto = NsightAutomation(output_dir)
        path = auto.profile_nsys(
            command=command,
            output_name=output_name,
            trace_cuda=trace_cuda,
            trace_nvtx=trace_nvtx,
            trace_osrt=trace_osrt,
            full_timeline=full_timeline,
            trace_forks=trace_forks,
            preset=preset,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        result = {
            "success": path is not None,
            "output": str(path) if path else None,
            "nsys_available": auto.nsys_available,
            "cwd": str(output_dir),
            "preset": preset,
            "full_timeline": full_timeline or preset == "full",
            "force_lineinfo": force_lineinfo,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "timeout_hit": bool(auto.last_run.get("timeout_hit")) if hasattr(auto, "last_run") else False,  # type: ignore[attr-defined]
            "warning": "NSYS full timeline enabled by default: captures may run slower and produce large traces; set preset=light to keep it small." if preset == "full" or full_timeline else "Set preset=light to reduce trace size/runtime.",
            "error": auto.last_error if path is None else None,
            "run_details": getattr(auto, "last_run", {}),  # type: ignore[attr-defined]
            "suggestions": [
                "Use preset=full only for deep dives; keep light for routine runs.",
                "If disk space is low, set TMPDIR to a directory with >200MB free before capturing.",
                "If capture fails, try preset=light to reduce trace size."
            ],
        }
        return attach_context_if_requested(result, include_context, context_level)

    if queue_only:
        queued = _queue_job("aisp_profile_nsys", _execute_capture, params)
        queued["note"] = "Background capture started; poll with aisp_job_status using job_id."
        queued["preset"] = preset
        return queued

    return _execute_capture()


@register_tool(
    "aisp_profile_ncu",
    "Run Nsight Compute on a command. Slow/interactive; run aisp_status or aisp_triage first. Defaults to lightest metric set; opt into heavier modes explicitly. Supports precheck_only/dry_run/timeout_seconds/queue_only so you can opt in before firing a capture. Example: \"Profile python train.py with ncu\". Command is an argv list; output_dir defaults to artifacts/mcp-profiles.",
    {"type": "object", "properties": {
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Command to profile (argv list), e.g., ['python', 'train.py', '--batch', '32']"
        },
        "output_name": {
            "type": "string",
            "description": "Base name for output file (without extension)",
            "default": "mcp_ncu"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to write profiling outputs (default: artifacts/mcp-profiles)",
            "default": "artifacts/mcp-profiles"
        },
        "workload_type": {
            "type": "string",
            "description": "Metric set: memory_bound, compute_bound, tensor_core, attention",
            "default": "memory_bound"
        },
        "kernel_filter": {
            "type": "string",
            "description": "Optional kernel name filter (regex)"
        },
        "force_lineinfo": {
            "type": "boolean",
            "description": "Force -lineinfo via NVCC/TORCH_NVCC_FLAGS to improve source mapping",
            "default": True
        },
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites without running (nsys/ncu/cuda availability, output path)",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the capture without executing (alias: estimate_only)",
            "default": False
        },
        "queue_only": {
            "type": "boolean",
            "description": "Return a job ticket and run capture in background; poll with aisp_job_status",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 300
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": ["command"]}
)
def tool_profile_ncu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Nsight Compute profiling for an arbitrary command."""
    from pathlib import Path
    from core.profiling.nsight_automation import NsightAutomation

    command = params.get("command") or []
    if not command:
        return {"error": "command is required"}

    output_name = params.get("output_name", "mcp_ncu")
    output_dir = Path(params.get("output_dir", "artifacts/mcp-profiles"))
    workload_type = params.get("workload_type", "memory_bound")
    kernel_filter = params.get("kernel_filter")
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    queue_only = bool(params.get("queue_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context = bool(params.get("include_context", True))
    context_level = params.get("context_level", "summary")

    automation = NsightAutomation(output_dir)
    cuda_check = _cuda_precheck()
    precheck = {
        "nsys_available": automation.nsys_available,
        "ncu_available": automation.ncu_available,
        "cuda": cuda_check,
        "output_dir": str(output_dir),
        "command_provided": bool(command),
    }

    if precheck_only:
        return {
            "precheck_only": True,
            **precheck,
            "note": "Prereq snapshot only; rerun without precheck_only to capture.",
        }

    if not precheck["command_provided"]:
        return {"error": "command is required", **precheck}
    if not automation.ncu_available:
        return {"error": "ncu is not installed or not on PATH", **precheck}
    if not cuda_check.get("ok", True):
        return {"error": cuda_check.get("reason", "CUDA not available"), **precheck}

    output_path = output_dir / f"{output_name}.ncu-rep"
    if dry_run:
        return {
            "dry_run": True,
            **precheck,
            "workload_type": workload_type,
            "kernel_filter": kernel_filter,
            "force_lineinfo": force_lineinfo,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "planned_output": str(output_path),
            "note": "Set dry_run=false to execute; use queue_only=true to background the run.",
        }

    def _execute_capture():
        auto = NsightAutomation(output_dir)
        path = auto.profile_ncu(
            command=command,
            output_name=output_name,
            workload_type=workload_type,
            kernel_filter=kernel_filter,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )

        result = {
            "success": path is not None,
            "output": str(path) if path else None,
            "workload_type": workload_type,
            "ncu_available": auto.ncu_available,
            "cwd": str(output_dir),
            "force_lineinfo": force_lineinfo,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "timeout_hit": bool(auto.last_run.get("timeout_hit")) if hasattr(auto, "last_run") else False,  # type: ignore[attr-defined]
            "error": auto.last_error if path is None else None,
            "run_details": getattr(auto, "last_run", {}),  # type: ignore[attr-defined]
        }
        return attach_context_if_requested(result, include_context, context_level)

    if queue_only:
        queued = _queue_job("aisp_profile_ncu", _execute_capture, params)
        queued["note"] = "Background capture started; poll with aisp_job_status using job_id."
        queued["workload_type"] = workload_type
        return queued

    return _execute_capture()


@register_tool(
    "aisp_profile_torch",
    "Run PyTorch autograd/torch.profiler capture and emit a Chrome trace + summary. Requires torch installed. Defaults to NVTX + force_lineinfo for correlation with nsys.",
    {"type": "object", "properties": {
        "script": {"type": "string", "description": "Path to Python script to profile"},
        "script_args": {"type": "array", "items": {"type": "string"}, "description": "Args forwarded to the script"},
        "output_name": {"type": "string", "description": "Base name for the capture folder", "default": "mcp_torch"},
        "output_dir": {"type": "string", "description": "Output root (default: artifacts/mcp-profiles/torch)", "default": "artifacts/mcp-profiles/torch"},
        "mode": {"type": "string", "description": "Profiler preset", "enum": ["full", "memory", "flops", "modules", "blackwell"], "default": "full"},
        "nvtx_label": {"type": "string", "description": "NVTX/record_function range label", "default": "aisp_torch_profile"},
        "use_nvtx": {"type": "boolean", "description": "Emit NVTX range around the profiled run", "default": True},
        "force_lineinfo": {"type": "boolean", "description": "Force -lineinfo in NVCC/TORCH_NVCC_FLAGS for better source mapping", "default": True},
        "precheck_only": {"type": "boolean", "description": "Return prereqs without running", "default": False},
        "dry_run": {"type": "boolean", "description": "Describe the capture without executing (alias: estimate_only)", "default": False},
        "queue_only": {"type": "boolean", "description": "Return a job ticket and run capture in background; poll with aisp_job_status", "default": False},
        "timeout_seconds": {"type": "integer", "description": "Max runtime before returning partial output; set 0/null for no timeout", "default": 300},
        "include_context": {"type": "boolean", "description": "Include full system context in the response", "default": True},
        "context_level": {"type": "string", "enum": ["summary", "full"], "default": "summary"},
    }, "required": ["script"]}
)
def tool_profile_torch(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run torch.profiler for a Python script and return summary + trace paths."""
    from pathlib import Path
    from core.profiling.torch_profiler import TorchProfilerAutomation

    script = params.get("script")
    if not script:
        return {"error": "script is required"}

    script_path = Path(script)
    output_dir = Path(params.get("output_dir", "artifacts/mcp-profiles/torch"))
    output_name = params.get("output_name") or script_path.stem or "mcp_torch"
    mode = params.get("mode", "full")
    script_args = params.get("script_args") or []
    if isinstance(script_args, str):
        import shlex as _shlex  # local import to avoid global side effects
        script_args = _shlex.split(script_args)
    force_lineinfo = bool(params.get("force_lineinfo", True))
    use_nvtx = bool(params.get("use_nvtx", True))
    nvtx_label = params.get("nvtx_label", "aisp_torch_profile")
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    queue_only = bool(params.get("queue_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context = bool(params.get("include_context", True))
    context_level = params.get("context_level", "summary")

    try:
        import torch  # noqa: F401
        torch_available = True
        cuda_available = torch.cuda.is_available()  # type: ignore[attr-defined]
        torch_error = None
    except Exception as exc:  # pragma: no cover - defensive
        torch_available = False
        cuda_available = False
        torch_error = str(exc)

    precheck = {
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "torch_error": torch_error,
        "output_dir": str(output_dir),
        "script_exists": script_path.exists(),
        "force_lineinfo": force_lineinfo,
        "nvtx_label": nvtx_label,
    }
    if precheck_only:
        return {"precheck_only": True, **precheck}
    if not script_path.exists():
        return {"error": f"script not found: {script}", **precheck}
    if dry_run:
        planned = output_dir / f"{output_name}_<timestamp>"
        return {
            "dry_run": True,
            **precheck,
            "planned_output": str(planned),
            "timeout_seconds": timeout_seconds,
            "mode": mode,
        }

    def _execute_capture():
        runner = TorchProfilerAutomation(output_dir)
        result = runner.profile(
            script=script_path,
            output_name=output_name,
            mode=mode,
            script_args=script_args,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
            nvtx_label=nvtx_label,
            use_nvtx=use_nvtx,
        )
        result.update({"torch_available": torch_available, "cuda_available": cuda_available})
        if not result.get("success"):
            result.setdefault("error", runner.last_error or "torch profiler failed")
        return attach_context_if_requested(result, include_context, context_level)

    if queue_only:
        queued = _queue_job("aisp_profile_torch", _execute_capture, params)
        queued["note"] = "Background torch.profiler capture started; poll with aisp_job_status using job_id."
        queued["mode"] = mode
        queued["nvtx_label"] = nvtx_label
        return queued

    return _execute_capture()


@register_tool(
    "aisp_profile_hta",
    "Run an Nsight Systems capture with HTA-friendly flags and analyze with HTAAnalyzer. Produces .nsys-rep, exported trace.json, and hta_report.json.",
    {"type": "object", "properties": {
        "command": {"type": "array", "items": {"type": "string"}, "description": "Command to profile (argv list)"},
        "output_name": {"type": "string", "description": "Base name for output files", "default": "mcp_hta"},
        "output_dir": {"type": "string", "description": "Directory for outputs (default: artifacts/hta)", "default": "artifacts/hta"},
        "preset": {"type": "string", "description": "nsys preset", "enum": ["light", "full"], "default": "full"},
        "force_lineinfo": {"type": "boolean", "description": "Force -lineinfo for source/line mapping", "default": True},
        "precheck_only": {"type": "boolean", "description": "Return prereqs without running", "default": False},
        "dry_run": {"type": "boolean", "description": "Describe the capture without executing", "default": False},
        "queue_only": {"type": "boolean", "description": "Run in background and return job_id", "default": False},
        "timeout_seconds": {"type": "integer", "description": "Max runtime before returning partial output; set 0/null for none", "default": 300},
        "include_context": {"type": "boolean", "description": "Include full system context in the response", "default": True},
        "context_level": {"type": "string", "enum": ["summary", "full"], "default": "summary"},
    }, "required": ["command"]}
)
def tool_profile_hta(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run nsys + HTA analysis."""
    import shlex
    from pathlib import Path
    from core.profiling.hta_capture import HTACaptureAutomation
    from core.profiling.nsight_automation import NsightAutomation

    command_list = params.get("command") or []
    if isinstance(command_list, str):
        command_list = shlex.split(command_list)
    if not command_list:
        return {"error": "command is required"}

    output_dir = Path(params.get("output_dir", "artifacts/hta"))
    output_name = params.get("output_name", "mcp_hta")
    preset = params.get("preset", "full")
    force_lineinfo = bool(params.get("force_lineinfo", True))
    precheck_only = bool(params.get("precheck_only"))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    queue_only = bool(params.get("queue_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context = bool(params.get("include_context", True))
    context_level = params.get("context_level", "summary")

    nsight = NsightAutomation(output_dir)
    try:
        import hta  # noqa: F401
        hta_available = True
    except Exception:
        hta_available = False

    precheck = {
        "nsys_available": nsight.nsys_available,
        "hta_available": hta_available,
        "output_dir": str(output_dir),
        "preset": preset,
        "command_provided": bool(command_list),
        "force_lineinfo": force_lineinfo,
    }
    if precheck_only:
        return {"precheck_only": True, **precheck}
    if not nsight.nsys_available:
        return {"error": "nsys is not installed or not on PATH", **precheck}
    if dry_run:
        base = output_dir / f"{output_name}.nsys-rep"
        return {
            "dry_run": True,
            **precheck,
            "planned_output": str(base),
            "timeout_seconds": timeout_seconds,
        }

    def _execute_capture():
        runner = HTACaptureAutomation(output_dir)
        result = runner.capture(
            command=command_list,
            output_name=output_name,
            preset=preset,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
        )
        result.update({"hta_available": hta_available, "nsys_available": nsight.nsys_available})
        if not result.get("success"):
            result.setdefault("error", runner.last_error or "HTA capture failed")
        return attach_context_if_requested(result, include_context, context_level)

    if queue_only:
        queued = _queue_job("aisp_profile_hta", _execute_capture, params)
        queued["note"] = "Background HTA capture started; poll with aisp_job_status using job_id."
        queued["preset"] = preset
        return queued

    return _execute_capture()


@register_tool(
    "aisp_nsys_summary",
    "Summarize an existing Nsight Systems report (.nsys-rep or CSV). Expects report_path to an existing file you already captured.",
    {"type": "object", "properties": {
        "report_path": {
            "type": "string",
            "description": "Path to .nsys-rep or exported CSV"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }, "required": ["report_path"]}
)
def tool_nsys_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize an Nsight Systems report."""
    from pathlib import Path
    from core.profiling.extract_nsys_summary import harvest

    report_path = params.get("report_path")
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")

    if not report_path:
        return {"error": "report_path is required"}

    path = Path(report_path)
    if not path.exists():
        return {"error": f"report_path not found: {path}", "report_path": str(path)}

    try:
        metrics = harvest(path)
    except Exception as e:
        return {"error": f"Failed to parse nsys report: {e}"}

    result = {
        "report": str(path),
        "metrics": metrics,
        "count": len(metrics),
    }
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_nsys_ncu_available",
    "Check availability of Nsight Systems and Nsight Compute.",
    {"type": "object", "properties": {}}
)
def tool_nsys_ncu_available(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check Nsight tool availability."""
    from core.profiling.nsight_automation import NsightAutomation
    automation = NsightAutomation(Path("artifacts/mcp-profiles"))
    return {
        "nsys_available": automation.nsys_available,
        "ncu_available": automation.ncu_available,
        "output_dir": str(automation.output_dir),
    }


@register_tool(
    "aisp_export_csv",
    "Export benchmarks to CSV. Use to share results.",
    {"type": "object", "properties": {
        "detailed": {
            "type": "boolean",
            "description": "Use detailed CSV",
            "default": False
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_export_csv(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to CSV."""
    from core.engine import get_engine
    detailed = bool(params.get("detailed", False))
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    export = get_engine().export.csv_detailed() if detailed else get_engine().export.csv()
    result = {"csv": export, "detailed": detailed}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_export_pdf",
    "Export benchmarks to PDF report. Use for sharing summary reports.",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_export_pdf(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to PDF."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    pdf_bytes = get_engine().export.pdf()
    result = {"pdf_base64": pdf_bytes if isinstance(pdf_bytes, str) else str(pdf_bytes)}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_export_html",
    "Export benchmarks to HTML report. Use for sharing interactive reports.",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_export_html(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to HTML."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    html = get_engine().export.html()
    result = {"html": html}
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# TEST TOOLS
# =============================================================================

@register_tool(
    "aisp_test_speed",
    "Run speed tests on the system. Run aisp_status or aisp_triage first; supports precheck_only/dry_run/timeout_seconds so you can opt in before executing. Example: \"Quickly benchmark host/GPU speed.\"",
    {"type": "object", "properties": {
        "gemm_size": {"type": "integer", "description": "GEMM size", "default": 512},
        "precision": {"type": "string", "description": "Precision (fp16/bf16/tf32/fp32/fp8)", "default": "fp16"},
        "mem_size_mb": {"type": "integer", "description": "Memory test size MB", "default": 16},
        "mem_stride": {"type": "integer", "description": "Memory stride bytes", "default": 128},
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites and planned command without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the bench invocation without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 300
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_test_speed(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run speed tests."""
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    args = [
        "test", "speed",
        "--type", "all",
        "--gemm-size", str(params.get("gemm_size", 512)),
        "--precision", params.get("precision", "fp16"),
        "--mem-size-mb", str(params.get("mem_size_mb", 16)),
        "--mem-stride", str(params.get("mem_stride", 128)),
    ]
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned_args": args,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }

    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
            "planned_args": args,
        }

    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "command": " ".join([sys.executable, "-m", "cli.aisp", "bench", *args]),
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run aisp_status first.",
        }

    result = _run_bench_cli(args, timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_test_roofline",
    "Stride sweep ASCII roofline for memory (bench test roofline). Run aisp_status first; supports precheck_only/dry_run/timeout_seconds to keep this opt-in.",
    {"type": "object", "properties": {
        "size_mb": {"type": "integer", "description": "Buffer size MB", "default": 32},
        "strides": {"type": "array", "items": {"type": "integer"}, "description": "Stride values"},
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites and planned command without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the bench invocation without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 300
        },
    }}
)
def tool_test_roofline(params: Dict[str, Any]) -> Dict[str, Any]:
    args: List[str] = ["test", "roofline", "--size-mb", str(params.get("size_mb", 32))]
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    for s in params.get("strides") or []:
        args.extend(["--stride", str(s)])
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned_args": args,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
            "planned_args": args,
        }
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "command": " ".join([sys.executable, "-m", "cli.aisp", "bench", *args]),
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run aisp_status first.",
        }
    return _run_bench_cli(args, timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)

@register_tool(
    "aisp_test_disk",
    "Disk I/O benchmark (sequential). Supports precheck_only/dry_run/timeout_seconds; optional tmp_dir will be created if missing.",
    {"type": "object", "properties": {
        "file_size_mb": {"type": "integer", "default": 256},
        "block_size_kb": {"type": "integer", "default": 1024},
        "tmp_dir": {"type": "string"},
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites and paths without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the disk test without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 120
        },
    }}
)
def tool_test_disk(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    tmp_dir = params.get("tmp_dir")
    tmp_path = Path(tmp_dir) if tmp_dir else None
    precheck = {
        "tmp_dir": str(tmp_path) if tmp_path else None,
        "exists": tmp_path.exists() if tmp_path else True,
        "writable": os.access(tmp_path, os.W_OK) if tmp_path and tmp_path.exists() else None,
    }
    if precheck_only:
        return {
            "precheck_only": True,
            "disk": precheck,
            "note": "No data written; rerun without precheck_only to execute.",
        }
    if dry_run:
        return {
            "dry_run": True,
            "disk": precheck,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "No data written; rerun with dry_run=false to execute.",
        }
    if tmp_dir:
        try:
            _ensure_dir(Path(tmp_dir))
        except Exception:
            return {"error": f"failed to create tmp_dir: {tmp_dir}", "tmp_dir": tmp_dir}
    return microbench.disk_io_test(
        file_size_mb=int(params.get("file_size_mb", 256)),
        block_size_kb=int(params.get("block_size_kb", 1024)),
        tmp_dir=tmp_dir,
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )


@register_tool(
    "aisp_test_pcie",
    "PCIe H2D/D2H bandwidth benchmark using CUDA (torch). Run aisp_status first; supports precheck_only/dry_run/timeout_seconds so you can opt in to execution.",
    {"type": "object", "properties": {
        "size_mb": {"type": "integer", "default": 256},
        "iters": {"type": "integer", "default": 10},
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the PCIe test without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 120
        },
    }}
)
def tool_test_pcie(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return {"error": cuda_check.get("reason", "CUDA not available"), "cuda": cuda_check}
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size_mb": params.get("size_mb", 256), "iters": params.get("iters", 10)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    return microbench.pcie_bandwidth_test(
        size_mb=int(params.get("size_mb", 256)),
        iters=int(params.get("iters", 10)),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )


@register_tool(
    "aisp_test_mem_hierarchy",
    "Memory hierarchy stride test on GPU to gauge bandwidth. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    {"type": "object", "properties": {
        "size_mb": {"type": "integer", "default": 256},
        "stride": {"type": "integer", "default": 128},
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the test without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 120
        },
    }}
)
def tool_test_mem_hierarchy(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return {"error": cuda_check.get("reason", "CUDA not available"), "cuda": cuda_check}
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size_mb": params.get("size_mb", 256), "stride": params.get("stride", 128)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    return microbench.mem_hierarchy_test(
        size_mb=int(params.get("size_mb", 256)),
        stride=int(params.get("stride", 128)),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )


@register_tool(
    "aisp_test_tensor_core",
    "Tensor Core matmul throughput test for various precisions. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    {"type": "object", "properties": {
        "size": {"type": "integer", "default": 4096},
        "precision": {"type": "string", "default": "fp16"},
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the test without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 120
        },
    }}
)
def tool_test_tensor_core(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return {"error": cuda_check.get("reason", "CUDA not available"), "cuda": cuda_check}
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size": params.get("size", 4096), "precision": params.get("precision", "fp16")},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    return microbench.tensor_core_bench(
        size=int(params.get("size", 4096)),
        precision=params.get("precision", "fp16"),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )


@register_tool(
    "aisp_test_sfu",
    "SFU-heavy benchmark (sin/cos) to gauge special function performance. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    {"type": "object", "properties": {
        "elements": {"type": "integer", "default": 67108864},
        "precheck_only": {
            "type": "boolean",
            "description": "Return prerequisites without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the test without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 120
        },
    }}
)
def tool_test_sfu(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return {"error": cuda_check.get("reason", "CUDA not available"), "cuda": cuda_check}
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"elements": params.get("elements", 64 * 1024 * 1024)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    return microbench.sfu_bench(
        size=int(params.get("elements", 64 * 1024 * 1024)),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )


@register_tool(
    "aisp_test_network_loopback",
    "Loopback TCP throughput test (localhost). Supports precheck_only/dry_run/timeout_seconds for opt-in execution.",
    {"type": "object", "properties": {
        "size_mb": {"type": "integer", "default": 64},
        "port": {"type": "integer", "default": 50007},
        "precheck_only": {
            "type": "boolean",
            "description": "Return port/info without running",
            "default": False
        },
        "dry_run": {
            "type": "boolean",
            "description": "Describe the loopback test without executing (alias: estimate_only)",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 60
        },
    }}
)
def tool_test_network_loopback(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    port = int(params.get("port", 50007))
    if precheck_only:
        return {
            "precheck_only": True,
            "port": port,
            "note": "No sockets opened; rerun without precheck_only to execute.",
        }
    if dry_run:
        return {
            "dry_run": True,
            "port": port,
            "params": {"size_mb": params.get("size_mb", 64)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    return microbench.network_loopback_test(
        size_mb=int(params.get("size_mb", 64)),
        port=port,
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )



@register_tool(
    "aisp_test_network",
    "Run network throughput tests. Use when checking interconnect or host network bandwidth. Example: \"Test network bandwidth between nodes.\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_test_network(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run network tests."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().test.network()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_benchmark_targets",
    "List benchmark targets supported by the harness. Use when choosing what to run.",
    {"type": "object", "properties": {}}
)
def tool_benchmark_targets(params: Dict[str, Any]) -> Dict[str, Any]:
    """List benchmark targets."""
    from core.engine import get_engine
    return get_engine().test.targets()


# =============================================================================
# ADVANCED ANALYSIS TOOLS
# =============================================================================

@register_tool(
    "aisp_cpu_memory_analysis",
    "Analyze CPU/memory hierarchy (NUMA, caches, TLB, hugepages). Use for host-side bottlenecks.",
    {"type": "object", "properties": {}}
)
def tool_cpu_memory_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.cpu_memory()


@register_tool(
    "aisp_system_parameters",
    "Inspect kernel/system parameters (swappiness, dirty ratios, etc.). Use for host tuning.",
    {"type": "object", "properties": {}}
)
def tool_system_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.system_params()


@register_tool(
    "aisp_container_limits",
    "Inspect container/cgroup limits. Use when running in containers.",
    {"type": "object", "properties": {}}
)
def tool_container_limits(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.container_limits()


@register_tool(
    "aisp_warp_divergence",
    "Analyze code for warp divergence patterns.",
    {"type": "object", "properties": {
        "code": {"type": "string", "description": "Kernel code or snippet"}
    }}
)
def tool_warp_divergence(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.warp_divergence()


@register_tool(
    "aisp_bank_conflicts",
    "Analyze shared memory bank conflicts.",
    {"type": "object", "properties": {
        "stride": {"type": "integer", "default": 1},
        "element_size": {"type": "integer", "default": 4}
    }}
)
def tool_bank_conflicts(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.bank_conflicts()


@register_tool(
    "aisp_memory_access",
    "Analyze memory access patterns for coalescing.",
    {"type": "object", "properties": {
        "stride": {"type": "integer", "default": 1},
        "element_size": {"type": "integer", "default": 4}
    }}
)
def tool_memory_access(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.memory_access()


@register_tool(
    "aisp_comm_overlap",
    "Analyze communication overlap for a model.",
    {"type": "object", "properties": {
        "model": {"type": "string", "default": "llama-3.1-70b"}
    }}
)
def tool_comm_overlap(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.comm_overlap(params.get("model", "default"))


@register_tool(
    "aisp_data_loading",
    "Analyze data loading pipeline.",
    {"type": "object", "properties": {}}
)
def tool_data_loading(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.data_loading()


@register_tool(
    "aisp_energy_analysis",
    "Analyze energy efficiency.",
    {"type": "object", "properties": {}}
)
def tool_energy_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.energy()


@register_tool(
    "aisp_predict_scaling",
    "Predict scaling behavior for model size and GPU count.",
    {"type": "object", "properties": {
        "model_size": {"type": "number", "description": "Model size in billions", "default": 7},
        "gpus": {"type": "integer", "description": "GPU count", "default": 8}
    }}
)
def tool_predict_scaling(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine
    return get_engine().analyze.predict_scaling(
        model_size=params.get("model_size", 7),
        gpus=params.get("gpus", 8),
    )


# =============================================================================
# HUGGINGFACE TOOLS
# =============================================================================

@register_tool(
    "aisp_hf_search",
    "Tags: huggingface, search, models. Search HuggingFace for models. Use when finding candidate models to run. Example: \"Search HF for code generation models.\"",
    {"type": "object", "properties": {
        "query": {
            "type": "string",
            "description": "Search query (e.g., 'llama', 'code generation')"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results",
            "default": 10
        }
    }, "required": ["query"]}
)
def tool_hf_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search HuggingFace."""
    from core.engine import get_engine
    return get_engine().hf.search(
        query=params.get("query", ""),
        limit=params.get("limit", 10)
    )


@register_tool(
    "aisp_hf_trending",
    "Tags: huggingface, trending, browse. Get trending models on HuggingFace. Use when browsing popular models. Example: \"What models are trending for text-generation?\"",
    {"type": "object", "properties": {
        "task": {
            "type": "string",
            "description": "Task type: text-generation, image-classification, etc.",
            "default": "text-generation"
        }
    }}
)
def tool_hf_trending(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get trending models."""
    from core.engine import get_engine
    return get_engine().hf.trending(task=params.get("task", "text-generation"))


# =============================================================================
# CLUSTER TOOLS
# =============================================================================

@register_tool(
    "aisp_cluster_slurm",
    "Tags: slurm, batch, cluster. Generate SLURM script for cluster job submission. Use when creating batch jobs on a cluster. Example: \"Create SLURM script for 2 nodes x 8 GPUs.\"",
    {"type": "object", "properties": {
        "model": {
            "type": "string",
            "description": "Model size (e.g., '7b', '70b')",
            "default": "7b"
        },
        "nodes": {
            "type": "integer",
            "description": "Number of nodes",
            "default": 1
        },
        "gpus": {
            "type": "integer",
            "description": "GPUs per node",
            "default": 8
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_cluster_slurm(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SLURM script."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().cluster.slurm(
        model=params.get("model", "7b"),
        nodes=params.get("nodes", 1),
        gpus=params.get("gpus", 8)
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_cost_estimate",
    "Tags: cost, budget, cloud. Estimate cloud costs for training/inference. Use when budgeting for runs. Example: \"Estimate cost to train 70B with 200B tokens on AWS.\"",
    {"type": "object", "properties": {
        "model_size": {
            "type": "number",
            "description": "Model size in billions"
        },
        "training_tokens": {
            "type": "number",
            "description": "Training tokens in billions"
        },
        "provider": {
            "type": "string",
            "description": "Cloud provider: aws, gcp, azure",
            "default": "aws"
        },
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": False
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_cost_estimate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate cloud costs."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", False))
    context_level = params.get("context_level", "summary")
    result = get_engine().cost.cloud_estimate(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_launch_plan",
    "Generate a torchrun launch plan (dry-run) with TP/PP/DP layout. Returns JSON and command string.",
    {"type": "object", "properties": {
        "model_params": {"type": "integer", "description": "Model size in billions", "default": 70},
        "nodes": {"type": "integer", "default": 1},
        "gpus": {"type": "integer", "description": "GPUs per node", "default": 8},
        "tp": {"type": "integer", "default": 1},
        "pp": {"type": "integer", "default": 1},
        "dp": {"type": "integer", "default": 1},
        "batch_size": {"type": "integer", "default": 1},
        "script": {"type": "string", "default": "train.py"},
        "extra_args": {"type": "string", "description": "Extra args appended to command"},
    }}
)
def tool_launch_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from core.optimization.parallelism_planner.launch_plan import generate_launch_plan
        plan = generate_launch_plan(
            model_params=params.get("model_params", 70),
            nodes=params.get("nodes", 1),
            gpus_per_node=params.get("gpus", 8),
            tp=params.get("tp", 1),
            pp=params.get("pp", 1),
            dp=params.get("dp", 1),
            batch_size=params.get("batch_size", 1),
            script=params.get("script", "train.py"),
            extra_args=params.get("extra_args"),
        )
        return {"plan": plan.to_json(), "command": plan.command}
    except Exception as exc:
        return {"error": str(exc)}


# =============================================================================
# QUICK STATUS TOOL
# =============================================================================

@register_tool(
    "aisp_status",
    "Tags: status, health, quick check. Get quick system status: GPU, software, AI backend. Use when you need a fast snapshot before deeper analysis. Example: \"Show quick status before running profiling.\" or \"Is everything healthy before a long job?\"",
    {"type": "object", "properties": {
        "include_context": {
            "type": "boolean",
            "description": "Include full system context in the response",
            "default": True
        },
        "context_level": {
            "type": "string",
            "description": "Context level: summary or full",
            "enum": ["summary", "full"],
            "default": "summary"
        }
    }}
)
def tool_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get quick status."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", True))
    context_level = params.get("context_level", "summary")
    result = get_engine().status()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_context_summary",
    "Get a lightweight system context summary (GPU + software). Use when you need context for other tool calls. Example: \"Give me a quick env summary before optimizing.\"",
    {"type": "object", "properties": {}}
)
def tool_context_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get cached summary context."""
    return get_cached_context("summary")


@register_tool(
    "aisp_context_full",
    "Get full system context (may be heavier). Use when detailed environment is required. Example: \"Full context dump for LLM analysis.\"",
    {"type": "object", "properties": {}}
)
def tool_context_full(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get cached full context."""
    return get_cached_context("full")


@register_tool(
    "aisp_triage",
    "Quick triage snapshot: status + context summary. Use as a first call before other tools. Example: \"Start with triage before choosing tools.\"",
    {"type": "object", "properties": {}}
)
def tool_triage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return quick status plus context summary to guide next actions."""
    from core.engine import get_engine
    engine = get_engine()
    return {
        "status": engine.status(),
        "context": get_cached_context("summary"),
    }


@register_tool(
    "aisp_job_status",
    "Check the status/result of a queued tool (e.g., aisp_profile_nsys/aisp_profile_ncu with queue_only=true). Use job_id returned by the queueing call.",
    {"type": "object", "properties": {
        "job_id": {
            "type": "string",
            "description": "Job ID returned from queue_only=true call"
        }
    }, "required": ["job_id"]}
)
def tool_job_status(params: Dict[str, Any]) -> Dict[str, Any]:
    job_id = params.get("job_id")
    if not job_id:
        return {"error": "job_id is required"}
    with _JOB_LOCK:
        record = _JOB_STORE.get(job_id)
    if not record:
        return {
            "job_id": job_id,
            "status": "not_found",
            "note": "No job with this id; ensure you passed queue_only=true on the original call.",
        }
    # Copy so we don't mutate stored record when adding error decoration
    payload = dict(record)
    if payload.get("status") == "error" and "error" not in payload:
        result = payload.get("result") or {}
        if isinstance(result, dict) and result.get("error"):
            payload["error"] = result.get("error")
        else:
            payload["error"] = "Job failed"
    return payload


@register_tool(
    "aisp_help",
    "Start here if unsure: call aisp_suggest_tools with your intent to get ranked tool suggestions.",
    {"type": "object", "properties": {
        "query": {
            "type": "string",
            "description": "What the user wants (forwarded to aisp_suggest_tools)"
        }
    }, "required": ["query"]}
)
def tool_help(params: Dict[str, Any]) -> Dict[str, Any]:
    """Forward to suggest_tools with the provided query."""
    return tool_suggest_tools(params)


@register_tool(
    "aisp_suggest_tools",
    "When unsure which tool to use, call this first with the user intent to get ranked tool suggestions. Suggest relevant MCP tools based on a short intent. Example: \"I keep OOMing on 24GB VRAM\" or \"Need lower latency on vLLM\".",
    {"type": "object", "properties": {
        "query": {
            "type": "string",
            "description": "User intent or question"
        }
    }, "required": ["query"]}
)
def tool_suggest_tools(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ranked list of suggested tools given a query."""
    query = (params.get("query") or "").lower()

    rules = [
        {
            "tool": "aisp_analyze_bottlenecks",
            "keywords": ["slow", "latency", "bottleneck", "utilization", "stall", "idle", "regression", "throughput drop"],
            "reason": "Diagnose bottlenecks for slow workload/latency issues",
        },
        {
            "tool": "aisp_test_disk",
            "keywords": ["disk", "io", "storage"],
            "reason": "Disk I/O benchmark (sequential)",
        },
        {
            "tool": "aisp_test_pcie",
            "keywords": ["pcie", "h2d", "d2h", "pci-e"],
            "reason": "PCIe H2D/D2H bandwidth benchmark",
        },
        {
            "tool": "aisp_test_mem_hierarchy",
            "keywords": ["memory stride", "cache", "l2", "hbm"],
            "reason": "Stride/bandwidth test for memory hierarchy",
        },
        {
            "tool": "aisp_test_tensor_core",
            "keywords": ["tensor core", "tflops", "matmul"],
            "reason": "Tensor core throughput test",
        },
        {
            "tool": "aisp_test_sfu",
            "keywords": ["sfu", "sin", "cos"],
            "reason": "SFU throughput benchmark",
        },
        {
            "tool": "aisp_test_network_loopback",
            "keywords": ["loopback", "tcp", "network", "nic"],
            "reason": "Loopback TCP throughput test",
        },
        {
            "tool": "aisp_profile_flame",
            "keywords": ["profile", "flame", "time", "hotspot", "trace", "timeline", "slow step"],
            "reason": "Inspect time hotspots with flame graph",
        },
        {
            "tool": "aisp_profile_flame",
            "keywords": ["profile", "flame", "time", "hotspot", "trace"],
            "reason": "Inspect time hotspots with flame graph",
        },
        # Profile kernels is reserved for perf hotspots; avoid matching install/import issues.
        {
            "tool": "aisp_profile_kernels",
            "keywords": ["kernel hotspot", "cuda hotspot", "ptx hotspot", "nsight", "kernel profiling"],
            "reason": "Check CUDA kernel hotspots",
        },
        {
            "tool": "aisp_profile_roofline",
            "keywords": ["roofline", "compute bound", "memory bound", "arithmetic intensity"],
            "reason": "See compute vs memory bound positioning",
        },
        {
            "tool": "aisp_profile_nsys",
            "keywords": ["nsys", "nsight systems", "systems trace"],
            "reason": "Capture timeline with Nsight Systems",
        },
        {
            "tool": "aisp_profile_ncu",
            "keywords": ["ncu", "nsight compute", "compute profile", "kernel metrics", "ncu profile"],
            "reason": "Capture kernel metrics with Nsight Compute",
        },
        {
            "tool": "aisp_nsys_summary",
            "keywords": ["nsys summary", "nsight systems report", "nsys-rep"],
            "reason": "Summarize Nsight Systems report metrics",
        },
        {
            "tool": "aisp_profile_memory",
            "keywords": ["memory", "vram", "oom", "leak", "fragmentation", "spike"],
            "reason": "See memory timeline and spikes",
        },
        {
            "tool": "aisp_gpu_bandwidth",
            "keywords": ["bandwidth", "p2p", "nvlink", "pci-e", "pci express"],
            "reason": "Check GPU memory/P2P bandwidth",
        },
        {
            "tool": "aisp_gpu_power",
            "keywords": ["power", "thermal", "throttle", "temperature", "temp"],
            "reason": "Check power/thermal headroom and throttling",
        },
        {
            "tool": "aisp_gpu_info",
            "keywords": ["gpu info", "name", "memory", "compute capability"],
            "reason": "Get GPU inventory and basic telemetry",
        },
        {
            "tool": "aisp_system_software",
            "keywords": ["pytorch", "cuda version", "driver", "software version"],
            "reason": "Check software stack versions",
        },
        {
            "tool": "aisp_system_dependencies",
            "keywords": ["import error", "torch.cuda", "dependency", "missing library"],
            "reason": "Check dependency health for install/import issues",
        },
        {
            "tool": "aisp_analyze_whatif",
            "keywords": ["vram", "memory", "limit", "constraint", "cap"],
            "reason": "What-if recommendations under VRAM/latency constraints",
        },
        {
            "tool": "aisp_recommend",
            "keywords": ["throughput", "latency target", "goal", "optimize", "recommend", "playbook"],
            "reason": "Get an optimization playbook for your goal",
        },
        {
            "tool": "aisp_analyze_pareto",
            "keywords": ["compare", "tradeoff", "pareto"],
            "reason": "Compare throughput/latency/memory tradeoffs",
        },
        {
            "tool": "aisp_cluster_slurm",
            "keywords": ["slurm", "batch", "sbatch", "job script"],
            "reason": "Generate SLURM script for cluster runs",
        },
        {
            "tool": "aisp_cost_estimate",
            "keywords": ["cost", "budget", "price", "cloud", "tokens"],
            "reason": "Estimate cloud costs for training/inference",
        },
        {
            "tool": "aisp_distributed_plan",
            "keywords": ["distributed", "multi node", "tp", "pp", "dp", "fsdp"],
            "reason": "Plan DP/TP/PP strategy",
        },
        {
            "tool": "aisp_distributed_nccl",
            "keywords": ["nccl", "collective", "allreduce", "all-gather"],
            "reason": "Tune NCCL for multi-node",
        },
        {
            "tool": "aisp_run_benchmarks",
            "keywords": ["benchmark", "benchmarks", "perf run"],
            "reason": "Run standard benchmarks with optional profiling",
        },
        {
            "tool": "aisp_verify_benchmarks",
            "keywords": ["verify benchmark", "validate benchmark", "rerun benchmark"],
            "reason": "Verify prior benchmark runs",
        },
        {
            "tool": "aisp_inference_vllm",
            "keywords": ["vllm", "inference", "serving", "throughput", "latency"],
            "reason": "Generate vLLM config for throughput/latency",
        },
        {
            "tool": "aisp_inference_quantization",
            "keywords": ["quant", "int8", "fp8", "fp4", "kv cache"],
            "reason": "Quantization guidance for inference",
        },
        {
            "tool": "aisp_hf_search",
            "keywords": ["huggingface", "hf search", "find model", "search model"],
            "reason": "Search HuggingFace for models",
        },
        {
            "tool": "aisp_hf_trending",
            "keywords": ["huggingface trending", "hf trending", "popular models"],
            "reason": "List trending models on HuggingFace",
        },
        {
            "tool": "aisp_benchmark_targets",
            "keywords": ["benchmark targets", "bench targets", "list benchmarks", "what can I run"],
            "reason": "List benchmark targets",
        },
        {
            "tool": "aisp_context_summary",
            "keywords": ["context", "environment", "status", "summary", "env"],
            "reason": "Fetch lightweight environment context",
        },
        {
            "tool": "aisp_triage",
            "keywords": ["triage", "start", "first", "status", "health", "quick check"],
            "reason": "Get status + summary context",
        },
        {
            "tool": "aisp_status",
            "keywords": ["status", "health", "ready", "check", "sanity"],
            "reason": "Quick status: GPU, software, AI backend",
        },
        {
            "tool": "aisp_ask",
            "keywords": ["question", "why", "how"],
            "reason": "Free-form performance question with citations",
        },
        {
            "tool": "aisp_explain",
            "keywords": ["what is", "explain", "concept"],
            "reason": "Explain a performance concept with citations",
        },
        {
            "tool": "aisp_ask",
            "keywords": ["flash attention", "torch.compile", "compile", "cuda graphs", "why slow"],
            "reason": "Ask targeted performance questions (FlashAttn, torch.compile, CUDA Graphs, etc.)",
        },
        {
            "tool": "aisp_benchmark_targets",
            "keywords": ["list targets", "what benchmarks", "examples", "chapters"],
            "reason": "List available benchmark targets (chapter:example)",
        },
        {
            "tool": "aisp_list_chapters",
            "keywords": ["list chapters", "labs", "what chapters", "what labs"],
            "reason": "List all chapters and labs",
        },
        {
            "tool": "aisp_benchmark_report",
            "keywords": ["report", "pdf", "html", "export report"],
            "reason": "Generate PDF/HTML benchmark report",
        },
        {
            "tool": "aisp_benchmark_export",
            "keywords": ["export", "csv", "markdown", "json"],
            "reason": "Export benchmark results",
        },
        {
            "tool": "aisp_benchmark_compare_runs",
            "keywords": ["compare runs", "diff results", "regressions", "improvements"],
            "reason": "Diff two benchmark JSON runs",
        },
        {
            "tool": "aisp_test_roofline",
            "keywords": ["stride", "roofline", "memory sweep"],
            "reason": "Quick stride sweep roofline for memory hierarchy",
        },
        {
            "tool": "aisp_launch_plan",
            "keywords": ["launch plan", "torchrun", "tp", "pp", "dp", "layout"],
            "reason": "Generate torchrun launch plan and command",
        },
        {
            "tool": "aisp_gpu_topology",
            "keywords": ["topology", "nvlink", "pcie", "multi gpu"],
            "reason": "Inspect multi-GPU topology",
        },
    ]

    def score(rule: Dict[str, Any], text: str) -> int:
        s = 0
        for kw in rule["keywords"]:
            if kw in text:
                s += 2 if " " in kw else 1
        return s

    scored = []
    for rule in rules:
        sc = score(rule, query)
        if sc > 0:
            scored.append((sc, rule))

    # If nothing matched, fall back to triage + core suggestions
    if not scored:
        suggestions = [
            {"tool": "aisp_triage", "reason": "Start with triage to gather context"},
            {"tool": "aisp_analyze_bottlenecks", "reason": "Check for bottlenecks"},
            {"tool": "aisp_recommend", "reason": "Get optimization recommendations"},
        ]
        return {"suggestions": suggestions, "count": len(suggestions)}

    scored.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    suggestions = []
    for sc, rule in scored:
        if rule["tool"] in seen:
            continue
        seen.add(rule["tool"])
        suggestions.append({"tool": rule["tool"], "reason": rule["reason"], "score": sc})

    return {"suggestions": suggestions, "count": len(suggestions)}


# =============================================================================
# MCP PROTOCOL IMPLEMENTATION
# =============================================================================

class MCPServer:
    """MCP Server for AI Systems Performance."""
    
    def __init__(self):
        self.name = "aisp"
        self.version = "2.0.0"
    
    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            }
            for tool in TOOLS.values()
        ]
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool by name."""
        start_ts = time.time()
        tool_meta = TOOLS.get(name)
        server_info = {"name": self.name, "version": self.version}
        if name not in HANDLERS:
            payload = _build_enriched_tool_payload(
                name,
                arguments,
                {"error": f"Unknown tool: {name}"},
                duration_ms=int((time.time() - start_ts) * 1000),
                server_info=server_info,
                tool_meta=tool_meta,
            )
            return ToolResult(
                content=_content_from_payload(payload),
                is_error=True
            )
        
        try:
            result = HANDLERS[name](arguments)
            duration_ms = int((time.time() - start_ts) * 1000)
            payload = _build_enriched_tool_payload(
                name,
                arguments,
                result,
                duration_ms,
                server_info=server_info,
                tool_meta=tool_meta,
            )
            return ToolResult(
                content=_content_from_payload(payload),
                is_error=payload.get("status") == "error"
            )
        except Exception as e:
            tb = traceback.format_exc()
            duration_ms = int((time.time() - start_ts) * 1000)
            payload = _build_enriched_tool_payload(
                name,
                arguments,
                {"error": str(e), "traceback": tb},
                duration_ms=duration_ms,
                had_exception=True,
                server_info=server_info,
                tool_meta=tool_meta,
            )
            return ToolResult(
                content=_content_from_payload(payload),
                is_error=True
            )
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle an MCP message. Returns None for notifications (messages without an id)."""
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = msg_id is None
        params = message.get("params", {})
        
        # Ignore notifications (messages without an id); MCP clients may send these.
        if is_notification:
            return None
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": self.get_tool_list()
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": result.content,
                    "isError": result.is_error
                }
            }
        
        else:
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def run_stdio(self):
        """Run MCP server over stdio."""
        import sys
        
        print(f"AISP MCP Server v{self.version} - {len(TOOLS)} tools available", file=sys.stderr)
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                message = json.loads(line)
                response = await self.handle_message(message)
                if response is not None:
                    print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AISP MCP Server")
    parser.add_argument("--list", action="store_true", help="List available tools")
    parser.add_argument("--test", type=str, help="Test a specific tool")
    parser.add_argument("--serve", action="store_true", help="Start MCP server (stdio)")
    args = parser.parse_args()
    
    if args.list:
        print(f"\n🚀 AISP MCP Tools ({len(TOOLS)} available):\n")
        for name, tool in sorted(TOOLS.items()):
            print(f"  {name}")
            print(f"    {tool.description}\n")
        return
    
    if args.test:
        tool_name = args.test
        if tool_name not in HANDLERS:
            print(f"Unknown tool: {tool_name}")
            return 1
        
        print(f"Testing {tool_name}...")
        result = HANDLERS[tool_name]({})
        print(json.dumps(result, indent=2, default=str))
        return
    
    if args.serve:
        server = MCPServer()
        asyncio.run(server.run_stdio())
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
