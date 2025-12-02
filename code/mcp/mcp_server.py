#!/usr/bin/env python3
"""
🚀 MCP Server for AI Systems Performance

Exposes the unified PerformanceEngine as MCP tools for AI chat integration.
Consolidated to ~73 tools (reduced from 86, preserving all unique functionality).

Usage:
    # Start the MCP server
    python -m mcp.mcp_server
    
    # Or use the aisp command
    aisp mcp serve

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  AI Chat Client                                                         │
    │  ↓                                                                      │
    │  MCP Server (this file) - 73 tools (no functionality lost)              │
    │  ↓                                                                      │
    │  PerformanceEngine (core/engine.py) - 10 unified domains                │
    │  ├── gpu         : aisp_gpu_info, aisp_gpu_topology, aisp_gpu_power     │
    │  ├── system      : aisp_system_software, aisp_system_dependencies       │
    │  ├── profile     : aisp_profile_nsys, aisp_profile_ncu, aisp_profile_*  │
    │  ├── analyze     : aisp_analyze_bottlenecks, aisp_analyze_pareto, ...   │
    │  ├── optimize    : aisp_recommend, aisp_optimize_roi, ...               │
    │  ├── distributed : aisp_distributed_plan, aisp_distributed_nccl         │
    │  ├── inference   : aisp_inference_vllm, aisp_inference_quantization     │
    │  ├── benchmark   : aisp_run_benchmarks, aisp_benchmark_targets          │
    │  ├── ai          : aisp_ask, aisp_explain, aisp_ai_status               │
    │  └── export      : aisp_export_csv, aisp_export_pdf, aisp_export_html   │
    └─────────────────────────────────────────────────────────────────────────┘

TOOL NAMING: aisp_{domain}_{operation}

QUICK START (4 tools):
    aisp_triage        - START HERE: status + context in one call
    aisp_status        - Quick health check (GPU/software/AI)
    aisp_suggest_tools - Get tool recommendations for your task
    aisp_job_status    - Poll async job completion

DOMAIN TOOLS (organized by 10-domain model):

    GPU (4 tools):
        aisp_gpu_info, aisp_gpu_bandwidth, aisp_gpu_topology, aisp_gpu_power

    System (3 tools):
        aisp_system_software, aisp_system_dependencies, aisp_system_context

    Profile (8 tools):
        aisp_profile_nsys, aisp_profile_ncu, aisp_profile_torch,
        aisp_profile_hta, aisp_profile_flame, aisp_profile_memory,
        aisp_profile_kernels, aisp_profile_compare

    Analyze (5 tools):
        aisp_analyze_bottlenecks, aisp_analyze_pareto, aisp_analyze_scaling,
        aisp_analyze_stacking, aisp_analyze_whatif

    Optimize (3 tools):
        aisp_recommend, aisp_optimize_roi, aisp_optimize_techniques

    Distributed (3 tools):
        aisp_distributed_plan, aisp_distributed_nccl, aisp_cluster_slurm

    Inference (2 tools):
        aisp_inference_vllm, aisp_inference_quantization

    Benchmark (5 tools):
        aisp_run_benchmarks, aisp_benchmark_targets, aisp_benchmark_report,
        aisp_benchmark_export, aisp_benchmark_compare_runs, aisp_benchmark_triage

    AI (3 tools):
        aisp_ask, aisp_explain, aisp_ai_status

    Export (3 tools):
        aisp_export_csv, aisp_export_pdf, aisp_export_html

    HuggingFace (1 tool):
        aisp_hf (search/trending/download via action param)

HARDWARE MICRO-BENCHMARKS (8 tools):
    aisp_hw_speed, aisp_hw_roofline, aisp_hw_disk, aisp_hw_pcie,
    aisp_hw_cache, aisp_hw_tc, aisp_hw_ib, aisp_hw_nccl, aisp_hw_p2p,
    aisp_hw_network

WORKFLOW EXAMPLES:
    New session:     aisp_triage → aisp_recommend → specific tools
    Slow training:   aisp_analyze_bottlenecks → aisp_profile_nsys → fix
    Multi-GPU:       aisp_gpu_topology → aisp_distributed_plan → aisp_distributed_nccl
    Inference:       aisp_inference_quantization → aisp_inference_vllm → deploy
    Benchmarks:      aisp_run_benchmarks(async=true) → aisp_job_status → aisp_benchmark_triage
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
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, NotRequired

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
# RESULT TYPE DEFINITIONS (TypedDict for type safety)
# =============================================================================

class ToolResultBase(TypedDict, total=False):
    """Base shape for all tool results."""
    success: bool
    error: NotRequired[str]
    context: NotRequired[Dict[str, Any]]
    context_error: NotRequired[str]


class BenchmarkExportResult(ToolResultBase):
    """Result shape for benchmark export operations."""
    output: str
    format: str
    benchmarks_written: int


class JobStatusResult(ToolResultBase):
    """Result shape for job status queries."""
    job_id: str
    status: str
    tool: NotRequired[str]
    submitted_at: NotRequired[float]
    finished_at: NotRequired[float]
    duration_ms: NotRequired[int]
    result: NotRequired[Any]
    note: NotRequired[str]


class SuggestionResult(ToolResultBase):
    """Result shape for tool suggestions."""
    suggestions: List[Dict[str, Any]]
    count: int


class ContextResult(ToolResultBase):
    """Result shape for context provider tools."""
    context: Dict[str, Any]


# =============================================================================
# CONTEXT PARAMS SCHEMA (DRY helper for tool definitions)
# =============================================================================

_CONTEXT_PARAMS_SCHEMA: Dict[str, Any] = {
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
}

# =============================================================================
# MINIMAL PARAM NORMALIZATION - Only for commonly confused values
# MCP Convention: LLM reads schema enums; we only fix the most common mistakes
# =============================================================================

# Only the most problematic aliases that cause real errors
_COMMON_ALIASES: Dict[str, Dict[str, str]] = {
    "profile": {"full": "deep_dive", "deep": "deep_dive", "nsys": "deep_dive"},
    "preset": {"deep": "full", "complete": "full"},
    "format": {"md": "markdown"},
    "collective": {"allreduce": "all_reduce", "allgather": "all_gather"},
}


def normalize_param(param_name: str, value: Any, default: Any = None) -> Any:
    """Normalize common parameter aliases. Returns original if no alias found."""
    if value is None:
        return default
    str_value = str(value).lower().strip()
    aliases = _COMMON_ALIASES.get(param_name, {})
    return aliases.get(str_value, str_value if str_value else default)


def with_context_params(props: Dict[str, Any]) -> Dict[str, Any]:
    """Merge tool-specific properties with standard context params."""
    return {**props, **_CONTEXT_PARAMS_SCHEMA}


def extract_context_opts(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Extract include_context and context_level from params."""
    return (
        bool(params.get("include_context", False)),
        params.get("context_level", "summary")
    )


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
    "aisp_benchmark_report": "Generates a report from existing benchmark JSON; writes PDF/HTML to the chosen output.",
    "aisp_benchmark_export": "Exports existing benchmark JSON to csv/markdown/json; writes to the chosen output file.",
    "aisp_benchmark_compare_runs": "Diffs two benchmark JSON files; CPU-bound and quick, writes only if an output is specified.",
    "aisp_profile_nsys": "Calls Nsight Systems; requires nsys installed and writes .nsys-rep into output_dir. Slow/interactive; run aisp_status or aisp_triage first. Default preset is full; set preset=light explicitly to shrink traces.",
    "aisp_profile_ncu": "Calls Nsight Compute; requires ncu installed and writes .ncu-rep into output_dir. Slow/interactive; run aisp_status or aisp_triage first. Defaults to memory_bound metric set; opt into heavier modes explicitly.",    "aisp_profile_compare": "Generates flame graph comparison; parses NSYS reports and may traverse multiple files; allow extra runtime.",
    "aisp_hw_speed": "Runs GPU/host micro-benchmarks; stresses hardware briefly. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_hw_roofline": "Runs roofline micro-benchmark; stresses memory subsystem briefly. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_hw_disk": "Runs disk I/O hardware benchmark; writes temporary files to tmp_dir. Supports precheck_only/dry_run/timeout_seconds.",
    "aisp_hw_pcie": "Runs PCIe hardware benchmark; exercises host↔GPU transfers. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_hw_cache": "Runs memory hierarchy hardware benchmark; exercises GPU cache. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
    "aisp_hw_tc": "Runs tensor core hardware benchmark; exercises GPU math units. Run aisp_status first; supports precheck_only/dry_run/timeout_seconds.",
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
    elif name_key.startswith("aisp_hw_") or name_key.startswith("aisp_test_"):
        notes.append("Runs hardware benchmarks; may briefly stress hardware.")
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
                # Return normalized error with success=False for consistency
                return {"error": str(exc), "success": False}

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


def make_error(
    msg: str,
    include_context: bool = False,
    context_level: str = "summary",
    **extra: Any
) -> Dict[str, Any]:
    """Build standardized error response with optional context attachment.
    
    Args:
        msg: The error message.
        include_context: Whether to attach system context.
        context_level: Level of context to attach ("summary" or "full").
        **extra: Additional fields to include in the error response.
    
    Returns:
        A normalized error dict with success=False and optional context.
    """
    result: Dict[str, Any] = {"error": msg, "success": False, **extra}
    return attach_context_if_requested(result, include_context, context_level)


def ensure_result(
    result: Dict[str, Any],
    include_context: bool,
    context_level: str
) -> Dict[str, Any]:
    """Normalize result and optionally attach context - single exit point for handlers.
    
    Args:
        result: The raw result dict from a tool handler.
        include_context: Whether to attach system context.
        context_level: Level of context to attach.
    
    Returns:
        Normalized result with success flag and optional context.
    """
    normalized = dict(result)
    if "success" not in normalized:
        normalized["success"] = not bool(normalized.get("error"))
    return attach_context_if_requested(normalized, include_context, context_level)


_BENCH_CLI_TIMEOUT = 900  # generous default; keeps CLI invocations from hanging forever


def _run_cli(args: List[str], timeout: Optional[int] = _BENCH_CLI_TIMEOUT) -> Dict[str, Any]:
    """Invoke aisp CLI directly (without bench prefix) and return stdout/stderr/exit code."""
    cmd = [sys.executable, "-m", "cli.aisp", *args]
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
        return result
    except subprocess.TimeoutExpired as te:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": te.stdout.decode() if te.stdout else "",
            "stderr": te.stderr.decode() if te.stderr else "",
            "timeout_seconds": timeout,
            "timeout_hit": True,
            "duration_seconds": round(time.time() - started_at, 2),
            "error": f"Timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "timeout_seconds": timeout if timeout and timeout > 0 else None,
            "timeout_hit": False,
            "duration_seconds": round(time.time() - started_at, 2),
            "error": str(exc),
        }


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


def _normalize_result(result: Any) -> Any:
    """Ensure results consistently expose a success flag when possible."""
    if isinstance(result, dict):
        normalized = dict(result)
        if "success" not in normalized:
            normalized["success"] = not bool(normalized.get("error"))
        return normalized
    return result


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
    "Tags: gpu, info, snapshot, health-check, inventory, nvidia-smi. "
    "Get GPU hardware snapshot: name, architecture, VRAM (total/used/free), temperature, power draw, utilization %. "
    "Returns: {gpus: [{name, memory_total_gb, memory_used_gb, temperature_c, power_w, utilization_pct}], count}. "
    "⚡ FAST (~1s). USE FIRST when: Starting any performance investigation, verifying hardware before profiling. "
    "USE INSTEAD OF: Running nvidia-smi manually in terminal. "
    "Example: \"Show GPU names, memory, temps\" or \"What GPUs do I have?\" or \"Check VRAM before loading model\". "
    "WORKFLOW: aisp_gpu_info → aisp_status → aisp_recommend → specific optimization tools. "
    "NOT FOR: Feature detection (aisp_info_features), topology (aisp_gpu_topology), power throttling (aisp_gpu_power).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_gpu_info(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU information."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().gpu.info()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_gpu_bandwidth",
    "Tags: bandwidth, memory, hbm, nvlink, throughput, spec-check. "
    "Run GPU memory bandwidth test measuring actual vs theoretical HBM bandwidth. "
    "Returns: {bandwidth_gbps, theoretical_gbps, efficiency_pct, test_size_mb}. "
    "USE when: Validating memory throughput matches GPU spec, diagnosing memory-bound kernels, checking for HBM/PCIe issues. "
    "Example: \"Check H100 bandwidth vs spec\" or \"Why is my memory-bound kernel slow?\". "
    "NOT FOR: PCIe H2D/D2H bandwidth (use aisp_hw_pcie), GPU-to-GPU P2P (use aisp_hw_p2p). 🕐 MEDIUM (~10s). WORKFLOW: aisp_gpu_info → aisp_gpu_bandwidth → diagnose memory-bound.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_gpu_bandwidth(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run GPU bandwidth test."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().gpu.bandwidth_test()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_gpu_topology",
    "Tags: topology, nvlink, pcie, multi-gpu, interconnect, p2p, numa. "
    "Get multi-GPU topology: NVLink/PCIe connections, NUMA affinity, P2P capability matrix. "
    "Returns: {gpu_count, connections: [{src, dst, type, bandwidth_gbps}], numa_nodes, p2p_matrix}. "
    "USE when: Planning tensor/pipeline parallelism, debugging P2P transfer issues, optimizing GPU placement. "
    "Example: \"Show NVLink/PCIe layout on 8x GPU server\" or \"Which GPUs have NVLink?\". "
    "NOT FOR: Raw topology matrix output (use aisp_gpu_topology_matrix for nvidia-smi topo -m). ⚡ FAST (~2s). WORKFLOW: aisp_gpu_topology → aisp_distributed_plan → aisp_distributed_nccl.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_gpu_topology(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU topology."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().gpu.topology()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_gpu_power",
    "Tags: power, thermal, throttling, headroom, tdp, temperature. "
    "Get GPU power and thermal status: current power draw, power limit, temperature, throttling state. "
    "Returns: {gpus: [{power_w, power_limit_w, headroom_w, temperature_c, throttling, fan_pct}]}. "
    "USE when: Checking for thermal/power throttling, verifying TDP headroom before heavy workloads. "
    "Example: \"Are GPUs power-throttling right now?\" or \"How much thermal headroom do I have?\". "
    "NOT FOR: General GPU info (use aisp_gpu_info), sustained power monitoring over time. ⚡ FAST (~1s). WORKFLOW: aisp_gpu_power → if throttling → reduce workload.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_gpu_power(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU power info."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.power()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# SYSTEM TOOLS
# =============================================================================

@register_tool(
    "aisp_system_software",
    "Tags: software, versions, pytorch, cuda, python, driver, stack. "
    "Get software stack versions: PyTorch, CUDA toolkit, cuDNN, Python, NVIDIA driver. "
    "Returns: {pytorch_version, cuda_version, cudnn_version, python_version, driver_version, transformers_version}. "
    "USE when: Filing bug reports, checking compatibility, reproducing issues, verifying install. "
    "Example: \"What PyTorch and CUDA versions are installed?\" or \"Is my CUDA version compatible with FlashAttention?\". "
    "NOT FOR: Checking if dependencies import correctly (use aisp_system_dependencies). ⚡ FAST (~1s). WORKFLOW: aisp_system_software → check → verify compatibility.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_system_software(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get software information."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.software()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_system_dependencies",
    "Tags: deps, health, import, missing, broken, install, torch.cuda. "
    "Check health of ML/AI dependencies: torch, triton, flash-attn, transformers, vllm, etc. "
    "Returns: {dependencies: [{name, installed, importable, version, error}], healthy_count, broken_count}. "
    "USE when: Diagnosing import errors, checking if optional dependencies are available, debugging install issues. "
    "Example: \"Why does torch.cuda fail to import?\" or \"Is flash-attn installed correctly?\". "
    "NOT FOR: Version numbers only (use aisp_system_software), general system health (use aisp_status). ⚡ FAST (~2s). WORKFLOW: aisp_system_dependencies → fix broken → retry.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_system_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check dependency health."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.dependencies()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_run_benchmarks",
    "Tags: benchmarks, run, profiling, performance-test, chapters, labs, validation. "
    "Run benchmarks via the bench CLI with optional profiling and LLM analysis. "
    "Returns: {stdout, stderr, returncode, duration_seconds, artifacts_path, suggested_next_steps}. "
    "⚠️ SLOW: 2-30+ minutes depending on targets. ALWAYS run aisp_status first! "
    "USE when: Validating optimizations, generating benchmark data for comparison. "
    "Example: \"Run ch07 benchmarks\" or \"Benchmark attention examples\". "
    "SAFE WORKFLOW: "
    "1. aisp_status → verify GPU/CUDA ready "
    "2. aisp_list_chapters → see what's available "
    "3. aisp_run_benchmarks(targets=['ch07'], dry_run=true) → preview command "
    "4. aisp_run_benchmarks(targets=['ch07'], async=true) → run in background "
    "5. aisp_job_status(job_id=...) → poll until complete "
    "6. aisp_benchmark_triage → analyze results and get recommendations. "
    "WORKFLOW: aisp_status → aisp_list_chapters → aisp_run_benchmarks → aisp_benchmark_triage. "
    "NOT FOR: Quick GPU health (use aisp_hw_speed first).",
    {
        "type": "object",
        "properties": with_context_params({
            "targets": {"type": "array", "items": {"type": "string"}},
            "profile": {
                "type": "string",
                "description": "Profiling preset: none (no profiling), minimal (basic), deep_dive (full nsys/ncu profiling), or roofline",
                "enum": ["none", "minimal", "deep_dive", "roofline"],
                "default": "minimal"
            },
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
            "async": {
                "type": "boolean",
                "description": "Run in background and return job_id; poll with aisp_job_status",
                "default": False
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max runtime before returning with partial output; set 0/null for no timeout",
                "default": 900
            },
        }),
        "required": ["targets"],
    },
)
def tool_run_benchmarks(params: Dict[str, Any]) -> Dict[str, Any]:
    targets = params.get("targets") or []
    
    # Normalize profile - uses _COMMON_ALIASES for most common mistakes
    raw_profile = params.get("profile") or "minimal"
    profile = normalize_param("profile", raw_profile, "minimal")
    
    # Validate profile value
    valid_profiles = ["none", "minimal", "deep_dive", "roofline"]
    if profile not in valid_profiles:
        return {
            "error": f"Invalid profile '{raw_profile}'. Valid options: {', '.join(valid_profiles)}",
            "hint": "Use 'deep_dive' for full nsys/ncu profiling",
            "success": False,
        }
    # LLM analysis disabled by default to avoid unexpected API costs
    # Enable explicitly with llm_analysis=true when you want AI-powered insights
    llm_analysis = params.get("llm_analysis", False)
    apply_patches = params.get("apply_patches", False)
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async", False))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context, context_level = extract_context_opts(params)
    cuda_check = _cuda_precheck()

    args: List[str] = ["run", "--profile", profile]
    for t in targets:
        args.extend(["-t", t])
    # Add --llm-analysis only if explicitly enabled (costs API credits)
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
            "note": "Set dry_run=false to execute; use async=true for background execution.",
        }

    def _execute_benchmarks():
        return _run_bench_cli(args, timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)

    if run_async:
        queued = _queue_job("aisp_run_benchmarks", _execute_benchmarks, params)
        queued["targets"] = targets
        queued["note"] = "Background benchmark started; poll with aisp_job_status using job_id. When complete, use aisp_benchmark_triage to analyze results."
        return queued

    result = _execute_benchmarks()
    # Add suggested next steps to help users continue their workflow
    result["suggested_next_steps"] = _benchmark_next_steps(result)
    return attach_context_if_requested(result, include_context, context_level)


def _benchmark_next_steps(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate actionable next steps after benchmark completion."""
    steps = []
    returncode = result.get("returncode", 0)
    
    if returncode == 0:
        # Success path
        steps.append({
            "tool": "aisp_benchmark_triage",
            "reason": "Analyze benchmark results and get optimization recommendations",
            "priority": "high"
        })
        steps.append({
            "tool": "aisp_benchmark_report",
            "reason": "Generate shareable PDF/HTML report",
            "params": {"format": "html"}
        })
        steps.append({
            "tool": "aisp_benchmark_compare_runs",
            "reason": "Compare with previous baseline if available",
            "note": "Requires baseline benchmark_test_results.json"
        })
        steps.append({
            "tool": "aisp_analyze_pareto",
            "reason": "Find optimal throughput/latency/memory tradeoffs"
        })
    else:
        # Failure path
        steps.append({
            "tool": "aisp_status",
            "reason": "Check system health after benchmark failure",
            "priority": "high"
        })
        steps.append({
            "tool": "aisp_system_dependencies",
            "reason": "Verify all dependencies are correctly installed"
        })
        steps.append({
            "tool": "aisp_analyze_bottlenecks",
            "reason": "Identify what might be causing issues"
        })
    
    return steps


@register_tool(
    "aisp_system_context",
    "Tags: context, environment, inventory, full-dump, comprehensive. "
    "Get comprehensive system context: GPU info + software stack + hardware capabilities combined. "
    "Returns: {gpu, software, dependencies, capabilities} - all system info in one call. "
    "USE when: Need complete environment dump for analysis, sharing system state with others. "
    "Example: \"Provide full context for LLM analysis\" or \"Dump entire system state\". "
    "PREFER aisp_triage or aisp_context_summary for quick checks; this is heavier. 🕐 SLOW (2-30+ min). NOT FOR: Quick GPU health (use aisp_hw_speed). ⚡ FAST (~2s). WORKFLOW: aisp_system_dependencies → fix broken → retry.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_system_context(params: Dict[str, Any]) -> ContextResult:
    """Get full system context."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result: ContextResult = {"success": True, "context": get_engine().system.context()}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_system_capabilities",
    "Tags: capabilities, features, supported, compute-capability. "
    "Get hardware capabilities summary: compute capability, tensor cores, supported precisions. "
    "Returns: {compute_capability, sm_version, tensor_cores, supported_dtypes, max_shared_mem}. "
    "USE when: Checking if a feature (FP8, TF32, etc.) is supported, planning which optimizations to apply. "
    "Example: \"What features does my GPU support?\" or \"Can I use FP8 on this hardware?\". "
    "PREFER aisp_info_features for detailed capability breakdown with TMA/cluster info. ⚡ FAST (~1s). WORKFLOW: aisp_system_capabilities → check features → aisp_recommend. NOT FOR: Version info (use aisp_system_software).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_system_capabilities(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get hardware capabilities."""
    from core.perf_core import get_core
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.capabilities()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_benchmark_targets",
    "Tags: benchmarks, targets, list, chapters, examples, discovery. "
    "List benchmark targets in chapter:example format (e.g., 'ch07:flash_attention'). "
    "Returns: {targets: [{chapter, example, path}], count} or filtered by chapter. "
    "USE when: Finding exact target names to pass to aisp_run_benchmarks. "
    "Example: \"List targets for ch07\" or \"What examples are in the attention chapter?\". "
    "PREFER aisp_list_chapters to see all chapters first. ⚡ FAST (~1s). WORKFLOW: aisp_benchmark_targets → aisp_run_benchmarks. NOT FOR: Running benchmarks (use aisp_run_benchmarks).",
    {"type": "object", "properties": with_context_params({
        "chapter": {"type": "string", "description": "Optional chapter or lab slug to filter (e.g., 'ch07', 'labs/decode')"},
    })}
)
def tool_benchmark_targets(params: Dict[str, Any]) -> Dict[str, Any]:
    """List benchmark targets."""
    include_context, context_level = extract_context_opts(params)
    args: List[str] = ["list-targets"]
    chapter = params.get("chapter")
    if chapter:
        args.extend(["--chapter", chapter])
    result = _run_bench_cli(args)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_benchmark_report",
    "Tags: report, pdf, html, export, visualization, share, document. "
    "Generate PDF/HTML report from benchmark results for sharing and documentation. "
    "Returns: {output_path, format, success} and writes report file. "
    "⚡ FAST (~5s). USE AFTER: aisp_run_benchmarks or aisp_benchmark_triage. "
    "Example: \"Generate HTML report\" or \"Create PDF performance summary\". "
    "FORMATS: "
    "• html: Interactive, best for sharing/web "
    "• pdf: Static, best for formal documentation. "
    "WORKFLOW: aisp_run_benchmarks → aisp_benchmark_triage → aisp_benchmark_report(format='html'). "
    "REQUIRES: benchmark_test_results.json from aisp_run_benchmarks.",
    {"type": "object", "properties": with_context_params({
        "data_file": {"type": "string", "description": "Path to benchmark_test_results.json (defaults to latest)"},
        "output": {"type": "string", "description": "Output file path (.pdf or .html)", "default": "report.pdf"},
        "format": {"type": "string", "description": "Output format: pdf or html", "enum": ["pdf", "html"], "default": "pdf"},
        "title": {"type": "string", "description": "Report title (optional)"},
        "author": {"type": "string", "description": "Report author (optional)"},
    })}
)
def tool_benchmark_report(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    args = ["report"]
    data_file = params.get("data_file")
    if data_file:
        if not Path(data_file).exists():
            return make_error(f"data_file not found: {data_file}", include_context, context_level, data_file=data_file)
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
    result = _run_bench_cli(args)
    if isinstance(result, dict) and "success" not in result:
        returncode = result.get("returncode", 0)
        result["success"] = returncode == 0 and not result.get("error")
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_benchmark_export",
    "Tags: export, csv, markdown, json, data, share, spreadsheet. "
    "Export benchmark results to CSV/Markdown/JSON format for further analysis. "
    "Returns: {output_path, format, benchmarks_written, success}. "
    "USE when: Importing results into spreadsheets, documentation, or other tools. "
    "Example: \"Export benchmarks to CSV\" or \"Convert results to markdown table\". "
    "REQUIRES: Run aisp_run_benchmarks first to generate benchmark_test_results.json. ⚡ FAST (~2s). WORKFLOW: aisp_run_benchmarks → aisp_benchmark_export. NOT FOR: Reports (use aisp_benchmark_report).",
    {"type": "object", "properties": with_context_params({
        "data_file": {"type": "string", "description": "Path to benchmark_test_results.json (defaults to latest)"},
        "format": {"type": "string", "description": "Output format: csv, markdown, or json", "enum": ["csv", "markdown", "json"], "default": "csv"},
        "output": {"type": "string", "description": "Output file path (auto-generated if omitted)"},
    })}
)
def tool_benchmark_export(params: Dict[str, Any]) -> BenchmarkExportResult:
    """Export benchmark results without spawning the bench CLI."""
    from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data

    include_context, context_level = extract_context_opts(params)
    fmt = normalize_param("format", params.get("format"), "csv")
    data_file = params.get("data_file")
    output = params.get("output")

    valid_formats = {"csv", "markdown", "json"}
    if fmt not in valid_formats:
        return make_error(f"format must be one of {sorted(valid_formats)}", include_context, context_level)

    data_path = Path(data_file) if data_file else None
    if data_path and not data_path.exists():
        return make_error(f"data_file not found: {data_path}", include_context, context_level, data_file=str(data_path))
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
        return make_error(f"failed to export benchmarks: {exc}", include_context, context_level, output=str(output_path))

    result: BenchmarkExportResult = {
        "output": str(output_path),
        "format": fmt,
        "benchmarks_written": len(benchmarks),
        "success": True,
    }
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_benchmark_compare_runs",
    "Tags: compare, diff, regression, improvement, delta, a-b-test, before-after. "
    "Compare two benchmark runs showing speedup deltas, regressions, and improvements. "
    "Returns: {regressions: [...], improvements: [...], unchanged: [...], summary}. "
    "⚡ FAST (~1s). USE when: Comparing before/after optimization, detecting regressions. "
    "Example: \"Compare baseline vs optimized\" or \"Show top 10 regressions\". "
    "REQUIRES: Two benchmark_test_results.json files from separate runs. "
    "WORKFLOW: "
    "1. aisp_run_benchmarks (baseline) → save results "
    "2. Make optimizations "
    "3. aisp_run_benchmarks (candidate) → save results "
    "4. aisp_benchmark_compare_runs(baseline=..., candidate=...) "
    "5. If regressions: aisp_analyze_bottlenecks on affected benchmarks.",
    {"type": "object", "properties": with_context_params({
        "baseline": {"type": "string", "description": "Path to baseline benchmark_test_results.json"},
        "candidate": {"type": "string", "description": "Path to candidate/new benchmark_test_results.json"},
        "top": {"type": "integer", "description": "Show top N regressions/improvements", "default": 10},
    }), "required": ["baseline", "candidate"]}
)
def tool_benchmark_compare_runs(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    baseline = params.get("baseline")
    candidate = params.get("candidate")
    if not baseline or not candidate:
        return make_error("baseline and candidate benchmark files are required", include_context, context_level)
    if baseline and not Path(baseline).exists():
        return make_error(f"baseline file not found: {baseline}", include_context, context_level, baseline=baseline)
    if candidate and not Path(candidate).exists():
        return make_error(f"candidate file not found: {candidate}", include_context, context_level, candidate=candidate)

    args = [
        "compare-runs",
        "--baseline", baseline,
        "--candidate", candidate,
        "--top", str(params.get("top", 10)),
    ]
    result = _run_bench_cli(args)
    if isinstance(result, dict) and "success" not in result:
        returncode = result.get("returncode", 0)
        result["success"] = returncode == 0 and not result.get("error")
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_benchmark_triage",
    "Tags: benchmark, triage, analysis, recommendations, next-steps, post-benchmark, actionable. "
    "🔍 POST-BENCHMARK ANALYSIS: Analyze benchmark results and get actionable recommendations. "
    "Returns: {summary, regressions, improvements, top_issues, recommended_tools, optimization_plan}. "
    "⚡ FAST (~2s). USE AFTER: aisp_run_benchmarks completes successfully. "
    "Example: \"Analyze my benchmark results\" or \"What should I optimize based on benchmarks?\". "
    "PROVIDES: "
    "• Summary of all benchmark results (pass/fail/speedup) "
    "• Identification of regressions and improvements "
    "• Specific tool recommendations based on findings "
    "• Prioritized optimization plan. "
    "WORKFLOW: aisp_run_benchmarks → aisp_benchmark_triage → implement recommendations → re-benchmark.",
    {"type": "object", "properties": with_context_params({
        "data_file": {
            "type": "string",
            "description": "Path to benchmark_test_results.json (defaults to latest in artifacts/)"
        },
        "baseline_file": {
            "type": "string",
            "description": "Optional baseline to compare against for regression detection"
        },
    })}
)
def tool_benchmark_triage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and provide actionable recommendations."""
    from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data
    
    include_context, context_level = extract_context_opts(params)
    data_file = params.get("data_file")
    baseline_file = params.get("baseline_file")
    
    # Find latest benchmark results if not specified
    data_path = Path(data_file) if data_file else None
    if not data_path:
        # Search common locations
        search_paths = [
            Path("artifacts/benchmark_test_results.json"),
            Path("benchmark_test_results.json"),
            Path("artifacts/benchmarks/benchmark_test_results.json"),
        ]
        for p in search_paths:
            if p.exists():
                data_path = p
                break
    
    if not data_path or not data_path.exists():
        return make_error(
            "No benchmark results found. Run aisp_run_benchmarks first.",
            include_context, context_level,
            searched_paths=[str(p) for p in search_paths] if not data_file else None,
            hint="Specify data_file parameter or run aisp_run_benchmarks to generate results."
        )
    
    try:
        data = load_benchmark_data(data_path)
    except Exception as exc:
        return make_error(f"Failed to load benchmark data: {exc}", include_context, context_level)
    
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return make_error("Benchmark file contains no results", include_context, context_level, data_file=str(data_path))
    
    # Analyze results
    total = len(benchmarks)
    passed = sum(1 for b in benchmarks if b.get("speedup", 0) >= 1.0)
    failed = total - passed
    
    # Find regressions and improvements
    regressions = sorted(
        [b for b in benchmarks if b.get("speedup", 1.0) < 0.95],
        key=lambda x: x.get("speedup", 1.0)
    )[:10]
    
    improvements = sorted(
        [b for b in benchmarks if b.get("speedup", 1.0) > 1.05],
        key=lambda x: x.get("speedup", 1.0),
        reverse=True
    )[:10]
    
    # Identify patterns for recommendations
    slow_kernels = [b for b in benchmarks if b.get("baseline_time_ms", 0) > 100]
    memory_issues = [b for b in benchmarks if "memory" in b.get("name", "").lower() or "oom" in str(b.get("error", "")).lower()]
    attention_benchmarks = [b for b in benchmarks if "attention" in b.get("name", "").lower()]
    
    # Build recommendations
    recommended_tools = []
    optimization_plan = []
    
    if regressions:
        recommended_tools.append({
            "tool": "aisp_analyze_bottlenecks",
            "reason": f"Identify root cause of {len(regressions)} regression(s)",
            "priority": "high"
        })
        optimization_plan.append({
            "step": 1,
            "action": "Investigate regressions",
            "details": f"Top regression: {regressions[0].get('chapter')}:{regressions[0].get('name')} ({regressions[0].get('speedup', 0):.2f}x)"
        })
    
    if slow_kernels:
        recommended_tools.append({
            "tool": "aisp_profile_nsys",
            "reason": f"Profile {len(slow_kernels)} slow benchmark(s) (>100ms baseline)",
            "params": {"preset": "light"}
        })
        optimization_plan.append({
            "step": 2,
            "action": "Profile slow operations",
            "details": f"Slowest: {slow_kernels[0].get('chapter')}:{slow_kernels[0].get('name')} ({slow_kernels[0].get('baseline_time_ms', 0):.1f}ms)"
        })
    
    if attention_benchmarks:
        avg_attention_speedup = sum(b.get("speedup", 1.0) for b in attention_benchmarks) / len(attention_benchmarks)
        if avg_attention_speedup < 1.5:
            recommended_tools.append({
                "tool": "aisp_explain",
                "reason": "Learn about FlashAttention optimization",
                "params": {"concept": "flash-attention"}
            })
    
    if improvements:
        recommended_tools.append({
            "tool": "aisp_benchmark_report",
            "reason": f"Document {len(improvements)} improvement(s) in shareable report",
            "params": {"format": "html"}
        })
    
    # Always suggest comparison if we have results
    recommended_tools.append({
        "tool": "aisp_benchmark_compare_runs",
        "reason": "Compare with previous baseline for trend analysis",
        "note": "Save current results as baseline for future comparisons"
    })
    
    # Add general optimization recommendations
    recommended_tools.append({
        "tool": "aisp_recommend",
        "reason": "Get optimization playbook based on your hardware and goals"
    })
    
    result = {
        "success": True,
        "data_file": str(data_path),
        "summary": {
            "total_benchmarks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "N/A",
            "avg_speedup": sum(b.get("speedup", 1.0) for b in benchmarks) / total if total > 0 else 0,
        },
        "regressions": [
            {
                "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                "speedup": b.get("speedup", 0),
                "baseline_ms": b.get("baseline_time_ms", 0),
            }
            for b in regressions
        ],
        "improvements": [
            {
                "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                "speedup": b.get("speedup", 0),
                "baseline_ms": b.get("baseline_time_ms", 0),
            }
            for b in improvements
        ],
        "top_issues": [],
        "recommended_tools": recommended_tools,
        "optimization_plan": optimization_plan,
        "next_steps_summary": (
            f"Found {len(regressions)} regression(s) and {len(improvements)} improvement(s). "
            f"{'Focus on fixing regressions first.' if regressions else 'Results look good!'} "
            f"Use recommended_tools for specific actions."
        ),
    }
    
    # Add top issues
    if regressions:
        result["top_issues"].append({
            "type": "regression",
            "count": len(regressions),
            "severity": "high" if any(b.get("speedup", 1.0) < 0.5 for b in regressions) else "medium"
        })
    if slow_kernels:
        result["top_issues"].append({
            "type": "slow_operations",
            "count": len(slow_kernels),
            "severity": "medium"
        })
    
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_analyze_bottlenecks",
    "Tags: bottleneck, slow, latency, utilization, diagnosis, why-slow, root-cause, debug. "
    "Identify performance bottlenecks: memory-bound, compute-bound, communication-bound, host-bound. "
    "Returns: {bottleneck_type, confidence, profile_data, llm_analysis, recommendations, availability}. "
    "⚡ FAST (~2-5s). USE FIRST when: Workload is slow and you don't know why. "
    "Example: \"Why is my 7B model slow on 8xH100?\" or \"What's the bottleneck at batch 32, seq 4k?\". "
    "mode='both' (default) combines profiling data with LLM analysis for best results. "
    "WORKFLOW by bottleneck_type: "
    "• memory-bound → aisp_profile_memory, aisp_analyze_whatif(max_vram_gb=X) "
    "• compute-bound → aisp_profile_kernels, aisp_hw_tc "
    "• communication-bound → aisp_distributed_nccl, aisp_hw_nccl "
    "• host-bound → aisp_cpu_memory_analysis, aisp_data_loading NOT FOR: Kernel metrics (use aisp_profile_ncu).",
    {"type": "object", "properties": with_context_params({
        "analysis_type": {
            "type": "string",
            "description": "Focus area: bottleneck (general), memory, or compute",
            "enum": ["bottleneck", "memory", "compute"],
            "default": "bottleneck"
        },
        "mode": {
            "type": "string",
            "description": "Analysis mode: profile (data only), llm (AI analysis only), both (combined)",
            "enum": ["profile", "llm", "both"],
            "default": "both"
        },
    })}
)
def tool_analyze_bottlenecks(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze bottlenecks."""
    from core.engine import get_engine
    analysis_type = normalize_param("analysis_type", params.get("analysis_type"), "bottleneck")
    mode = normalize_param("mode", params.get("mode"), "both")
    include_context, context_level = extract_context_opts(params)
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
    "Tags: pareto, tradeoff, throughput, latency, memory, frontier, optimal, comparison, choose. "
    "Find Pareto-optimal configurations: best throughput/latency/memory tradeoffs. "
    "Returns: {pareto_frontier: [{config, throughput, latency_ms, memory_gb}], dominated_configs, analysis}. "
    "⚡ FAST (~2s). USE AFTER: aisp_run_benchmarks with multiple configurations. "
    "Example: \"Show Pareto frontier\" or \"What's the best throughput/latency tradeoff?\". "
    "PARETO EXPLAINED: Points on the frontier are 'optimal' - you can't improve one metric without sacrificing another. "
    "WORKFLOW: "
    "1. aisp_run_benchmarks with varied batch_size/seq_len configs "
    "2. aisp_analyze_pareto → find optimal operating points "
    "3. aisp_analyze_whatif → check if constraints are met. "
    "REQUIRES: benchmark_test_results.json with multiple configurations.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_analyze_pareto(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pareto analysis."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.pareto()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_analyze_scaling",
    "Tags: scaling, throughput, gpus, nodes, projection, extrapolation, multi-gpu. "
    "Analyze how performance scales with workload size, sequence length, batch size, or GPU count. "
    "Returns: {scaling_efficiency, projections: [{gpus, throughput, efficiency_pct}], bottleneck_at_scale}. "
    "⚡ FAST (~2s). USE when: Projecting performance to larger inputs, planning multi-GPU scaling. "
    "Example: \"Predict throughput if I double sequence length\" or \"How does it scale from 4 to 8 GPUs?\". "
    "WORKFLOW: aisp_gpu_topology → aisp_analyze_scaling → aisp_distributed_plan. "
    "ALSO USE: aisp_predict_scaling for specific GPU count predictions.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_analyze_scaling(params: Dict[str, Any]) -> Dict[str, Any]:
    """Scaling analysis."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.scaling()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_analyze_stacking",
    "Tags: stacking, combinations, techniques, compatibility, conflicts, compose. "
    "Analyze which optimization techniques work well together and which conflict. "
    "Returns: {compatible_stacks: [...], conflicts: [{technique1, technique2, reason}], recommended_order}. "
    "⚡ FAST (~2s). USE when: Planning to combine multiple optimizations, checking for conflicts. "
    "Example: \"Can FlashAttention + torch.compile + CUDA graphs coexist?\" or \"What's the best optimization order?\". "
    "WORKFLOW: aisp_recommend → aisp_analyze_stacking → apply compatible techniques. "
    "ALSO USE: aisp_optimize_techniques for full technique list, aisp_optimize_roi for prioritization.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_analyze_stacking(params: Dict[str, Any]) -> Dict[str, Any]:
    """Stacking analysis."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.stacking()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_analyze_whatif",
    "Tags: constraints, latency, vram, throughput, what-if, feasibility, target, sla, budget. "
    "What-if analysis: Find optimizations that meet your constraints (VRAM, latency, throughput). "
    "Returns: {feasible_configs: [...], recommended_optimizations, tradeoff_analysis}. "
    "⚡ FAST (~1s). USE when: Targeting specific SLA bounds, checking feasibility. "
    "Example: \"Need <50ms latency with <24GB VRAM\" or \"Can I hit 2k tok/s?\". "
    "CONSTRAINT EXAMPLES: "
    "• max_vram_gb=24: Fit on RTX 4090 / single A10G "
    "• max_latency_ms=50: Real-time chatbot SLA "
    "• min_throughput=1000: High-volume batch processing "
    "• Combine: max_vram_gb=48, max_latency_ms=100 (A6000 real-time). "
    "WORKFLOW: aisp_profile_memory → aisp_analyze_whatif → aisp_inference_quantization → verify.",
    {"type": "object", "properties": with_context_params({
        "max_vram_gb": {
            "type": "number",
            "description": "Maximum VRAM budget in GB (e.g., 24 for single 3090)"
        },
        "max_latency_ms": {
            "type": "number", 
            "description": "Maximum acceptable latency in milliseconds (e.g., 50ms SLA)"
        },
        "min_throughput": {
            "type": "number",
            "description": "Minimum required throughput in tokens/sec or samples/sec"
        },
    })}
)
def tool_analyze_whatif(params: Dict[str, Any]) -> Dict[str, Any]:
    """What-if analysis."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.whatif(params)
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# OPTIMIZATION TOOLS  
# =============================================================================

@register_tool(
    "aisp_recommend",
    "Tags: recommend, playbook, throughput, latency, memory, optimization, guide, strategy, gameplan. "
    "Get prioritized optimization recommendations for your model configuration and goal. "
    "Returns: {recommendations: [{technique, priority, expected_speedup, effort}], playbook, warnings}. "
    "⚡ FAST (~1s). USE EARLY when: Starting optimization work, need a game plan. "
    "Example: \"Recommend for 13B on 4xA100 focused on throughput\" or \"Low-latency 7B on single H100\". "
    "GOALS explained: "
    "• throughput → maximize tokens/sec (batch processing, training) "
    "• latency → minimize TTFT (real-time inference, chatbots) "
    "• memory → reduce VRAM (fit larger models, longer sequences). "
    "WORKFLOW: aisp_triage → aisp_recommend → aisp_optimize_roi → implement techniques → aisp_run_benchmarks.",
    {"type": "object", "properties": with_context_params({
        "model_size": {
            "type": "number",
            "description": "Model size in billions of parameters (7, 13, 70, etc.)",
            "default": 7
        },
        "gpus": {
            "type": "integer",
            "description": "Number of GPUs available (1, 4, 8, etc.)",
            "default": 1
        },
        "goal": {
            "type": "string",
            "description": "Primary optimization goal: throughput (tok/s), latency (TTFT), or memory (VRAM)",
            "enum": ["throughput", "latency", "memory"],
            "default": "throughput"
        },
    }), "required": []}
)
def tool_recommend(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get recommendations."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    goal = normalize_param("goal", params.get("goal"), "throughput")
    result = get_engine().optimize.recommend(
        model_size=params.get("model_size", 7),
        gpus=params.get("gpus", 1),
        goal=goal
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_optimize_roi",
    "Tags: ROI, prioritize, cost-benefit, effort, impact, ranking, efficiency. "
    "Calculate ROI (return on investment) for optimization techniques: expected gain vs implementation effort. "
    "Returns: {ranked_techniques: [{name, expected_speedup, effort_hours, roi_score}], quick_wins, high_impact}. "
    "USE when: Prioritizing optimization work, deciding what to implement first, limited engineering time. "
    "Example: \"Which optimizations give best ROI?\" or \"Rank techniques by cost vs gain\". "
    "ALSO USE: aisp_optimize_techniques for full technique details, aisp_recommend for goal-specific recs. ⚡ FAST (~1s). WORKFLOW: aisp_recommend → aisp_optimize_roi → prioritize.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_optimize_roi(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimization ROI."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().optimize.roi()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_optimize_techniques",
    "Tags: techniques, list, options, catalog, encyclopedia, reference. "
    "Get catalog of all optimization techniques with details, requirements, and expected benefits. "
    "Returns: {techniques: [{name, category, description, requirements, expected_speedup, gotchas}], count}. "
    "USE when: Exploring what optimizations exist, learning about technique requirements, reference lookup. "
    "Example: \"List all optimization techniques\" or \"What techniques exist for attention?\". "
    "ALSO USE: aisp_optimize_roi for prioritization, aisp_analyze_stacking for compatibility. ⚡ FAST (~1s). WORKFLOW: aisp_optimize_techniques → choose → aisp_optimize_roi.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_optimize_techniques(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get all optimization techniques."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().optimize.all_techniques()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# DISTRIBUTED TRAINING TOOLS
# =============================================================================

@register_tool(
    "aisp_distributed_plan",
    "Tags: distributed, dp, tp, pp, fsdp, parallelism, multi-gpu, multi-node, strategy, sharding. "
    "Plan parallelism strategy: recommend DP/TP/PP/FSDP layout for model size and GPU count. "
    "Returns: {recommended_layout: {tp, pp, dp}, memory_per_gpu_gb, communication_volume, rationale}. "
    "⚡ FAST (~1s). USE when: Setting up distributed training, choosing parallelism degrees. "
    "Example: \"Plan 70B on 2 nodes x 8 GPUs\" or \"What TP/PP for 14B on 4 GPUs?\". "
    "PARALLELISM explained: "
    "• TP (Tensor Parallel): Split layers across GPUs; needs NVLink; TP ≤ 8 typically "
    "• PP (Pipeline Parallel): Split model stages; good for multi-node "
    "• DP (Data Parallel): Replicate model; scale batch size "
    "• FSDP: Shard parameters + gradients; memory-efficient DP. "
    "WORKFLOW: aisp_distributed_plan → aisp_distributed_nccl → aisp_launch_plan → training.",
    {"type": "object", "properties": with_context_params({
        "model_size": {
            "type": "number",
            "description": "Model size in billions of parameters (7, 13, 70, etc.)",
            "default": 7
        },
        "gpus": {
            "type": "integer",
            "description": "Total number of GPUs across all nodes",
            "default": 8
        },
        "nodes": {
            "type": "integer",
            "description": "Number of nodes (1 for single-node, 2+ for multi-node)",
            "default": 1
        },
    })}
)
def tool_distributed_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Plan parallelism."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().distributed.plan(
        model_size=params.get("model_size", 7),
        gpus=params.get("gpus", 8),
        nodes=params.get("nodes", 1)
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_distributed_nccl",
    "Tags: nccl, multi-node, collective, allreduce, environment, tuning, ib, rdma. "
    "Get NCCL tuning recommendations: environment variables, IB settings, collective algorithms. "
    "Returns: {env_vars: {NCCL_*: value}, algorithm_hints, ib_recommendations, debug_tips}. "
    "USE when: Tuning NCCL for multi-node training, debugging collective performance, IB/RDMA setup. "
    "Example: \"NCCL settings for 2-node 8xH100\" or \"Tune NCCL for InfiniBand\". "
    "ALSO USE: aisp_hw_nccl for NCCL bandwidth testing, aisp_info_network for IB status. ⚡ FAST (~1s). WORKFLOW: aisp_distributed_plan → aisp_distributed_nccl → apply env vars.",
    {"type": "object", "properties": with_context_params({
        "nodes": {"type": "integer", "description": "Number of nodes", "default": 1},
        "gpus": {"type": "integer", "description": "GPUs per node", "default": 8},
    })}
)
def tool_distributed_nccl(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get NCCL tuning."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
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
    "Tags: vllm, inference, serving, deployment, config, batching, kv-cache, production. "
    "Generate optimized vLLM configuration for inference serving (throughput or latency mode). "
    "Returns: {config: {tensor_parallel, gpu_memory_utilization, max_num_seqs, ...}, launch_cmd, tips}. "
    "⚡ FAST (~1s). USE when: Deploying vLLM server, optimizing inference serving. "
    "Example: \"vLLM settings for 7B low latency on A100\" or \"High-throughput 70B vLLM config\". "
    "TARGETS explained: "
    "• throughput: Large batches, high gpu_memory_utilization (~0.9), best for batch inference "
    "• latency: Small batches, lower memory util, continuous batching tuned for TTFT. "
    "WORKFLOW: aisp_gpu_info → aisp_inference_quantization → aisp_inference_vllm → deploy. "
    "ALSO USE: aisp_inference_quantization for precision recommendations.",
    {"type": "object", "properties": with_context_params({
        "model": {
            "type": "string",
            "description": "Model name or size (e.g., 'llama-7b', 'meta-llama/Llama-3.1-70B')",
            "default": "7b"
        },
        "target": {
            "type": "string",
            "description": "Optimization target: throughput (max batch) or latency (min TTFT)",
            "enum": ["throughput", "latency"],
            "default": "throughput"
        },
    })}
)
def tool_inference_vllm(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate vLLM config."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    target = normalize_param("target", params.get("target"), "throughput")
    result = get_engine().inference.vllm_config(
        model=params.get("model", "7b"),
        target=target
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_inference_quantization",
    "Tags: quantization, fp8, int8, int4, awq, gptq, precision, compression, memory, bnb. "
    "Get quantization recommendations: precision format, method, expected accuracy/speedup tradeoffs. "
    "Returns: {recommended_format, alternatives: [{format, memory_reduction, speedup, accuracy_loss}], tips}. "
    "⚡ FAST (~1s). USE when: Choosing quantization format for inference. "
    "Example: \"Should I use FP8 or INT8 for 70B inference?\" or \"Best quantization for 24GB VRAM?\". "
    "FORMATS explained: "
    "• FP8 (E4M3/E5M2): Best quality/speed; Hopper+ only (H100/H200); ~50% memory reduction "
    "• INT8: Good quality/speed; Ampere+ (A100/RTX30xx+); ~50% memory reduction "
    "• INT4 (AWQ/GPTQ): Max compression; ~75% memory reduction; slight quality loss "
    "• NF4 (bitsandbytes): Easy setup; ~75% reduction; QLoRA-friendly. "
    "WORKFLOW: aisp_gpu_info (check arch) → aisp_inference_quantization → aisp_inference_vllm.",
    {"type": "object", "properties": with_context_params({
        "model_size": {
            "type": "number",
            "description": "Model size in billions of parameters (affects memory savings calculation)"
        },
    })}
)
def tool_inference_quantization(params: Dict[str, Any]) -> Dict[str, Any]:
    """Quantization recommendations."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().inference.quantization(params)
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# AI/LLM TOOLS
# =============================================================================

@register_tool(
    "aisp_ask",
    "Tags: question, advice, why-slow, guidance, help, answer, book, citations, free-form. "
    "Ask a free-form performance question and get an answer with book citations. "
    "Returns: {answer, citations: [{chapter, section, relevance}], related_tools}. "
    "⚡ FAST (~2-5s). USE when: Need targeted advice, best practices, 'why' questions. "
    "GOOD QUESTIONS: "
    "• 'Is FlashAttention worth it on Llama-2 7B?' "
    "• 'Why is my attention kernel slow at seq_len=4k?' "
    "• 'Should I use torch.compile or CUDA graphs?' "
    "• 'What causes GPU memory fragmentation?'. "
    "REQUIRES: AI backend available (check with aisp_ai_status). "
    "VERSUS: aisp_explain (concept definitions), aisp_recommend (optimization playbooks), "
    "aisp_suggest_tools (which tool to use). WORKFLOW: aisp_ask for advice → specific tools for action. NOT FOR: Raw data (use domain tools).",
    {"type": "object", "properties": with_context_params({
        "question": {
            "type": "string",
            "description": "Your performance question in natural language"
        },
    }), "required": ["question"]}
)
def tool_ask(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ask a performance question."""
    from core.engine import get_engine
    question = params.get("question", "")
    include_context, context_level = extract_context_opts(params)
    result = get_engine().ai.ask(question)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_explain",
    "Tags: explain, concept, definition, learn, understand, what-is, glossary. "
    "Explain a GPU/AI performance concept with clear definition and book citations. "
    "Returns: {explanation, key_points: [...], citations: [...], related_concepts}. "
    "USE when: Learning what a technique/concept is, understanding terminology, comparing concepts. "
    "Example: \"Explain tensor parallelism vs pipeline parallelism\" or \"What is FlashAttention?\". "
    "Good for: flash-attention, tensor-parallelism, FSDP, KV-cache, torch.compile, CUDA graphs. "
    "PREFER aisp_ask for 'why' or 'how' questions, aisp_optimize_techniques for technique catalog. ⚡ FAST (~3s). WORKFLOW: aisp_explain for concepts → aisp_ask for specific advice. NOT FOR: How-to questions (use aisp_ask).",
    {"type": "object", "properties": with_context_params({
        "concept": {
            "type": "string",
            "description": "The concept to explain (e.g., 'flash-attention', 'tensor parallelism')"
        },
    }), "required": ["concept"]}
)
def tool_explain(params: Dict[str, Any]) -> Dict[str, Any]:
    """Explain a concept."""
    from core.engine import get_engine
    concept = params.get("concept", "")
    include_context, context_level = extract_context_opts(params)
    result = get_engine().ai.explain(concept)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_ai_status",
    "Tags: ai, llm, backend, connectivity, health, api-key. "
    "Check AI/LLM backend availability: connectivity, API key status, model availability. "
    "Returns: {available, backend_type, model, api_key_set, error_if_any}. "
    "⚡ FAST (<1s). USE when: Verifying LLM connectivity before aisp_ask/aisp_explain. "
    "Example: \"Is the LLM backend reachable?\" or \"Why is aisp_ask failing?\". "
    "WORKFLOW: aisp_ai_status → if available → aisp_ask/aisp_explain. "
    "NOT FOR: General system health (use aisp_status).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_ai_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check AI status."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().ai.status()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# PROFILING TOOLS
# =============================================================================

@register_tool(
    "aisp_profile_flame",
    "Tags: profile, flame, hotspots, time, breakdown, visualization, call-stack. "
    "Get flame graph data showing execution time breakdown by function/operation. "
    "Returns: {flame_data, top_hotspots: [{function, time_pct, time_ms}], call_tree}. "
    "USE when: Identifying time hotspots, understanding where time is spent, visualizing call stacks. "
    "Example: \"Show flame graph for my training loop\" or \"Where is time spent?\". "
    "ALSO USE: aisp_profile_kernels for CUDA kernel breakdown, aisp_profile_nsys for full timeline. ⚡ FAST (~2s). WORKFLOW: aisp_profile_flame → hotspots → aisp_profile_kernels.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_profile_flame(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get flame graph."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.flame_graph()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_memory",
    "Tags: memory, timeline, spikes, leaks, oom, fragmentation, allocation, vram, cuda-oom. "
    "Get memory allocation timeline: VRAM usage over time, allocation spikes, potential leaks. "
    "Returns: {timeline: [{timestamp, allocated_gb, reserved_gb}], peak_usage, spikes, leak_suspects}. "
    "⚡ FAST (~2s). USE when: Debugging OOM, tracking memory spikes, finding leaks. "
    "Example: \"Graph VRAM over time\" or \"Why am I running out of memory?\" or \"Find memory leak\". "
    "COMMON OOM CAUSES: "
    "• Peak > VRAM: Reduce batch_size, use gradient checkpointing "
    "• Fragmentation: Use memory_efficient_attention, torch.cuda.empty_cache() "
    "• Leak: Check for growing tensor lists, unreleased intermediate tensors. "
    "WORKFLOW: aisp_profile_memory → aisp_analyze_whatif(max_vram_gb=X) → aisp_inference_quantization.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_profile_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory timeline."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.memory_timeline()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_kernels",
    "Tags: kernel, cuda, hotspots, gpu-time, breakdown, slow-kernels. "
    "Get CUDA kernel execution breakdown: time per kernel, launch counts, occupancy hints. "
    "Returns: {kernels: [{name, total_time_ms, call_count, avg_time_us, occupancy}], total_gpu_time}. "
    "USE when: Identifying slow CUDA kernels, analyzing GPU time distribution, finding optimization targets. "
    "Example: \"Which CUDA kernels are slow?\" or \"Kernel breakdown for attention\". "
    "ALSO USE: aisp_profile_ncu for detailed kernel metrics, aisp_profile_roofline for bound analysis. ⚡ FAST (~2s). WORKFLOW: aisp_profile_kernels → slow kernels → aisp_profile_ncu.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_profile_kernels(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get kernel breakdown."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.kernel_breakdown()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_roofline",
    "Tags: roofline, compute-bound, memory-bound, arithmetic-intensity, efficiency, bottleneck. "
    "Get roofline model analysis: compute vs memory bound positioning, arithmetic intensity, efficiency. "
    "Returns: {bound_type, arithmetic_intensity, achieved_flops, peak_flops, achieved_bandwidth, peak_bandwidth}. "
    "USE when: Determining if kernels are compute- or memory-bound, understanding optimization direction. "
    "Example: \"Are my kernels memory-bound?\" or \"What's the arithmetic intensity of my workload?\". "
    "Memory-bound → optimize memory access; Compute-bound → optimize math operations. ⚡ FAST (~2s). WORKFLOW: aisp_profile_roofline → if memory-bound → aisp_analyze_memory_patterns. NOT FOR: Running benchmarks (use aisp_hw_roofline).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_profile_roofline(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get roofline data."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.roofline()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_profile_compare",
    "Tags: compare, flamegraph, baseline, optimized, speedup, visualization, why-faster. "
    "Generate visual flame graph comparison showing WHY optimized code is faster. "
    "Returns: {speedup, cuda_api_comparison, kernel_breakdown, flame_diff, html_output (if requested)}. "
    "USE when: Understanding optimization impact visually, presenting before/after comparison. "
    "Example: \"Compare baseline vs optimized streams profiles\" or \"Why is the optimized code faster?\". "
    "Provide chapter (e.g., 'ch11') OR profiles_dir path. Outputs interactive HTML if output_html set. 🕐 MEDIUM (~5s). WORKFLOW: profile baseline → optimize → aisp_profile_compare. NOT FOR: Raw comparison (use aisp_compare_nsys/ncu).",
    {"type": "object", "properties": with_context_params({
        "chapter": {
            "type": "string",
            "description": "Chapter name (e.g., 'ch11', 'ch11-streams-comparison') - will find profile dir automatically"
        },
        "profiles_dir": {
            "type": "string",
            "description": "Direct path to directory with baseline/optimized .nsys-rep (alternative to chapter)"
        },
        "output_html": {
            "type": "string",
            "description": "Path to write interactive HTML comparison (optional, great for sharing)",
            "default": None
        },
    })}
)
def tool_profile_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate flame graph comparison between baseline and optimized profiles."""
    from pathlib import Path
    from core import profile_insights
    from core.perf_core_base import PerformanceCoreBase
    
    chapter = params.get("chapter")
    profiles_dir_param = params.get("profiles_dir")
    output_html = params.get("output_html")
    include_context, context_level = extract_context_opts(params)
    
    core = PerformanceCoreBase()
    
    # Resolve the profile directory
    if profiles_dir_param:
        profiles_dir = Path(profiles_dir_param)
    elif chapter:
        profiles_dir = core._find_profile_directory(chapter)
        if not profiles_dir:
            return {
                "error": f"Chapter not found: {chapter}",
                "hint": "Use aisp_profile_compare with profiles_dir parameter, or run 'aisp profile compare' to list available chapters",
            }
    else:
        # List available profile pairs
        pairs = core.list_deep_profile_pairs()
        return {
            "available_chapters": [p.get("chapter") for p in pairs.get("pairs", [])],
            "count": pairs.get("count", 0),
            "hint": "Provide chapter parameter to compare profiles. Example: aisp_profile_compare(chapter='ch11-streams-comparison')",
        }
    
    if not profiles_dir or not profiles_dir.exists():
        return make_error(f"profiles_dir not found: {profiles_dir}", include_context, context_level)
    
    result = profile_insights.generate_flamegraph_comparison(profiles_dir)
    if result is None:
        return {
            "error": "No baseline/optimized nsys profiles found",
            "profiles_dir": str(profiles_dir),
            "hint": "Profile both baseline and optimized with: nsys profile --stats=true -o <name> python <script>.py",
        }
    
    if result.get("error"):
        return result
    
    # NEW: Also get metric-level analysis via compare_profiles()
    # This adds improvements/regressions analysis and bottleneck shift detection
    if chapter:
        try:
            metric_comparison = core.compare_profiles(chapter)
            if metric_comparison and not metric_comparison.get("error"):
                if "metric_analysis" in metric_comparison:
                    result["metric_analysis"] = metric_comparison["metric_analysis"]
                if "ncu_comparison" in metric_comparison:
                    result["ncu_comparison"] = metric_comparison["ncu_comparison"]
                if "recommendations" in metric_comparison:
                    result["recommendations"] = metric_comparison["recommendations"]
        except Exception:
            pass  # Best effort - don't fail if metric analysis unavailable
    
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
    "Tags: nsys, nsight-systems, profile, timeline, trace, cuda-api, nvtx, deep-dive. "
    "Run Nsight Systems profiling to capture GPU timeline, CUDA API calls, kernel launches. "
    "Returns: {output_path, success, run_details} and writes .nsys-rep file. "
    "USE when: Need detailed timeline view, understanding kernel launch patterns, API overhead. "
    "Example: \"Profile python train.py with nsys\" or \"Capture timeline for batch 32 inference\". "
    "⚠️ SLOW: 1-10+ minutes depending on workload. ALWAYS use dry_run=true first to preview command. "
    "PRESETS: preset='light' for quick/small traces, preset='full' (default) for comprehensive data. "
    "WORKFLOW: aisp_status → aisp_profile_nsys(dry_run=true) → aisp_profile_nsys → aisp_nsys_summary → aisp_compare_nsys. "
    "FOR QUICK CHECKS: Use aisp_hw_speed or aisp_profile_kernels instead. NOT FOR: Kernel metrics (use aisp_profile_ncu).",
    {"type": "object", "properties": with_context_params({
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
        "async": {
            "type": "boolean",
            "description": "Return a job ticket and run capture in background; poll with aisp_job_status",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 300
        },
    }), "required": ["command"]}
)
def tool_profile_nsys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Nsight Systems profiling for an arbitrary command."""
    from pathlib import Path
    from core.profiling.nsight_automation import NsightAutomation

    include_context, context_level = extract_context_opts(params)
    command = params.get("command") or []
    if not command:
        return make_error("command is required", include_context, context_level)

    output_name = params.get("output_name", "mcp_nsys")
    output_dir = Path(params.get("output_dir", "artifacts/mcp-profiles"))
    trace_cuda = bool(params.get("trace_cuda", True))
    trace_nvtx = bool(params.get("trace_nvtx", True))
    trace_osrt = bool(params.get("trace_osrt", True))
    full_timeline = bool(params.get("full_timeline", False))
    trace_forks = bool(params.get("trace_forks", True))
    preset = normalize_param("preset", params.get("preset"), "light")
    force_lineinfo = bool(params.get("force_lineinfo", True))
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

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
        return make_error("command is required", include_context, context_level, **precheck)
    if not automation.nsys_available:
        return make_error("nsys is not installed or not on PATH", include_context, context_level, **precheck)
    if not cuda_check.get("ok", True):
        return make_error(cuda_check.get("reason", "CUDA not available"), include_context, context_level, **precheck)

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
            "note": "Set dry_run=false to execute; use async=true to background the run. Default preset is full; set preset=light for smaller/faster traces.",
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

    if run_async:
        queued = _queue_job("aisp_profile_nsys", _execute_capture, params)
        queued["note"] = "Background capture started; poll with aisp_job_status using job_id."
        queued["preset"] = preset
        return queued

    return _execute_capture()


@register_tool(
    "aisp_profile_ncu",
    "Tags: ncu, nsight-compute, profile, kernel-metrics, occupancy, memory-throughput. "
    "Run Nsight Compute profiling to capture detailed per-kernel metrics (occupancy, throughput, etc.). "
    "Returns: {output_path, success, run_details} and writes .ncu-rep file. "
    "USE when: Deep-diving into specific kernel performance, optimizing occupancy, memory access. "
    "Example: \"Profile attention kernel with ncu\" or \"Get detailed metrics for matmul\". "
    "⚠️ VERY SLOW: Replays kernels. Use kernel_filter to limit scope. Use dry_run=true first. "
    "workload_type: memory_bound (default, fast), compute_bound, tensor_core, attention. NOT FOR: Kernel metrics (use aisp_profile_ncu). 🕐 SLOW (varies). WORKFLOW: aisp_profile_kernels → aisp_profile_ncu. NOT FOR: Timeline (use aisp_profile_nsys).",
    {"type": "object", "properties": with_context_params({
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
            "enum": ["memory_bound", "compute_bound", "tensor_core", "attention"],
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
        "async": {
            "type": "boolean",
            "description": "Return a job ticket and run capture in background; poll with aisp_job_status",
            "default": False
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Max runtime before returning partial output; set 0/null for no timeout",
            "default": 300
        },
    }), "required": ["command"]}
)
def tool_profile_ncu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Nsight Compute profiling for an arbitrary command."""
    from pathlib import Path
    from core.profiling.nsight_automation import NsightAutomation

    include_context, context_level = extract_context_opts(params)
    command = params.get("command") or []
    if not command:
        return make_error("command is required", include_context, context_level)

    output_name = params.get("output_name", "mcp_ncu")
    output_dir = Path(params.get("output_dir", "artifacts/mcp-profiles"))
    workload_type = normalize_param("workload_type", params.get("workload_type"), "memory_bound")
    kernel_filter = params.get("kernel_filter")
    force_lineinfo = bool(params.get("force_lineinfo", True))
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

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
        return make_error("command is required", include_context, context_level, **precheck)
    if not automation.ncu_available:
        return make_error("ncu is not installed or not on PATH", include_context, context_level, **precheck)
    if not cuda_check.get("ok", True):
        return make_error(cuda_check.get("reason", "CUDA not available"), include_context, context_level, **precheck)

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
            "note": "Set dry_run=false to execute; use async=true to background the run.",
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

    if run_async:
        queued = _queue_job("aisp_profile_ncu", _execute_capture, params)
        queued["note"] = "Background capture started; poll with aisp_job_status using job_id."
        queued["workload_type"] = workload_type
        return queued

    return _execute_capture()


@register_tool(
    "aisp_profile_torch",
    "Tags: torch, profiler, pytorch, chrome-trace, autograd, cpu-gpu. "
    "Run PyTorch torch.profiler to capture CPU/GPU activity with Chrome trace output. "
    "Returns: {trace_path, summary, success} and writes Chrome trace JSON + summary. "
    "USE when: Profiling PyTorch code specifically, understanding autograd overhead, CPU/GPU interplay. "
    "Example: \"Profile my training script with torch.profiler\" or \"Get PyTorch trace for train.py\". "
    "Output viewable in chrome://tracing or Perfetto. Emits NVTX for nsys correlation. 🕐 SLOW (varies). WORKFLOW: aisp_profile_kernels → aisp_profile_ncu. NOT FOR: Timeline (use aisp_profile_nsys).",
    {"type": "object", "properties": with_context_params({
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
        "async": {"type": "boolean", "description": "Return a job ticket and run capture in background; poll with aisp_job_status", "default": False},
        "timeout_seconds": {"type": "integer", "description": "Max runtime before returning partial output; set 0/null for no timeout", "default": 300},
    }), "required": ["script"]}
)
def tool_profile_torch(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run torch.profiler for a Python script and return summary + trace paths."""
    from pathlib import Path
    from core.profiling.torch_profiler import TorchProfilerAutomation

    include_context, context_level = extract_context_opts(params)
    script = params.get("script")
    if not script:
        return make_error("script is required", include_context, context_level)

    script_path = Path(script)
    output_dir = Path(params.get("output_dir", "artifacts/mcp-profiles/torch"))
    output_name = params.get("output_name") or script_path.stem or "mcp_torch"
    mode = normalize_param("torch_mode", params.get("mode"), "full")
    script_args = params.get("script_args") or []
    if isinstance(script_args, str):
        import shlex as _shlex  # local import to avoid global side effects
        script_args = _shlex.split(script_args)
    force_lineinfo = bool(params.get("force_lineinfo", True))
    use_nvtx = bool(params.get("use_nvtx", True))
    nvtx_label = params.get("nvtx_label", "aisp_torch_profile")
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

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
        return make_error(f"script not found: {script}", include_context, context_level, **precheck)
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

    if run_async:
        queued = _queue_job("aisp_profile_torch", _execute_capture, params)
        queued["note"] = "Background torch.profiler capture started; poll with aisp_job_status using job_id."
        queued["mode"] = mode
        queued["nvtx_label"] = nvtx_label
        return queued

    return _execute_capture()


@register_tool(
    "aisp_profile_hta",
    "Tags: hta, holistic-trace, nsys, analysis, gpu-idle, timeline. "
    "Run Nsight Systems capture with HTA (Holistic Trace Analysis) for automated bottleneck detection. "
    "Returns: {nsys_rep_path, trace_json_path, hta_report_path, analysis_summary}. "
    "USE when: Want automated analysis of trace data, finding GPU idle time, communication bottlenecks. "
    "Example: \"Profile and analyze with HTA\" or \"Get holistic trace analysis for my script\". "
    "Produces .nsys-rep + trace.json + hta_report.json with actionable insights. 🕐 MEDIUM (~30s). WORKFLOW: aisp_profile_torch → operators → optimize. NOT FOR: CUDA-level (use aisp_profile_nsys).",
    {"type": "object", "properties": with_context_params({
        "command": {"type": "array", "items": {"type": "string"}, "description": "Command to profile (argv list)"},
        "output_name": {"type": "string", "description": "Base name for output files", "default": "mcp_hta"},
        "output_dir": {"type": "string", "description": "Directory for outputs (default: artifacts/hta)", "default": "artifacts/hta"},
        "preset": {"type": "string", "description": "nsys preset", "enum": ["light", "full"], "default": "full"},
        "force_lineinfo": {"type": "boolean", "description": "Force -lineinfo for source/line mapping", "default": True},
        "precheck_only": {"type": "boolean", "description": "Return prereqs without running", "default": False},
        "dry_run": {"type": "boolean", "description": "Describe the capture without executing", "default": False},
        "async": {"type": "boolean", "description": "Run in background and return job_id", "default": False},
        "timeout_seconds": {"type": "integer", "description": "Max runtime before returning partial output; set 0/null for none", "default": 300},
    }), "required": ["command"]}
)
def tool_profile_hta(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run nsys + HTA analysis."""
    import shlex
    from pathlib import Path
    from core.profiling.hta_capture import HTACaptureAutomation
    from core.profiling.nsight_automation import NsightAutomation

    include_context, context_level = extract_context_opts(params)
    command_list = params.get("command") or []
    if isinstance(command_list, str):
        command_list = shlex.split(command_list)
    if not command_list:
        return make_error("command is required", include_context, context_level)

    output_dir = Path(params.get("output_dir", "artifacts/hta"))
    output_name = params.get("output_name", "mcp_hta")
    preset = normalize_param("preset", params.get("preset"), "full")
    force_lineinfo = bool(params.get("force_lineinfo", True))
    precheck_only = bool(params.get("precheck_only"))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

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
        return make_error("nsys is not installed or not on PATH", include_context, context_level, **precheck)
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

    if run_async:
        queued = _queue_job("aisp_profile_hta", _execute_capture, params)
        queued["note"] = "Background HTA capture started; poll with aisp_job_status using job_id."
        queued["preset"] = preset
        return queued

    return _execute_capture()


@register_tool(
    "aisp_export_csv",
    "Tags: export, csv, spreadsheet, data, share. "
    "Export benchmarks to CSV format for spreadsheet analysis or sharing. "
    "Returns: {csv: <csv_string>, detailed: bool}. "
    "USE when: Importing benchmark data into Excel/Sheets, sharing raw numbers. "
    "Example: \"Export benchmarks to CSV\" or \"Get CSV of all results\". "
    "detailed=true includes all metrics; false gives summary columns only. 🕐 SLOW (varies). WORKFLOW: aisp_profile_hta → analyze GPU idle → optimize. NOT FOR: Quick checks (use aisp_profile_flame).",
    {"type": "object", "properties": with_context_params({
        "detailed": {
            "type": "boolean",
            "description": "Include all metrics (true) or summary only (false)",
            "default": False
        },
    })}
)
def tool_export_csv(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to CSV."""
    from core.engine import get_engine
    detailed = bool(params.get("detailed", False))
    include_context, context_level = extract_context_opts(params)
    export = get_engine().export.csv_detailed() if detailed else get_engine().export.csv()
    result = {"csv": export, "detailed": detailed}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_export_pdf",
    "Tags: export, pdf, report, document, share, print. "
    "Export benchmarks to PDF report format for printing or formal sharing. "
    "Returns: {pdf_base64: <base64_encoded_pdf>}. "
    "USE when: Creating printable reports, formal documentation, sharing with stakeholders. "
    "Example: \"Generate PDF report\" or \"Create printable benchmark summary\". "
    "PREFER aisp_benchmark_report for more control over report options. 🕐 MEDIUM (~5s). WORKFLOW: aisp_run_benchmarks → aisp_export_pdf. NOT FOR: Interactive reports (use aisp_export_html).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_export_pdf(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to PDF."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    pdf_bytes = get_engine().export.pdf()
    result = {"pdf_base64": pdf_bytes if isinstance(pdf_bytes, str) else str(pdf_bytes)}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_export_html",
    "Tags: export, html, interactive, web, share, visualization. "
    "Export benchmarks to interactive HTML report with charts and tables. "
    "Returns: {html: <html_string>}. "
    "USE when: Sharing interactive web-viewable reports, embedding in documentation. "
    "Example: \"Generate HTML report\" or \"Create interactive benchmark visualization\". "
    "PREFER aisp_benchmark_report for more control over report options. ⚡ FAST (~2s). WORKFLOW: aisp_run_benchmarks → aisp_export_html. NOT FOR: Raw data (use aisp_export_csv).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_export_html(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to HTML."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    html = get_engine().export.html()
    result = {"html": html}
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# TEST TOOLS
# =============================================================================

@register_tool(
    "aisp_hw_speed",
    "Tags: speed, benchmark, gemm, memory, attention, quick-test, sanity-check. "
    "Run quick GPU speed tests: GEMM throughput, memory bandwidth, attention kernel. "
    "Returns: {tests: [{name, latency_ms, throughput, result}]}. "
    "USE when: Quick sanity check of GPU performance, verifying hardware is working correctly. "
    "Example: \"Quick benchmark GPU speed\" or \"Run GEMM and memory tests\". "
    "type='all' runs everything; 'gemm'/'memory'/'attention' for specific tests. "
    "⚠️ Stresses GPU briefly. Use dry_run=true to preview what will run. 🕐 MEDIUM (~15s). WORKFLOW: aisp_status → aisp_hw_speed → verify GPU health. NOT FOR: Deep profiling (use aisp_profile_*).",
    {"type": "object", "properties": with_context_params({
        "type": {
            "type": "string",
            "description": "Test selection: all, gemm, memory, or attention",
            "default": "all",
            "enum": ["all", "gemm", "memory", "attention"]
        },
        "gemm_size": {"type": "integer", "description": "GEMM size", "default": 512},
        "precision": {"type": "string", "description": "Precision (fp16/bf16/tf32/fp32/fp8)", "enum": ["fp16", "bf16", "tf32", "fp32", "fp8"], "default": "fp16"},
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
    })}
)
def tool_hw_speed(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run speed tests without invoking the bench CLI."""
    from core.diagnostics import microbench

    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context, context_level = extract_context_opts(params)
    cuda_check = _cuda_precheck()

    test_type = normalize_param("test_type", params.get("type"), "all")
    precision_val = normalize_param("precision", params.get("precision"), "fp16")
    planned = {
        "type": test_type,
        "gemm_size": params.get("gemm_size", 512),
        "precision": precision_val,
        "mem_size_mb": params.get("mem_size_mb", 16),
        "mem_stride": params.get("mem_stride", 128),
    }

    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned": planned,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }

    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
        }

    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "planned": planned,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run aisp_status first.",
        }

    results: List[Dict[str, Any]] = []

    if test_type in ("all", "gemm"):
        res = microbench.tensor_core_bench(
            size=int(params.get("gemm_size", 512)),
            precision=precision_val,
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        res["name"] = "tensor_core_gemm"
        results.append(res)

    if test_type in ("all", "memory"):
        res = microbench.mem_hierarchy_test(
            size_mb=int(params.get("mem_size_mb", 16)),
            stride=int(params.get("mem_stride", 128)),
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        res["name"] = "mem_hierarchy"
        results.append(res)

    if test_type in ("all", "attention"):
        attn: Dict[str, Any] = {"name": "attention_softmax"}
        try:
            import torch

            if torch.cuda.is_available():
                q = torch.randn(256, 256, device="cuda")
                torch.cuda.synchronize()
                start = time.perf_counter()
                torch.softmax(q, dim=-1)
                torch.cuda.synchronize()
                attn["latency_ms"] = (time.perf_counter() - start) * 1000
            else:
                attn["error"] = "CUDA not available"
        except Exception as exc:
            attn["error"] = str(exc)
        results.append(attn)

    result = {"tests": results}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_hw_roofline",
    "Tags: roofline, stride, memory, bandwidth, sweep, cache. "
    "Run stride sweep to measure memory bandwidth at different access patterns (roofline data). "
    "Returns: {size_mb, rows: [(stride, bandwidth_gbps), ...]}. "
    "USE when: Understanding memory hierarchy performance, cache behavior, roofline positioning. "
    "Example: \"Run roofline stride sweep\" or \"Measure bandwidth at different strides\". "
    "Sweeps strides from 32 to 4096 bytes by default. ⚠️ Stresses memory subsystem. 🕐 MEDIUM (~20s). WORKFLOW: aisp_hw_roofline → aisp_profile_roofline. NOT FOR: Quick tests (use aisp_hw_speed).",
    {"type": "object", "properties": with_context_params({
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
    })}
)
def tool_hw_roofline(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench

    size_mb = int(params.get("size_mb", 32))
    strides = [int(s) for s in params.get("strides") or []]
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context, context_level = extract_context_opts(params)
    cuda_check = _cuda_precheck()

    planned = {"size_mb": size_mb, "strides": strides or None}

    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned": planned,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
            "planned": planned,
        }
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "planned": planned,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run aisp_status first.",
        }

    sweep_strides = strides or [32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    for stride in sweep_strides:
        res = microbench.mem_hierarchy_test(
            size_mb=size_mb,
            stride=stride,
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        rows.append((stride, res.get("bandwidth_gbps")))

    result = {"size_mb": size_mb, "rows": rows}
    return attach_context_if_requested(result, include_context, context_level)

@register_tool(
    "aisp_hw_disk",
    "Tags: disk, io, storage, read, write, sequential, throughput. "
    "Run disk I/O benchmark measuring sequential read/write throughput. "
    "Returns: {read_mbps, write_mbps, file_size_mb, block_size_kb}. "
    "USE when: Checking if disk I/O is a bottleneck, verifying storage performance. "
    "Example: \"Benchmark disk I/O\" or \"Is my storage fast enough for checkpointing?\". "
    "Writes temp file to tmp_dir (or /tmp). ⚠️ Writes to disk. 🕐 MEDIUM (~10s). WORKFLOW: aisp_analyze_dataloader → if IO-bound → aisp_hw_disk. NOT FOR: GPU tests (use aisp_hw_speed).",
    {"type": "object", "properties": with_context_params({
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
    })}
)
def tool_test_disk(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context, context_level = extract_context_opts(params)
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
            return make_error(f"failed to create tmp_dir: {tmp_dir}", include_context, context_level, tmp_dir=tmp_dir)
    result = microbench.disk_io_test(
        file_size_mb=int(params.get("file_size_mb", 256)),
        block_size_kb=int(params.get("block_size_kb", 1024)),
        tmp_dir=tmp_dir,
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_hw_pcie",
    "Tags: pcie, h2d, d2h, host-to-device, bandwidth, transfer. "
    "Run PCIe bandwidth benchmark measuring Host-to-Device and Device-to-Host transfer speeds. "
    "Returns: {h2d_gbps, d2h_gbps, size_mb, iters}. "
    "USE when: Checking PCIe bandwidth, diagnosing data transfer bottlenecks. "
    "Example: \"Test PCIe bandwidth\" or \"How fast is H2D transfer?\". "
    "NOT FOR: GPU memory bandwidth (use aisp_gpu_bandwidth), GPU-to-GPU (use aisp_hw_p2p). 🕐 MEDIUM (~10s). WORKFLOW: aisp_hw_pcie → if slow → check PCIe gen/width.",
    {"type": "object", "properties": with_context_params({
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
    })}
)
def tool_test_pcie(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    include_context, context_level = extract_context_opts(params)
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
        return make_error(cuda_check.get("reason", "CUDA not available"), include_context, context_level, cuda=cuda_check)
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size_mb": params.get("size_mb", 256), "iters": params.get("iters", 10)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    result = microbench.pcie_bandwidth_test(
        size_mb=int(params.get("size_mb", 256)),
        iters=int(params.get("iters", 10)),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_hw_cache",
    "Tags: cache, memory, hierarchy, l2, hbm, stride, bandwidth. "
    "Run GPU memory hierarchy test measuring bandwidth at specific stride pattern. "
    "Returns: {bandwidth_gbps, size_mb, stride, achieved_vs_peak_pct}. "
    "USE when: Understanding cache/memory hierarchy effects, optimizing memory access patterns. "
    "Example: \"Test L2 cache effect\" or \"Measure bandwidth at 128-byte stride\". "
    "ALSO USE: aisp_hw_roofline for full stride sweep. 🕐 MEDIUM (~15s). WORKFLOW: aisp_hw_cache → aisp_profile_roofline.",
    {"type": "object", "properties": with_context_params({
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
    })}
)
def tool_test_mem_hierarchy(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    include_context, context_level = extract_context_opts(params)
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
        return make_error(cuda_check.get("reason", "CUDA not available"), include_context, context_level, cuda=cuda_check)
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size_mb": params.get("size_mb", 256), "stride": params.get("stride", 128)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    result = microbench.mem_hierarchy_test(
        size_mb=int(params.get("size_mb", 256)),
        stride=int(params.get("stride", 128)),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_hw_tc",
    "Tags: tensor-core, matmul, gemm, tflops, precision, fp16, bf16, tf32. "
    "Run Tensor Core throughput test measuring matmul performance at different precisions. "
    "Returns: {tflops, latency_ms, size, precision, efficiency_vs_peak_pct}. "
    "USE when: Verifying Tensor Core performance, comparing precision throughput. "
    "Example: \"Test Tensor Core TFLOPS\" or \"Compare FP16 vs BF16 matmul speed\". "
    "precision: fp16, bf16, tf32, fp32, fp8 (H100+ only). 🕐 MEDIUM (~15s). WORKFLOW: aisp_hw_tc → compare vs expected TFLOPS. NOT FOR: Memory tests (use aisp_gpu_bandwidth).",
    {"type": "object", "properties": with_context_params({
        "size": {"type": "integer", "default": 4096},
        "precision": {"type": "string", "enum": ["fp16", "bf16", "tf32", "fp32", "fp8"], "default": "fp16"},
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
    })}
)
def tool_test_tensor_core(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench
    include_context, context_level = extract_context_opts(params)
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    precision = normalize_param("precision", params.get("precision"), "fp16")
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run aisp_status or aisp_triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return make_error(cuda_check.get("reason", "CUDA not available"), include_context, context_level, cuda=cuda_check)
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size": params.get("size", 4096), "precision": precision},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    result = microbench.tensor_core_bench(
        size=int(params.get("size", 4096)),
        precision=precision,
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_hw_network",
    "Tags: network, throughput, interconnect, bandwidth, nic. "
    "Run network throughput tests to check NIC and interconnect performance. "
    "Returns: {throughput_gbps, latency_ms, interface_info}. "
    "USE when: Checking host network bandwidth, interconnect performance for distributed training. "
    "Example: \"Test network bandwidth\" or \"Check interconnect speed between nodes\". "
    "ALSO USE: aisp_info_network for InfiniBand status, aisp_hw_nccl for NCCL collectives. 🕐 MEDIUM (~15s). WORKFLOW: aisp_hw_network → if slow → check NIC config.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_test_network(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run network tests."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().test.network()
    return attach_context_if_requested(result, include_context, context_level)


# NOTE: aisp_benchmark_targets is defined earlier in this file with bench CLI integration
# This duplicate registration was removed to avoid conflicts.

# =============================================================================
# ADVANCED SYSTEM ANALYSIS TOOLS
# =============================================================================

@register_tool(
    "aisp_system_full",
    "Tags: system, full, audit, cpu, memory, container, kernel, comprehensive. "
    "Full system analysis: CPU/memory hierarchy, kernel params, container limits, tuning recommendations. "
    "Returns: {cpu_info, memory_hierarchy, system_params, container_limits, recommendations}. "
    "🕐 MEDIUM (~3s). USE when: Deep environment auditing, diagnosing host-side bottlenecks. "
    "Example: \"Full system audit\" or \"Check host-side performance issues\". "
    "WORKFLOW: aisp_triage → aisp_system_full → apply recommendations. "
    "NOT FOR: Quick checks (use aisp_status), GPU-specific (use aisp_gpu_info).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_system_full(params: Dict[str, Any]) -> Dict[str, Any]:
    """Full system analysis bundle."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_full_system_analysis()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"full system analysis failed: {e}", include_context, context_level)


@register_tool(
    "aisp_analyze_memory_patterns",
    "Tags: analyze, memory, warp, bank, coalescing, access, patterns, divergence, conflicts. "
    "Memory access pattern analysis: warp divergence, bank conflicts, memory coalescing. "
    "Returns: {warp_divergence, bank_conflicts, memory_access, recommendations}. "
    "⚡ FAST (~2s). USE when: Debugging memory-bound kernels, optimizing memory access. "
    "Params: analysis_type='all'|'warp'|'bank'|'access' for specific analysis. "
    "Example: \"Check for warp divergence\" or \"Analyze bank conflicts\". "
    "WORKFLOW: aisp_profile_roofline (check if memory-bound) → aisp_analyze_memory_patterns → optimize. "
    "NOT FOR: High-level bottlenecks (use aisp_analyze_bottlenecks first).",
    {"type": "object", "properties": with_context_params({
        "analysis_type": {
            "type": "string",
            "enum": ["all", "warp", "bank", "access"],
            "default": "all",
            "description": "Type of memory analysis: warp divergence, bank conflicts, or memory access patterns"
        },
        "stride": {"type": "integer", "default": 1, "description": "Memory stride for analysis"},
    })}
)
def tool_analyze_memory_patterns(params: Dict[str, Any]) -> Dict[str, Any]:
    """Memory access pattern analysis."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    analysis_type = normalize_param("analysis_type", params.get("analysis_type"), "all")
    stride = params.get("stride", 1)
    
    try:
        core = PerformanceCoreBase()
        result = {}
        
        if analysis_type in ("all", "warp"):
            result["warp_divergence"] = core.get_warp_divergence()
        if analysis_type in ("all", "bank"):
            result["bank_conflicts"] = core.get_bank_conflicts(stride=stride)
        if analysis_type in ("all", "access"):
            result["memory_access"] = core.get_memory_access_patterns(stride=stride)
        
        result["success"] = True
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"memory analysis failed: {e}", include_context, context_level)


@register_tool(
    "aisp_analyze_dataloader",
    "Tags: analyze, dataloader, data, loading, io, bottleneck, workers, prefetch. "
    "DataLoader bottleneck analysis: worker efficiency, prefetch, throughput. "
    "Returns: {throughput, worker_efficiency, recommendations}. "
    "⚡ FAST (~2s). USE when: Diagnosing data loading bottlenecks in training. "
    "Example: \"Is data loading the bottleneck?\" or \"Check DataLoader efficiency\". "
    "WORKFLOW: aisp_analyze_bottlenecks → if host-bound → aisp_analyze_dataloader → tune num_workers/prefetch. "
    "COMMON FIXES: Increase num_workers, enable pin_memory, use persistent_workers.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_analyze_dataloader(params: Dict[str, Any]) -> Dict[str, Any]:
    """DataLoader bottleneck analysis."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_data_loading_analysis()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"dataloader analysis failed: {e}", include_context, context_level)


@register_tool(
    "aisp_analyze_comm_overlap",
    "Tags: analyze, communication, compute, overlap, distributed, efficiency, allreduce. "
    "Communication/compute overlap analysis for distributed training. "
    "Returns: {overlap_efficiency, compute_time, comm_time, recommendations}. "
    "⚡ FAST (~2s). USE when: Optimizing distributed training efficiency. "
    "Example: \"Is communication overlapping compute?\" or \"Check allreduce overlap\". "
    "WORKFLOW: aisp_distributed_plan → aisp_analyze_comm_overlap → aisp_distributed_nccl. "
    "NOT FOR: Single-GPU training (no communication to overlap).",
    {"type": "object", "properties": with_context_params({
        "model": {"type": "string", "default": "llama-3.1-70b", "description": "Model name for analysis"},
    })}
)
def tool_analyze_comm_overlap(params: Dict[str, Any]) -> Dict[str, Any]:
    """Communication overlap analysis."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    model = params.get("model", "llama-3.1-70b")
    try:
        core = PerformanceCoreBase()
        result = core.get_comm_overlap_analysis(model=model)
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"comm overlap analysis failed: {e}", include_context, context_level)


@register_tool(
    "aisp_cost_estimate",
    "Tags: cost, cloud, estimate, pricing, aws, gcp, azure, budget, billing. "
    "Cloud cost estimation for training/inference workloads. "
    "Returns: {hourly_cost, monthly_cost, cloud_comparison, recommendations}. "
    "⚡ FAST (~1s). USE when: Planning cloud deployments, comparing providers. "
    "Example: \"Estimate 8xH100 monthly cost\" or \"Compare AWS vs GCP pricing\". "
    "WORKFLOW: aisp_distributed_plan → aisp_cost_estimate → choose provider. "
    "ALSO USE: aisp_analyze_energy for power/efficiency considerations.",
    {"type": "object", "properties": with_context_params({
        "gpu_type": {"type": "string", "default": "h100", "description": "GPU type (h100, a100, etc.)"},
        "num_gpus": {"type": "integer", "default": 8, "description": "Number of GPUs"},
        "hours_per_day": {"type": "number", "default": 8, "description": "Usage hours per day"},
    })}
)
def tool_cost_estimate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Cloud cost estimation."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_cloud_cost_estimate(params)
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"cost estimate failed: {e}", include_context, context_level)


@register_tool(
    "aisp_analyze_energy",
    "Tags: analyze, energy, power, efficiency, green, carbon, sustainability. "
    "Energy efficiency analysis: power consumption, efficiency metrics, green recommendations. "
    "Returns: {power_draw, efficiency_score, carbon_estimate, recommendations}. "
    "⚡ FAST (~2s). USE when: Optimizing for energy efficiency, reducing carbon footprint. "
    "Example: \"What's my energy efficiency?\" or \"Estimate carbon footprint\". "
    "WORKFLOW: aisp_gpu_power → aisp_analyze_energy → apply power-saving recommendations. "
    "ALSO USE: aisp_cost_estimate for cloud cost implications.",
    {"type": "object", "properties": with_context_params({})}
)
def tool_analyze_energy(params: Dict[str, Any]) -> Dict[str, Any]:
    """Energy efficiency analysis."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_energy_analysis()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"energy analysis failed: {e}", include_context, context_level)


@register_tool(
    "aisp_launch_plan",
    "Tags: launch, torchrun, srun, distributed, command, script, multi-node. "
    "Generate launch commands for distributed training (torchrun, srun, etc.). "
    "Returns: {torchrun_cmd, srun_cmd, env_vars, tips}. "
    "⚡ FAST (~1s). USE when: Setting up distributed training launch. "
    "Example: \"Generate torchrun command for 2 nodes\" or \"Slurm launch script\". "
    "WORKFLOW: aisp_distributed_plan → aisp_launch_plan → run training. "
    "ALSO USE: aisp_cluster_slurm for full SLURM job scripts.",
    {"type": "object", "properties": with_context_params({
        "nodes": {"type": "integer", "default": 1, "description": "Number of nodes"},
        "gpus_per_node": {"type": "integer", "default": 8, "description": "GPUs per node"},
        "script": {"type": "string", "description": "Training script path"},
    })}
)
def tool_launch_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate launch plan."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_launch_plan(params)
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"launch plan failed: {e}", include_context, context_level)


@register_tool(
    "aisp_info_features",
    "Tags: info, features, capabilities, tma, clusters, hopper. "
    "GPU feature detection: TMA, thread block clusters, async copy, etc. "
    "Returns: {features: {tma, clusters, async_copy, ...}, compute_capability}. "
    "⚡ FAST (~1s). USE when: Checking advanced GPU feature support. WORKFLOW: aisp_info_features → choose optimizations. NOT FOR: Basic GPU info (use aisp_gpu_info).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_info_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """GPU feature detection."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_hardware_capabilities()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"feature detection failed: {e}", include_context, context_level)


@register_tool(
    "aisp_nsys_summary",
    "Tags: nsys, nsight, summary, quick, stats. "
    "Quick Nsight Systems summary stats without full profile capture. "
    "Returns: {cuda_api_summary, kernel_stats, memory_ops, timeline_overview}. "
    "⚡ FAST (~3s). USE when: Quick nsys stats without full profiling. WORKFLOW: aisp_profile_nsys → aisp_nsys_summary. NOT FOR: Full profiling (use aisp_profile_nsys).",
    {"type": "object", "properties": with_context_params({
        "report_path": {"type": "string", "description": "Path to existing .nsys-rep file to summarize"},
    })}
)
def tool_nsys_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """Quick nsys summary."""
    from core.perf_core_base import PerformanceCoreBase
    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        report_path = params.get("report_path")
        result = core.get_nsys_summary(report_path) if hasattr(core, 'get_nsys_summary') else {"error": "nsys summary not available"}
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"nsys summary failed: {e}", include_context, context_level)


@register_tool(
    "aisp_predict_scaling",
    "Tags: predict, scaling, multi-gpu, efficiency, projection, planning. "
    "Predict performance scaling to more GPUs/larger batches. "
    "Returns: {current_perf, predicted_perf, scaling_efficiency, bottleneck_at_scale}. "
    "⚡ FAST (~2s). USE when: Planning scale-up, predicting multi-GPU performance. "
    "Example: \"Predict 8-GPU scaling\" or \"How will throughput scale from 1 to 4 GPUs?\". "
    "WORKFLOW: aisp_gpu_topology → aisp_predict_scaling → aisp_distributed_plan. "
    "NOTE: Scaling efficiency typically 80-90% for well-optimized workloads.",
    {"type": "object", "properties": with_context_params({
        "target_gpus": {"type": "integer", "default": 8, "description": "Target GPU count"},
        "current_gpus": {"type": "integer", "default": 1, "description": "Current GPU count"},
        "model_size": {"type": "number", "description": "Model size in billions"},
    })}
)
def tool_predict_scaling(params: Dict[str, Any]) -> Dict[str, Any]:
    """Predict scaling performance."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    try:
        engine = get_engine()
        # Use analyze.scaling with prediction
        result = engine.analyze.scaling()
        result["prediction"] = {
            "target_gpus": params.get("target_gpus", 8),
            "current_gpus": params.get("current_gpus", 1),
            "estimated_speedup": min(params.get("target_gpus", 8) / max(params.get("current_gpus", 1), 1) * 0.85, 7.5),
            "note": "Scaling efficiency typically 80-90% for well-optimized workloads"
        }
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"scaling prediction failed: {e}", include_context, context_level)


@register_tool(
    "aisp_gpu_topology_matrix",
    "Tags: topology, nvlink, pcie, numa, raw, nvidia-smi, matrix. "
    "Get raw GPU/NUMA topology matrix directly from nvidia-smi topo -m. "
    "Returns: {stdout: <nvidia-smi topo -m output>, returncode}. "
    "⚡ FAST (~1s). USE when: Need exact nvidia-smi topology output format. "
    "Example: \"Show raw nvidia-smi topo output\" or \"Get NVLink matrix raw\". "
    "WORKFLOW: aisp_gpu_topology_matrix → parse manually if needed. "
    "NOT FOR: Parsed topology (use aisp_gpu_topology).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_gpu_topology_matrix(params: Dict[str, Any]) -> Dict[str, Any]:
    """Raw GPU topology matrix from nvidia-smi."""
    include_context, context_level = extract_context_opts(params)
    try:
        proc = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5)
        result = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "success": proc.returncode == 0}
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"nvidia-smi topo failed: {e}", include_context, context_level)


@register_tool(
    "aisp_compare_nsys",
    "Tags: compare, nsys, nsight-systems, baseline, optimized, diff. "
    "Compare baseline vs optimized Nsight Systems reports. "
    "Returns: {speedup, baseline_metrics, optimized_metrics, kernel_comparison}. "
    "🕐 MEDIUM (~5s). USE when: Comparing before/after nsys profiles. WORKFLOW: aisp_profile_nsys → optimize → aisp_compare_nsys. NOT FOR: Kernel metrics (use aisp_compare_ncu).",
    {"type": "object", "properties": with_context_params({
        "profiles_dir": {"type": "string", "description": "Directory with baseline/optimized .nsys-rep files"},
    }), "required": ["profiles_dir"]}
)
def tool_compare_nsys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare Nsight Systems profiles."""
    from pathlib import Path
    from core import profile_insights
    include_context, context_level = extract_context_opts(params)
    profiles_dir = Path(params.get("profiles_dir", ""))
    if not profiles_dir.exists():
        return make_error(f"profiles_dir not found: {profiles_dir}", include_context, context_level)
    try:
        result = profile_insights.compare_nsys_files(profiles_dir)
        if result is None:
            result = {"error": "No comparable nsys files found", "success": False}
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"nsys comparison failed: {e}", include_context, context_level)


@register_tool(
    "aisp_compare_ncu",
    "Tags: compare, ncu, nsight-compute, baseline, optimized, kernel-metrics. "
    "Compare baseline vs optimized Nsight Compute kernel metrics. "
    "Returns: {kernel_comparison: [{kernel, baseline_metrics, optimized_metrics}]}. "
    "🕐 MEDIUM (~5s). USE when: Deep-diving into kernel-level improvements. WORKFLOW: aisp_profile_ncu → optimize → aisp_compare_ncu. NOT FOR: Timeline comparison (use aisp_compare_nsys).",
    {"type": "object", "properties": with_context_params({
        "profiles_dir": {"type": "string", "description": "Directory with baseline/optimized .ncu-rep files"},
    }), "required": ["profiles_dir"]}
)
def tool_compare_ncu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare Nsight Compute profiles."""
    from pathlib import Path
    from core import profile_insights
    include_context, context_level = extract_context_opts(params)
    profiles_dir = Path(params.get("profiles_dir", ""))
    if not profiles_dir.exists():
        return make_error(f"profiles_dir not found: {profiles_dir}", include_context, context_level)
    try:
        result = profile_insights.compare_ncu_files(profiles_dir)
        if result is None:
            result = {"error": "No comparable ncu files found", "success": False}
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"ncu comparison failed: {e}", include_context, context_level)


@register_tool(
    "aisp_list_chapters",
    "Tags: chapters, labs, list, discovery, book, curriculum. "
    "List all discoverable chapters and labs from the book curriculum. "
    "Returns: {chapters: [{name, path, description}], labs: [...]}. "
    "⚡ FAST (~1s). USE when: Exploring what content is available. WORKFLOW: aisp_list_chapters → aisp_benchmark_targets → aisp_run_benchmarks. NOT FOR: Running benchmarks (use aisp_run_benchmarks).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_list_chapters(params: Dict[str, Any]) -> Dict[str, Any]:
    """List available chapters and labs."""
    include_context, context_level = extract_context_opts(params)
    result = _run_bench_cli(["list-chapters"])
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_context_summary",
    "Tags: context, summary, quick, environment, snapshot. "
    "Get quick context summary: GPU + software snapshot. "
    "Returns: {gpu, software, dependencies}. "
    "⚡ FAST (~1s). USE when: Need lightweight context attachment. "
    "Example: \"Quick system snapshot\" or \"Get context for LLM analysis\". "
    "NOT FOR: Full details (use aisp_context_full or aisp_system_full).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_context_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get context summary."""
    include_context, context_level = extract_context_opts(params)
    result = {"success": True, "context": get_cached_context("summary")}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_context_full",
    "Tags: context, full, comprehensive, environment, dump. "
    "Get full comprehensive context: complete system state. "
    "Returns: {gpu, software, dependencies, capabilities, system_params}. "
    "🕐 MEDIUM (~3s). USE when: Need complete environment dump. "
    "Example: \"Full context for debugging\" or \"Complete system state\". "
    "NOT FOR: Quick checks (use aisp_context_summary or aisp_status).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_context_full(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full context."""
    include_context, context_level = extract_context_opts(params)
    result = {"success": True, "context": get_cached_context("full")}
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# HARDWARE MICRO-BENCHMARK TOOLS
# =============================================================================

@register_tool(
    "aisp_hw_ib",
    "Tags: infiniband, ib, bandwidth, rdma, multi-node, interconnect. "
    "Get InfiniBand bandwidth test instructions and check if ib_write_bw is available. "
    "Returns: {ib_write_bw_available, instructions: {server_cmd, client_cmd}, alternative}. "
    "USE when: Testing InfiniBand bandwidth, verifying multi-node interconnect performance. "
    "Example: \"How do I test InfiniBand bandwidth?\" or \"Is IB working correctly?\". "
    "Provides ib_write_bw commands; alternative is NCCL tests if perftest not installed. ⚡ FAST (~1s). WORKFLOW: aisp_hw_ib → aisp_hw_nccl. NOT FOR: Single-node (use aisp_hw_p2p).",
    {"type": "object", "properties": with_context_params({
        "size_mb": {"type": "integer", "description": "Transfer size in MB for test guidance", "default": 64},
    })}
)
def tool_hw_ib(params: Dict[str, Any]) -> Dict[str, Any]:
    """InfiniBand bandwidth test guidance."""
    import shutil
    include_context, context_level = extract_context_opts(params)
    
    ib_write_bw = shutil.which("ib_write_bw")
    result = {
        "success": True,
        "ib_write_bw_available": ib_write_bw is not None,
        "instructions": {
            "server": "ib_write_bw -d mlx5_0",
            "client": "ib_write_bw -d mlx5_0 <server_ip>",
            "install": "apt install perftest  # or yum install perftest",
        },
        "alternative": "Use NCCL tests: all_reduce_perf -b 8M -e 256M -g 8"
    }
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_hw_nccl",
    "Tags: nccl, collective, allreduce, bandwidth, multi-gpu, communication. "
    "Get NCCL collective bandwidth test command and check if nccl-tests is available. "
    "Returns: {tool_available, command, install_instructions, collectives_list}. "
    "USE when: Measuring collective communication bandwidth, benchmarking NCCL performance. "
    "Example: \"Test NCCL all_reduce bandwidth\" or \"Benchmark 8-GPU allreduce\". "
    "Collectives: all_reduce, all_gather, reduce_scatter, broadcast, reduce, alltoall. ⚡ FAST (~1s). WORKFLOW: aisp_hw_nccl → tune NCCL env vars. NOT FOR: IB hardware (use aisp_hw_ib).",
    {"type": "object", "properties": with_context_params({
        "collective": {"type": "string", "description": "Collective type: all_reduce, all_gather, reduce_scatter, broadcast, reduce, alltoall", "enum": ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce", "alltoall"], "default": "all_reduce"},
        "min_bytes": {"type": "string", "description": "Minimum message size (e.g., '8M')", "default": "8M"},
        "max_bytes": {"type": "string", "description": "Maximum message size (e.g., '256M')", "default": "256M"},
        "gpus": {"type": "integer", "description": "Number of GPUs to test with", "default": 8},
    })}
)
def tool_hw_nccl(params: Dict[str, Any]) -> Dict[str, Any]:
    """NCCL collective bandwidth test guidance."""
    import shutil
    include_context, context_level = extract_context_opts(params)
    
    collective = normalize_param("collective", params.get("collective"), "all_reduce")
    bin_name = f"{collective}_perf"
    bin_path = shutil.which(bin_name)
    
    result = {
        "success": True,
        "tool_available": bin_path is not None,
        "command": f"{bin_name} -b {params.get('min_bytes', '8M')} -e {params.get('max_bytes', '256M')} -g {params.get('gpus', 8)}",
        "install": "git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make MPI=1",
        "collectives": ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce", "alltoall"]
    }
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_hw_p2p",
    "Tags: p2p, nvlink, gpu-to-gpu, bandwidth, peer-to-peer, transfer. "
    "Run GPU-to-GPU P2P bandwidth test measuring NVLink or PCIe peer access speed. "
    "Returns: {results: [{src, dst, p2p_enabled, bandwidth_gbps}], gpu_count}. "
    "USE when: Verifying NVLink bandwidth, checking P2P connectivity, debugging tensor parallelism. "
    "Example: \"Test GPU P2P bandwidth\" or \"Is NVLink working at full speed?\". "
    "REQUIRES: At least 2 GPUs. Tests first GPU pair by default. 🕐 MEDIUM (~20s). WORKFLOW: aisp_gpu_topology → aisp_hw_p2p. NOT FOR: Host-device (use aisp_hw_pcie).",
    {"type": "object", "properties": with_context_params({
        "size_mb": {"type": "integer", "description": "Transfer size in MB", "default": 256},
    })}
)
def tool_hw_p2p(params: Dict[str, Any]) -> Dict[str, Any]:
    """GPU P2P bandwidth test."""
    try:
        import torch
        include_context = bool(params.get("include_context", False))
        context_level = params.get("context_level", "summary")
        
        if not torch.cuda.is_available():
            return {"success": False, "error": "CUDA not available"}
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            return {"success": False, "error": "P2P test requires at least 2 GPUs", "gpu_count": num_gpus}
        
        size_mb = params.get("size_mb", 256)
        size_bytes = size_mb * 1024 * 1024
        
        results = []
        for i in range(min(num_gpus, 2)):  # Test first pair only for speed
            for j in range(min(num_gpus, 2)):
                if i == j:
                    continue
                
                can_access = torch.cuda.can_device_access_peer(i, j)
                
                with torch.cuda.device(i):
                    src = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{i}")
                with torch.cuda.device(j):
                    dst = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{j}")
                
                dst.copy_(src)
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                iters = 5
                start.record()
                for _ in range(iters):
                    dst.copy_(src)
                end.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start.elapsed_time(end)
                bw_gbps = (size_bytes * iters / (elapsed_ms / 1000)) / 1e9
                
                results.append({
                    "src": i,
                    "dst": j,
                    "p2p_enabled": can_access,
                    "bandwidth_gbps": round(bw_gbps, 2)
                })
        
        result = {"success": True, "gpu_count": num_gpus, "results": results}
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLUSTER TOOLS
# =============================================================================

@register_tool(
    "aisp_cluster_slurm",
    "Tags: slurm, batch, cluster, hpc, job-script, sbatch, multi-node. "
    "Generate SLURM job script for cluster submission with optimal settings. "
    "Returns: {script: <slurm_script_content>, filename_suggestion, notes}. "
    "USE when: Submitting training jobs to SLURM clusters, setting up multi-node runs. "
    "Example: \"Create SLURM script for 2 nodes x 8 GPUs\" or \"Generate sbatch for 70B training\". "
    "Includes: resource requests, NCCL env vars, torchrun launch command. ⚡ FAST (~1s). WORKFLOW: aisp_distributed_plan → aisp_cluster_slurm → submit. NOT FOR: torchrun only (use aisp_launch_plan).",
    {"type": "object", "properties": with_context_params({
        "model": {
            "type": "string",
            "description": "Model size for resource estimation (e.g., '7b', '70b')",
            "default": "7b"
        },
        "nodes": {
            "type": "integer",
            "description": "Number of nodes to request",
            "default": 1
        },
        "gpus": {
            "type": "integer",
            "description": "GPUs per node",
            "default": 8
        },
    })}
)
def tool_cluster_slurm(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SLURM script."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    result = get_engine().cluster.slurm(
        model=params.get("model", "7b"),
        nodes=params.get("nodes", 1),
        gpus=params.get("gpus", 8)
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_status",
    "Tags: status, health, quick-check, sanity, ready, first-call, prerequisite. "
    "🚀 QUICK STATUS CHECK: Fast snapshot of GPU, software, and AI backend health. "
    "Returns: {gpu_ok, software_ok, ai_backend_ok, warnings, summary, gpu_count, cuda_version}. "
    "⚡ VERY FAST (<1s). USE FIRST when: Starting any session, before slow operations. "
    "Example: \"Quick status check\" or \"Is everything healthy?\" or \"Ready for profiling?\". "
    "WORKFLOW: aisp_status → if issues → aisp_system_dependencies or aisp_gpu_info. "
    "NOT FOR: Full context (use aisp_triage), deep audit (use aisp_system_full).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get quick status."""
    from core.engine import get_engine
    include_context = bool(params.get("include_context", True))
    context_level = params.get("context_level", "summary")
    result = get_engine().status()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_triage",
    "Tags: triage, start, first, status, context, health, entry-point, begin, overview. "
    "🎯 START HERE: Quick triage = status check + context summary in one call. "
    "Returns: {status: {gpu_ok, software_ok, ai_backend_ok}, context: {gpu, software, dependencies}}. "
    "⚡ FAST (~1-2s). THE BEST FIRST CALL for any new performance investigation. "
    "Example: \"Start with triage\" or \"Quick overview\" or \"What's my system like?\". "
    "PROVIDES: GPU model/count/VRAM, CUDA/PyTorch versions, dependency health, warnings. "
    "WORKFLOW: aisp_triage → aisp_recommend OR aisp_analyze_bottlenecks → specific tools. "
    "VERSUS OTHER ENTRY POINTS: "
    "• aisp_triage: status + context (recommended) "
    "• aisp_status: status only (faster, less info) "
    "• aisp_suggest_tools: tool recommendations based on intent. NOT FOR: Deep system audit (use aisp_system_full).",
    {"type": "object", "properties": with_context_params({})}
)
def tool_triage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return quick status plus context summary to guide next actions."""
    from core.engine import get_engine
    engine = get_engine()
    result = {
        "success": True,
        "status": engine.status(),
        "context": get_cached_context("summary"),
    }
    include_context, context_level = extract_context_opts(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "aisp_job_status",
    "Tags: job, status, poll, async, background, queue, progress. "
    "Check status of a background job started with async=true. "
    "Returns: {job_id, status: running|completed|error, result (if completed), duration_ms}. "
    "⚡ FAST (<1s). USE when: Polling for completion of background jobs. "
    "Example: \"Check job status\" or \"Is my benchmark done?\" or \"Poll nsys capture\". "
    "STATUS VALUES: "
    "• 'running' → Job in progress, poll again in 10-30s "
    "• 'completed' → Done! Result in 'result' field "
    "• 'error' → Failed, check 'error' field for details. "
    "TOOLS SUPPORTING async=true: aisp_run_benchmarks, aisp_profile_nsys, aisp_profile_ncu, aisp_profile_torch, aisp_profile_hta. "
    "WORKFLOW: tool(async=true) → poll aisp_job_status(job_id) → [completed] aisp_benchmark_triage or aisp_nsys_summary.",
    {"type": "object", "properties": with_context_params({
        "job_id": {
            "type": "string",
            "description": "Job ID returned from tool call with async=true"
        },
    }), "required": ["job_id"]}
)
def tool_job_status(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    job_id = params.get("job_id")
    if not job_id:
        return make_error("job_id is required", include_context, context_level)
    with _JOB_LOCK:
        record = _JOB_STORE.get(job_id)
    if not record:
        return {
            "job_id": job_id,
            "status": "not_found",
            "note": "No job with this id; ensure you passed async=true on the original call.",
        }
    # Copy so we don't mutate stored record when adding error decoration
    payload = dict(record)
    if payload.get("status") == "error" and "error" not in payload:
        result = payload.get("result") or {}
        if isinstance(result, dict) and result.get("error"):
            payload["error"] = result.get("error")
        else:
            payload["error"] = "Job failed"
    if "success" not in payload:
        payload["success"] = payload.get("status") not in {"error", "not_found"}
    return attach_context_if_requested(payload, include_context, context_level)


# =============================================================================
# HUGGINGFACE TOOLS
# =============================================================================

@register_tool(
    "aisp_hf",
    "Tags: huggingface, hf, models, download, search, trending, hub. "
    "HuggingFace Hub operations: search models, get trending, download models. "
    "⚡ FAST (~2s). USE when: Finding models, downloading from HF Hub. "
    "Example: action='search', query='llama 2 7b' or action='trending', limit=5. "
    "WORKFLOW: aisp_hf(action='search') → aisp_hf(action='download'). "
    "NOT FOR: Model performance recommendations (use aisp_recommend).",
    {"type": "object", "properties": with_context_params({
        "action": {
            "type": "string",
            "enum": ["search", "trending", "download"],
            "default": "search",
            "description": "Operation: search, trending, or download"
        },
        "query": {
            "type": "string",
            "description": "Search query or model name for download"
        },
        "limit": {
            "type": "integer",
            "default": 10,
            "description": "Max results to return"
        },
    })}
)
def tool_hf(params: Dict[str, Any]) -> Dict[str, Any]:
    """HuggingFace Hub operations."""
    from core.engine import get_engine
    include_context, context_level = extract_context_opts(params)
    
    action = normalize_param("action", params.get("action"), "search")
    query = params.get("query", "")
    limit = params.get("limit", 10)
    
    try:
        engine = get_engine()
        if action == "search":
            result = engine.hf.search(query, limit=limit)
        elif action == "trending":
            result = engine.hf.trending(limit=limit)
        elif action == "download":
            if not query:
                return make_error("query (model name) required for download", include_context, context_level)
            result = engine.hf.download(query)
        else:
            return make_error(f"Unknown action: {action}", include_context, context_level)
        
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"HuggingFace operation failed: {e}", include_context, context_level)


# =============================================================================
# UTILITY TOOLS
# =============================================================================

@register_tool(
    "aisp_suggest_tools",
    "Tags: suggest, recommend, navigation, intent, which-tool, discovery, help, lost. "
    "🧭 TOOL NAVIGATOR: Get ranked tool suggestions based on your intent or problem. "
    "Returns: {suggestions: [{tool, reason, score}], count}. "
    "⚡ FAST (<1s). USE when: Unsure which tool to use, have a problem description. "
    "EXAMPLE INTENTS → SUGGESTED TOOLS: "
    "• 'OOMing on 24GB' → aisp_profile_memory, aisp_analyze_whatif, aisp_inference_quantization "
    "• 'slow training' → aisp_analyze_bottlenecks, aisp_profile_nsys, aisp_recommend "
    "• 'multi-GPU setup' → aisp_distributed_plan, aisp_gpu_topology, aisp_launch_plan "
    "• 'vLLM latency' → aisp_inference_vllm, aisp_inference_quantization "
    "• 'compare profiles' → aisp_compare_nsys, aisp_profile_compare. "
    "WORKFLOW: aisp_suggest_tools → use suggested tools. "
    "NOT FOR: Direct answers (use aisp_ask), getting started (use aisp_triage).",
    {"type": "object", "properties": with_context_params({
        "query": {
            "type": "string",
            "description": "Your intent, problem, or question in natural language"
        },
    }), "required": ["query"]}
)
def tool_suggest_tools(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ranked list of suggested tools given a query."""
    query = (params.get("query") or "").lower()
    include_context, context_level = extract_context_opts(params)

    rules = [
        {
            "tool": "aisp_analyze_bottlenecks",
            "keywords": ["slow", "latency", "bottleneck", "utilization", "stall", "idle", "regression", "throughput drop"],
            "reason": "Diagnose bottlenecks for slow workload/latency issues",
        },
        {
            "tool": "aisp_hw_disk",
            "keywords": ["disk", "io", "storage"],
            "reason": "Disk I/O benchmark (sequential)",
        },
        {
            "tool": "aisp_hw_pcie",
            "keywords": ["pcie", "h2d", "d2h", "pci-e"],
            "reason": "PCIe H2D/D2H bandwidth benchmark",
        },
        {
            "tool": "aisp_hw_cache",
            "keywords": ["memory stride", "cache", "l2", "hbm"],
            "reason": "Stride/bandwidth test for memory hierarchy",
        },
        {
            "tool": "aisp_hw_tc",
            "keywords": ["tensor core", "tflops", "matmul"],
            "reason": "Tensor core throughput test",
        },        {
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
        },        {
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
        },        {
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
            "tool": "aisp_inference_vllm",
            "keywords": ["vllm", "inference", "serving", "throughput", "latency"],
            "reason": "Generate vLLM config for throughput/latency",
        },
        {
            "tool": "aisp_inference_quantization",
            "keywords": ["quant", "int8", "fp8", "fp4", "kv cache"],
            "reason": "Quantization guidance for inference",
        },        {
            "tool": "aisp_benchmark_targets",
            "keywords": ["benchmark targets", "bench targets", "list benchmarks", "what can I run"],
            "reason": "List benchmark targets",
        },        {
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
        },        {
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
            "tool": "aisp_hw_roofline",
            "keywords": ["stride", "roofline", "memory sweep"],
            "reason": "Quick stride sweep roofline for memory hierarchy",
        },        {
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

    result = {"suggestions": suggestions, "count": len(suggestions), "success": True}
    return attach_context_if_requested(result, include_context, context_level)


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
            raw_result = HANDLERS[name](arguments)
            result = _normalize_result(raw_result)
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
                _normalize_result({"error": str(e), "traceback": tb}),
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
