"""Tools commands mounted by aisp (`aisp tools ...`).

Tools are non-comparable utilities and analysis scripts. They are intentionally
separated from:
  - `aisp bench run` (comparative baseline vs optimized benchmarks), and
  - `aisp demos` (runnable chapter demos / examples).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import typer
    from typer import Argument

    TYPER_AVAILABLE = True
except ImportError:  # pragma: no cover - Typer is optional for docs builds
    TYPER_AVAILABLE = False
    typer = None  # type: ignore
    Argument = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    script_path: Path
    description: str


TOOLS: Dict[str, ToolSpec] = {
    "kv-cache": ToolSpec(
        name="kv-cache",
        script_path=REPO_ROOT / "core" / "scripts" / "utilities" / "kv_cache_calc.py",
        description="KV-cache size calculator.",
    ),
    "cost-per-token": ToolSpec(
        name="cost-per-token",
        script_path=REPO_ROOT / "core" / "scripts" / "utilities" / "calculate_cost_per_token.py",
        description="Cost-per-token calculator.",
    ),
    "compare-precision": ToolSpec(
        name="compare-precision",
        script_path=REPO_ROOT / "core" / "scripts" / "utilities" / "compare_precision_accuracy.py",
        description="Compare precision/accuracy tradeoffs.",
    ),
    "detect-cutlass": ToolSpec(
        name="detect-cutlass",
        script_path=REPO_ROOT / "core" / "scripts" / "utilities" / "detect_cutlass_info.py",
        description="Detect CUTLASS info from the environment.",
    ),
    "dump-hw": ToolSpec(
        name="dump-hw",
        script_path=REPO_ROOT / "core" / "scripts" / "utilities" / "dump_hardware_capabilities.py",
        description="Dump hardware capability JSON.",
    ),
    "probe-hw": ToolSpec(
        name="probe-hw",
        script_path=REPO_ROOT / "core" / "scripts" / "utilities" / "probe_hardware_capabilities.py",
        description="Probe hardware capabilities (recommended).",
    ),
    "roofline": ToolSpec(
        name="roofline",
        script_path=REPO_ROOT / "ch08" / "roofline.py",
        description="Run the roofline analysis tool (chapter utility, not a benchmark pair).",
    ),
    "occupancy-tuning": ToolSpec(
        name="occupancy-tuning",
        script_path=REPO_ROOT / "ch08" / "occupancy_tuning_tool.py",
        description="Run the occupancy tuning sweep tool (chapter utility, not a benchmark pair).",
    ),
    "dynamic-router-eval": ToolSpec(
        name="dynamic-router-eval",
        script_path=REPO_ROOT / "labs" / "dynamic_router" / "cheap_eval.py",
        description="Run the dynamic-router cheap eval stack (tool, not a benchmark pair).",
    ),
    "vllm-monitoring": ToolSpec(
        name="vllm-monitoring",
        script_path=REPO_ROOT / "ch16" / "vllm_monitoring.py",
        description="Emit Prometheus/Grafana monitoring bundle for vLLM v1 metrics.",
    ),
    "spec-config-sweep": ToolSpec(
        name="spec-config-sweep",
        script_path=REPO_ROOT / "ch18" / "speculative_decode" / "spec_config_sweep.py",
        description="Sweep speculative-decoding config files and write summary JSON.",
    ),
    "cudagraph-bucketing": ToolSpec(
        name="cudagraph-bucketing",
        script_path=REPO_ROOT / "ch18" / "cudagraph_bucketing_simulator.py",
        description="Run the Chapter 18 CUDA graph bucketing simulator (tool; not a benchmark pair).",
    ),
    "uma-memory": ToolSpec(
        name="uma-memory",
        script_path=REPO_ROOT / "labs" / "uma_memory" / "uma_memory_reporting.py",
        description="Report UMA allocatable memory snapshot (tool; not a benchmark pair).",
    ),
    "moe-parallelism": ToolSpec(
        name="moe-parallelism",
        script_path=REPO_ROOT / "labs" / "moe_parallelism" / "run_lab.py",
        description="Run the MoE parallelism planner (tool; not a benchmark pair).",
    ),
    "moe-validation": ToolSpec(
        name="moe-validation",
        script_path=REPO_ROOT / "ch15" / "moe_validation" / "moe_validation.py",
        description="Sweep MoE routing guardrails and report overflow/Gini/entropy + throughput.",
    ),
    "kv-cache-math": ToolSpec(
        name="kv-cache-math",
        script_path=REPO_ROOT / "ch15" / "kv_cache_management_math.py",
        description="Run the math-only KV-cache attention tool (chapter utility).",
    ),
    "fp8-perchannel-bench": ToolSpec(
        name="fp8-perchannel-bench",
        script_path=REPO_ROOT / "ch13" / "fp8_perchannel_bench.py",
        description="Run the FP8 per-channel scaling benchmark (chapter utility).",
    ),
    "flex-attention-cute": ToolSpec(
        name="flex-attention-cute",
        script_path=REPO_ROOT / "labs" / "flexattention" / "flex_attention_cute.py",
        description="Run the FlashAttention CuTe backend tool (FlexAttention fallback utility).",
    ),
    "dtensor-mesh": ToolSpec(
        name="dtensor-mesh",
        script_path=REPO_ROOT / "ch13" / "dtensor_mesh_tool.py",
        description="Run the DTensor mesh setup utility (chapter tool; not a benchmark pair).",
    ),
    "nvfp4-trtllm": ToolSpec(
        name="nvfp4-trtllm",
        script_path=REPO_ROOT / "ch18" / "nvfp4_trtllm_tool.py",
        description="Run the NVFP4/TRT-LLM integration utility (chapter tool; not a benchmark pair).",
    ),
    "fp8-calibration-free": ToolSpec(
        name="fp8-calibration-free",
        script_path=REPO_ROOT / "ch19" / "fp8_calibration_free_tool.py",
        description="Run the calibration-free FP8 serving utility (chapter tool; not a benchmark pair).",
    ),
    "kernel-verification": ToolSpec(
        name="kernel-verification",
        script_path=REPO_ROOT / "ch20" / "kernel_verification_tool.py",
        description="Run Chapter 20 kernel verification utility (tool; not a benchmark pair).",
    ),
    "proofwright-verify": ToolSpec(
        name="proofwright-verify",
        script_path=REPO_ROOT / "ch20" / "proofwright_verify_tool.py",
        description="Run Chapter 20 ProofWright verification utility (tool; not a benchmark pair).",
    ),
}


def _run_tool(tool: str, tool_args: Optional[List[str]]) -> int:
    spec = TOOLS.get(tool)
    if spec is None:
        raise ValueError(f"Unknown tool '{tool}'.")
    if not spec.script_path.exists():
        raise FileNotFoundError(f"Tool script not found at {spec.script_path}")

    cmd = [sys.executable, str(spec.script_path), *(tool_args or [])]
    result = subprocess.run(cmd)
    return int(result.returncode)


if TYPER_AVAILABLE:
    app = typer.Typer(
        name="tools",
        help="Tools: run non-benchmark utilities and analysis scripts",
        add_completion=False,
    )

    @app.command("list", help="List available tools.")
    def list_tools() -> None:
        for name in sorted(TOOLS):
            typer.echo(f"{name}: {TOOLS[name].description}")

    def _make_tool_command(tool_name: str, description: str):
        def _cmd(
            tool_args: Optional[List[str]] = Argument(
                None,
                help="Arguments forwarded to the tool (use -- to separate).",
            ),
        ) -> None:
            try:
                exit_code = _run_tool(tool_name, tool_args)
            except (ValueError, FileNotFoundError) as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1)
            raise typer.Exit(code=exit_code)

        _cmd.__name__ = tool_name.replace("-", "_")
        _cmd.__doc__ = description
        return _cmd

    for _name, _spec in sorted(TOOLS.items()):
        app.command(_name, help=_spec.description)(_make_tool_command(_name, _spec.description))
else:
    app = None  # type: ignore
