"""Registry describing all runnable examples for profiling harnesses."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class ExampleKind(str, Enum):
    PYTHON = "python"
    CUDA = "cuda"
    SHELL = "shell"
    BINARY = "binary"


@dataclass(frozen=True)
class BuildStep:
    command: Tuple[str, ...]
    workdir: Path
    outputs: Tuple[Path, ...] = ()
    env: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass(frozen=True)
class Example:
    """Metadata wrapper for an example entry."""

    name: str
    path: Path
    description: str
    default_args: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    requires_modules: List[str] = field(default_factory=list)
    optional_modules: List[str] = field(default_factory=list)
    requires_commands: List[str] = field(default_factory=list)
    requires_cuda: bool = False
    min_cuda_gpus: int = 0
    allow_cpu_fallback: bool = True
    env: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    kind: ExampleKind = ExampleKind.PYTHON
    source: Optional[Path] = None
    build_steps: Tuple[BuildStep, ...] = field(default_factory=tuple)
    run_command: Optional[Tuple[str, ...]] = None
    workdir: Optional[Path] = None

    def resolved_path(self, repo_root: Optional[Path] = None) -> Path:
        base = repo_root if repo_root is not None else Path.cwd()
        return (base / self.path).resolve()

    def resolved_source(self, repo_root: Optional[Path] = None) -> Path:
        source_path = self.source if self.source is not None else self.path
        base = repo_root if repo_root is not None else Path.cwd()
        return (base / source_path).resolve()

    def resolved_workdir(self, repo_root: Optional[Path] = None) -> Path:
        if self.workdir is not None:
            base = repo_root if repo_root is not None else Path.cwd()
            return (base / self.workdir).resolve()
        if self.kind is ExampleKind.CUDA and self.source is not None:
            return self.resolved_source(repo_root).parent
        return self.resolved_path(repo_root).parent


def _example(
    name: str,
    path: str,
    description: str,
    *,
    default_args: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    requires_modules: Optional[Iterable[str]] = None,
    optional_modules: Optional[Iterable[str]] = None,
    requires_commands: Optional[Iterable[str]] = None,
    requires_cuda: bool = False,
    min_cuda_gpus: int = 0,
    allow_cpu_fallback: bool = True,
    env: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
    kind: ExampleKind = ExampleKind.PYTHON,
    source: Optional[str] = None,
    build_steps: Optional[Sequence[BuildStep]] = None,
    run_command: Optional[Sequence[str]] = None,
    workdir: Optional[str] = None,
) -> Example:
    return Example(
        name=name,
        path=Path(path),
        description=description,
        default_args=list(default_args or []),
        tags=sorted(set(tags or [])),
        requires_modules=sorted(set(requires_modules or [])),
        optional_modules=sorted(set(optional_modules or [])),
        requires_commands=sorted(set(requires_commands or [])),
        requires_cuda=requires_cuda,
        min_cuda_gpus=min_cuda_gpus,
        allow_cpu_fallback=allow_cpu_fallback,
        env=dict(env or {}),
        timeout_seconds=timeout_seconds,
        kind=ExampleKind(kind.value if isinstance(kind, ExampleKind) else kind),
        source=Path(source) if source else None,
        build_steps=tuple(build_steps or ()),
        run_command=tuple(run_command) if run_command else None,
        workdir=Path(workdir) if workdir else None,
    )


def _format_chapter_tag(chapter: str) -> str:
    if chapter.startswith("ch"):
        suffix = chapter[2:]
        if suffix.isdigit():
            return f"ch{int(suffix):02d}"
    return chapter


def _discover_cuda_examples(existing_names: Optional[Iterable[str]] = None) -> List[Example]:
    # core/scripts/ is now in code/, so __file__ is .../code/core/scripts/example_registry.py
    # parents[0] = .../code/scripts, parents[1] = .../code
    code_root = Path(__file__).resolve().parents[1]
    existing = set(existing_names or [])
    discovered: List[Example] = []

    if not code_root.exists():
        return discovered

    for cu_path in sorted(code_root.glob("ch*/**/*.cu")):
        chapter_dir = cu_path.parent
        chapter = chapter_dir.relative_to(code_root).parts[0]
        chapter_tag = _format_chapter_tag(chapter)

        rel_source = cu_path.relative_to(code_root)
        rel_to_code = cu_path.relative_to(code_root)
        name = "_".join(rel_to_code.with_suffix("").parts)
        if name in existing:
            continue

        binary_rel = Path("build") / rel_to_code.with_suffix("")
        build_step = BuildStep(
            command=(
                "nvcc",
                "-O3",
                "-std=c++17",
                "-arch=sm_100",
                "--expt-relaxed-constexpr",
                "-o",
                str(binary_rel),
                str(rel_source),
            ),
            workdir=code_root,
            outputs=(binary_rel,),
            description="Compile CUDA example for profiling",
        )

        discovered.append(
            Example(
                name=name,
                path=binary_rel,
                description=f"CUDA example {rel_to_code.name} ({chapter_tag})",
                default_args=[],
                tags=sorted({chapter_tag, "cuda"}),
                requires_modules=[],
                optional_modules=[],
                requires_commands=["nvcc"],
                requires_cuda=True,
                min_cuda_gpus=1,
                allow_cpu_fallback=False,
                env={},
                timeout_seconds=300,
                kind=ExampleKind.CUDA,
                source=rel_source,
                build_steps=(build_step,),
                run_command=None,
                workdir=None,
            )
        )
        existing.add(name)

    return discovered


EXAMPLES: List[Example] = [
    _example(
        name="ch01_performance_basics",
        path="ch01/baseline_performance.py",
        description="Goodput and profiling primer from Chapter 1.",
        tags=["ch01", "profiling", "training"],
        requires_modules=["torch", "psutil", "GPUtil", "numpy"],
        requires_cuda=False,
        allow_cpu_fallback=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch02_hardware_info",
        path="ch02/hardware_info.py",
        description="Hardware topology and monitoring walkthrough.",
        tags=["ch02", "system", "monitoring"],
        requires_modules=["torch", "psutil", "GPUtil", "numpy"],
        requires_cuda=False,
        allow_cpu_fallback=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch03_bind_numa_affinity",
        path="ch03/bind_numa_affinity.py",
        description="NUMA-aware data pipeline and affinity demo.",
        tags=["ch03", "system", "numa"],
        requires_modules=["torch", "psutil"],
        requires_commands=["numactl"],
        requires_cuda=True,
        allow_cpu_fallback=False,
        timeout_seconds=600,
    ),
    _example(
        name="ch04_before_dataparallel",
        path="ch04/before_dataparallel.py",
        description="Baseline single-GPU training before DataParallel.",
        tags=["ch04", "distributed", "baseline"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch04_before_no_overlap",
        path="ch04/before_no_overlap.py",
        description="Distributed training without communication overlap.",
        tags=["ch04", "distributed", "ddp"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch04_before_reinit_comm",
        path="ch04/before_reinit_comm.py",
        description="Communication-heavy workload prior to comm optimizations.",
        tags=["ch04", "distributed", "communication"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch04_after_ddp",
        path="ch04/after_ddp.py",
        description="Optimized DDP setup with improved overlap.",
        tags=["ch04", "distributed", "ddp"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch04_after_overlap_ddp",
        path="ch04/after_overlap_ddp.py",
        description="Gradient bucketing and comm overlap example.",
        tags=["ch04", "distributed", "overlap"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch04_after_reinit_comm",
        path="ch04/after_reinit_comm.py",
        description="Communication-optimized training loop post refactor.",
        tags=["ch04", "distributed", "communication"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch04_barrier_straggler",
        path="ch04/barrier_straggler.py",
        description="Barrier synchronization and straggler analysis.",
        tags=["ch04", "distributed", "synchronization"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch04_dist_allreduce",
        path="ch04/dist_allreduce.py",
        description="Gloo and NCCL all-reduce benchmark.",
        tags=["ch04", "distributed", "collective"],
        requires_modules=["torch"],
        requires_cuda=True,
        default_args=["--data-size", "1048576"],
        timeout_seconds=600,
    ),
    _example(
        name="ch04_nccl_benchmark",
        path="ch04/nccl_benchmark.py",
        description="Comprehensive NCCL collective benchmark suite.",
        tags=["ch04", "distributed", "collective"],
        requires_modules=["torch"],
        requires_cuda=True,
        default_args=[
            "--world-size", "1",
            "--max-size", "32",
            "--warmup", "1",
            "--trials", "3",
            "--operation", "allreduce",
            "--dtype", "float16",
        ],
        timeout_seconds=900,
    ),
    _example(
        name="ch04_ucx_fragmentation",
        path="ch04/ucx_fragmentation.py",
        description="UCX fragmentation avoidance strategies.",
        tags=["ch04", "distributed", "memory"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch05_gpudirect_storage",
        path="ch05/gpudirect_storage_example.py",
        description="GPUDirect Storage pipeline and monitoring demo.",
        tags=["ch05", "io", "storage"],
        requires_modules=["torch", "psutil", "numpy"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch05_storage_io_optimization",
        path="ch05/storage_io_optimization.py",
        description="High-throughput DataLoader and storage optimizations.",
        tags=["ch05", "io", "dataloader"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch06_add_sequential",
        path="ch06/add_sequential.py",
        description="Sequential kernel launch anti-pattern.",
        tags=["ch06", "kernel", "baseline"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch06_add_parallel",
        path="ch06/add_parallel.py",
        description="Fused parallel kernel addition example.",
        tags=["ch06", "kernel", "optimization"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch07_memory_access",
        path="ch07/memory_access_pytorch.py",
        description="Coalesced vs uncoalesced memory access patterns.",
        tags=["ch07", "memory", "access"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch07_vectorized",
        path="ch07/vectorized_pytorch.py",
        description="Vectorized operations for bandwidth efficiency.",
        tags=["ch07", "memory", "vectorization"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch07_lookup",
        path="ch07/lookup_pytorch.py",
        description="Embedding lookup optimization strategies.",
        tags=["ch07", "memory", "embedding"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch07_matmul",
        path="ch07/matmul_pytorch.py",
        description="Matmul optimization with tiling examples.",
        tags=["ch07", "compute", "matmul"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch08_warp_divergence",
        path="ch08/warp_divergence_pytorch.py",
        description="Warp divergence and control flow efficiency study.",
        tags=["ch08", "warp", "control-flow"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch08_occupancy",
        path="ch08/occupancy_pytorch.py",
        description="Occupancy tuning and launch configuration demo.",
        tags=["ch08", "warp", "occupancy"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch08_ilp",
        path="ch08/ilp_pytorch.py",
        description="Instruction level parallelism experiment.",
        tags=["ch08", "warp", "ilp"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch09_fusion",
        path="ch09/fusion_pytorch.py",
        description="Kernel fusion and compiler-assisted optimizations.",
        tags=["ch09", "compiler", "fusion"],
        requires_modules=["torch", "numpy"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch13_custom_allocator",
        path="ch13/custom_allocator.py",
        description="Custom CUDA memory allocator strategies.",
        tags=["ch13", "memory", "allocator"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch13_memory_profiling",
        path="ch13/memory_profiling.py",
        description="Memory profiling workflow for large models.",
        tags=["ch13", "memory", "profiling"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch13_fsdp",
        path="ch13/fsdp_example.py",
        description="Fully Sharded Data Parallel training demo.",
        tags=["ch13", "distributed", "fsdp"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch13_train_deepseek_coder",
        path="ch13/train_deepseek_coder.py",
        description="DeepSeek Coder 6.7B training workflow with profiling.",
        tags=["ch13", "training", "llm"],
        requires_modules=["torch", "transformers"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch14_torch_compiler_examples",
        path="ch14/torch_compiler_examples.py",
        description="Torch compile and dynamo optimization suite.",
        tags=["ch14", "compiler", "torch-compile"],
        requires_modules=["torch", "psutil", "GPUtil"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch14_triton_examples",
        path="ch14/triton_examples.py",
        description="Custom Triton kernels and PyTorch integration.",
        tags=["ch14", "triton", "compiler"],
        requires_modules=["torch", "triton"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch15_disaggregated_inference",
        path="ch15/disaggregated_inference.py",
        description="Disaggregated inference and pipeline parallel patterns.",
        tags=["ch15", "inference", "pipeline"],
        requires_modules=["torch", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch16_radix_attention",
        path="ch16/radix_attention_example.py",
        description="Radix attention and fast prefill kernels.",
        tags=["ch16", "attention", "radix"],
        requires_modules=["torch", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch16_inference_profiling",
        path="ch16/inference_profiling.py",
        description="Inference profiling dashboard and trace capture.",
        tags=["ch16", "inference", "profiling"],
        requires_modules=["torch", "psutil", "GPUtil", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch16_moe_benchmark",
        path="ch16/moe_performance_benchmark.py",
        description="Mixture-of-experts throughput benchmark.",
        tags=["ch16", "moe", "benchmark"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch16_inference_load_test",
        path="ch16/inference_server_load_test.py",
        description="Synthetic load testing harness for 8x B200 server.",
        tags=["ch16", "inference", "benchmark"],
        requires_modules=["torch", "numpy"],
        requires_cuda=True,
        timeout_seconds=1800,
    ),
    _example(
        name="ch16_multi_gpu_validation",
        path="ch16/multi_gpu_validation.py",
        description="Tensor-parallel consistency validation for GPT benchmark.",
        tags=["ch16", "validation", "tensor-parallel"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch16_perplexity_eval",
        path="ch16/perplexity_eval.py",
        description="Compute perplexity on tokenized datasets using the GPT benchmark model.",
        tags=["ch16", "evaluation", "perplexity"],
        requires_modules=["torch"],
        requires_cuda=False,
        timeout_seconds=900,
    ),
    _example(
        name="tool_memory_profiler",
        path="core/profiling/memory_profiler.py",
        description="Profiler wrapper focused on CUDA memory diagnostics.",
        tags=["tools", "profiling"],
        requires_modules=["torch"],
        requires_cuda=False,
        timeout_seconds=600,
    ),
    _example(
        name="tool_power_monitor",
        path="core/scripts/power_monitor.py",
        description="Sample GPU power usage while executing a command.",
        tags=["tools", "power"],
        requires_modules=["pynvml"],
        requires_cuda=True,
        timeout_seconds=3600,
    ),
    _example(
        name="tool_continuous_benchmark",
        path="core/benchmark/continuous_benchmark.py",
        description="Run a suite of benchmarks and store aggregated results.",
        tags=["tools", "automation"],
        requires_modules=["json"],
        requires_cuda=False,
        timeout_seconds=3600,
    ),
    _example(
        name="ch17_dynamic_routing",
        path="ch17/dynamic_routing.py",
        description="Dynamic expert routing strategies for mixture-of-experts.",
        tags=["ch17", "routing", "moe"],
        requires_modules=["torch", "yaml"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch17_early_rejection",
        path="ch17/early_rejection.py",
        description="Early token rejection and speculative decoding.",
        tags=["ch17", "routing", "speculative"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch18_flexdecoding",
        path="ch18/baseline_flexdecoding.py",
        description="FlexAttention decoding pipeline (baseline/optimized pair).",
        tags=["ch18", "attention", "flex"],
        requires_modules=["torch", "numpy"],
        optional_modules=["flex_attention"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch18_test_flex_attention",
        path="ch18/test_flex_attention.py",
        description="Environment probe for flex_attention availability.",
        tags=["ch18", "attention", "flex"],
        requires_modules=["torch"],
        requires_cuda=False,
        allow_cpu_fallback=True,
    ),
    _example(
        name="ch16_flash_sdp_baseline",
        path="ch16/baseline_flash_sdp.py",
        description="Flash SDP attention (fail-fast if Flash kernels are unavailable).",
        tags=["ch16", "attention", "flash"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=300,
    ),
    _example(
        name="ch16_flash_sdp_optimized",
        path="ch16/optimized_flash_sdp.py",
        description="Flash SDP attention compiled via torch.compile (fail-fast if Flash kernels are unavailable).",
        tags=["ch16", "attention", "flash", "compile"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=420,
    ),
    _example(
        name="ch18_paged_attn",
        path="ch18/baseline_paged_attn.py",
        description="Baseline PyTorch SDPA + MoE dispatch for the paged-attention demo.",
        tags=["ch18", "attention", "moe"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch18_paged_attn_vllm",
        path="ch18/optimized_paged_attn_vllm.py",
        description="vLLM-inspired paged attention with fused RMSNorm and CUDA Graph capture.",
        tags=["ch18", "attention", "moe", "vllm"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch19_adaptive_parallelism",
        path="ch19/adaptive_parallelism_strategy.py",
        description="Adaptive parallelism orchestration for LLM inference.",
        tags=["ch19", "scheduling", "inference"],
        requires_modules=["torch", "psutil"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch19_token_precision_switching",
        path="ch19/token_precision_switching.py",
        description="Token precision switching and dynamic quantization.",
        tags=["ch19", "quantization", "inference"],
        requires_modules=["torch", "transformers", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch20_ai_kernel_generator",
        path="ch20/ai_kernel_generator.py",
        description="AI-assisted CUDA kernel generation workflow.",
        tags=["ch20", "automation", "kernel"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
]


EXAMPLES.extend(_discover_cuda_examples(example.name for example in EXAMPLES))


EXAMPLE_BY_NAME = {example.name: example for example in EXAMPLES}


def list_example_names() -> List[str]:
    return sorted(EXAMPLE_BY_NAME)
