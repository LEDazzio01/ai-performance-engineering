# Capstone: Blackwell Cluster GEMM

This chapter ties the entire “AI performance engineering” storyline together.
It now exposes two complementary layers:

1. **Scenario suite** – full-stack baseline/optimized pairs that replay every
   chapter’s techniques through a single harness (`capstone/scenario_benchmark.py`).
2. **Cluster GEMM kernels** – the original CUDA 13 kernels (baseline vs clustered
   TMA) that integrate directly with PyTorch.

### Cluster GEMM kernels

The CUDA kernels live in `capstone/capstone_kernels.cu` and
`capstone/capstone_kernels_tcgen05.cu`:

| Kernel | Path | What it does |
| --- | --- | --- |
| `baseline_matmul` | `capstone/capstone_kernels.cu` | A naïve triple loop GEMM that touches global memory for every multiply–accumulate. |
| `optimized_cluster_tma` | `capstone/capstone_kernels.cu` | A fully pipelined Blackwell kernel that uses thread-block clusters, DSM-broadcasted B tiles, Tensor Memory Accelerator (TMA), warp specialization, and asynchronous tensor stores. |

Both kernels are exposed through a PyTorch extension so they can be called from
Python or from other chapters.

## Hardware / software requirements

* CUDA 13.0+ with driver ≥ 580.
* Compute capability 12.1 (GB10 / B100 / B200). The build script passes
  `-gencode=arch=compute_121,code=sm_121`.
* PyTorch 2.9 w/ CUDA 13 wheels (already part of this workspace).

## Layout

```
capstone/
 ├── __init__.py                  # convenience wrappers
 ├── README.md                    # this file
 ├── capstone_extension.py        # torch.utils.cpp_extension loader
 ├── capstone_kernels.cu          # baseline + optimized kernels + bindings
 └── run_capstone.py              # CLI benchmark + speedup demo
```

## Building the extension

The first invocation of any helper (CLI or Python) will JIT-compile
`capstone/capstone_kernels.cu` via `torch.utils.cpp_extension.load`.  Subsequent
runs reuse the cached `.so` under `~/.cache/torch_extensions/`.

The build steps (handled automatically) are equivalent to:

```bash
python - <<'PY'
from capstone import baseline_matmul
import torch
a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
baseline_matmul(a, a)
PY
```

## Benchmarking & reproducing the speedup

Run the CLI benchmark (defaults: 2048³ matmul, 3 optimized iterations, 1 baseline):

```bash
python capstone/run_capstone.py
```

Flags:

* `--size`: square GEMM dimension (must be divisible by 128).
* `--iters`: timing loops for the optimized kernel (defaults to 3).
* `--baseline-iters`: timing loops for the naive kernel (`1` by default).
* `--skip-baseline`: record only the optimized numbers.
* `--timeout`: wall-clock timeout (seconds) for each timing section.

The script also runs a correctness check on a 256² slice and reports TFLOP/s +
speedup.  On our GB10 devkit (CUDA 13.0) the default 2048³ benchmark produces:

```
Kernel              Avg ms    TFLOP/s
--------------------------------------
baseline_naive     28.31     0.61
optimized_cluster  17.98     0.96

Speedup (baseline/optimized): 1.57x
```

Larger matrices benefit more because the clustered TMA pipeline keeps tensor
cores fed while the naive kernel stalls on global memory.

> **Toolchain status (Nov 2025):** CUDA 13.0 refuses to assemble the new `tcgen05.*`
> instructions for SM 12.x targets (GB10) — ptxas errors out with “instruction not
> supported on .target sm_121”. That’s why the current kernel still uses scalar
> FMAs. Once NVIDIA enables tcgen05 on GB10 (or we run on SM 100 hardware such as
> B100/B200) we’ll drop in the real tensor-core path described below.

## Full-stack scenario suite

`capstone/scenario_benchmark.py` composes the chapter benchmarks into a single
BenchmarkHarness-driven storyline. Every scenario defines a list of phases (each
phase maps to the canonical `baseline_*` / `optimized_*` module inside the
chapter directories). The harness automatically enables smoke-test scale by
setting `BENCHMARK_SMOKE_TEST=1`, keeps NVTX off for baselines, and turns it on
for optimized runs. Results are logged phase-by-phase so you can see which
chapters executed (or were skipped due to hardware constraints).

Run any scenario directly:

```bash
python capstone/baseline_01_system_foundations.py
python capstone/optimized_01_system_foundations.py
```

Or via the unified CLI:

```bash
python tools/cli/benchmark_cli.py capstone --example baseline_01_system_foundations
python tools/cli/benchmark_cli.py capstone --example optimized_01_system_foundations
```

| Scenario | Baseline entry | Optimized entry | Chapters |
| --- | --- | --- | --- |
| 01. System foundations | `baseline_01_system_foundations.py` | `optimized_01_system_foundations.py` | ch1–ch3 |
| 02. Cluster parallelism | `baseline_02_cluster_parallelism.py` | `optimized_02_cluster_parallelism.py` | ch4 |
| 03. IO pipeline | `baseline_03_io_pipeline.py` | `optimized_03_io_pipeline.py` | ch5 |
| 04. Kernel optimization | `baseline_04_kernel_optimization.py` | `optimized_04_kernel_optimization.py` | ch6–ch10 |
| 05. Streams & CUDA Graphs | `baseline_05_streams_and_graphs.py` | `optimized_05_streams_and_graphs.py` | ch11–ch12 |
| 06. Compiler stack | `baseline_06_compiler_stack.py` | `optimized_06_compiler_stack.py` | ch13–ch14 |
| 07. Inference & attention | `baseline_07_inference_attention.py` | `optimized_07_inference_attention.py` | ch15–ch18 |
| 08. Low precision & memory | `baseline_08_low_precision.py` | `optimized_08_low_precision.py` | ch19 |
| 09. End-to-end production | `baseline_09_end_to_end.py` | `optimized_09_end_to_end.py` | ch20 |

Each scenario executes the referenced chapter benchmarks sequentially with the
shared harness, so you get comparable timing, validation, and profiling controls
without re-implementing the workloads.

## Optimization Roadmap & references

The kernel intentionally mirrors the staged optimization story from Modular’s
“Matrix Multiplication on NVIDIA Blackwell” series so readers can tie each code
change to the blog narrative.  Here’s how today’s implementation maps to the
worklog—and what’s queued up next:

| Stage | What we’ve implemented | Blog reference |
| --- | --- | --- |
| 1. Tiled shared memory + Tensor Memory Accelerator loads | Loops in `capstone/capstone_kernels.cu` move 128×64 tiles from GMEM→SMEM via `cp_async_bulk_tensor`, matching Part 2’s TMA introduction. | [Modular Part 2 – Using Hardware Features to Optimize Matmul](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul) |
| 2. Thread-block clusters + DSM multicast for B tiles | A single CTA in each column issues the B-tile TMA load, then `cluster.map_shared_rank` shares it with its partner CTA (the other row in the 2×2 cluster).  This is the “multicast” optimization in Part 3. | [Modular Part 3 – The Optimizations Behind 85% of SOTA Performance](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance) |
| 3. Warp-specialized pipeline (load / compute / epilogue roles) | Warps 0–1 issue TMA loads, warps 2–9 execute the matmul, and warps 10–11 drain outputs asynchronously, matching the producer/consumer layout in Part 3. | Part 3 (“2SM pipelining and warp specialization”) |
| 4. Async epilogue + TMA stores | After TMEM→register accumulation (currently scalar), output warps convert to FP16 and launch `cp_async.bulk.tensor` stores while the next tile’s loads begin, mirroring Part 3’s double-buffered epilogue. | Part 3 (“double-buffering the write-out”) |

### Why only the B tile uses DSM today

Part 3 shows DSM applied to both A and B, but the biggest win on GB10 came from
multicasting the B tile alone: both CTAs in a column consume the exact same B
panel every iteration, whereas each CTA needs a unique A panel.  Leaving A
private simplified synchronization and avoided the invalid DMA writes we hit
early on, while still halving GMEM traffic along the N dimension.  Once the rest
of the tensor-core path lands (see below), we can revisit A-sharing to squeeze
out the last few GB/s.

### Upcoming steps (in-flight)

1. **`tcgen05.mma` + TMEM accumulators (Part 3 Section “Tensor core matrices”)** –
   replace the per-thread FMA loop with tensor-core fragments, keep results in
   TMEM, and let compute/epilogue warps overlap without register↔SMEM bounces.  
   _Blocked on CUDA enabling tcgen05 for SM 12.x; works today on SM 100 (B100/B200)._
2. **2×SM MMA (`tcgen05.mma.cta_group::2`)** –
   have paired CTAs cooperate directly on the tensor cores, as shown in
   Part 3 Figure 6, so DSM no longer keeps duplicate B tiles and tensor-core
   throughput doubles.
3. **Persistent clusters + Cluster Launch Control (Part 4)** –
   adopt the tile scheduler + CLC pipeline from Part 4 so we feed new tiles
   without kernel relaunch gaps.
4. **Autotune shapes / pipeline depth (`kbench` in Part 4)** –
   hook up the same autotuning harness to choose MMA shapes, cluster dimensions,
   and pipeline depth per workload (e.g., MoE experts vs. attention blocks).

After tcgen05 lands we can also experiment with `cp.async.reduce` / `cp_reduce.async`
for cross-CTA epilogues (layernorm, gating stats) and SHARP-over-NVLink collectives,
but those only pay off once the kernel is tensor-core limited.

Each of those steps will be documented here with explicit references to the
matching section of the Modular posts as we land them, so Chapter “capstone/”
reads like a live coding companion to Parts 1–4.

### tcgen05 (SM100) preview build

Both tcgen05 paths live in `capstone/capstone_kernels_tcgen05.cu` and are loaded
by `capstone_extension_tcgen05.py`:

* **`optimized_matmul_tcgen05`** – CTA-group::1 pipeline (single-SM tcgen05)
  that mirrors Modular’s Part 3 kernel. It is the drop-in replacement for the
  `_non_tcgen05` helper once SM100 hardware/toolchain support the instructions.
* **`optimized_matmul_tcgen05_cta2`** – CTA-group::2 / 2SM variant that allocates
  TMEM cooperatively across a 2×1×1 cluster, maps the TMEM descriptors through
  DSMEM, and launches the `tcgen05.mma.cta_group::2` instructions. This is the
  “real-man” path asked for earlier and replaces the temporary CUTLASS preview.

Both functions are exported through `torch.ops.capstone_tcgen05.*` as well as the
high-level `capstone` Python package. The loader always tries to build them for
`sm_100`; if ptxas refuses (e.g., on GB10) the compile error is cached and the
benchmark scripts raise a `SKIPPED: …` message instead of hanging midway.

Harness coverage is limited to the capstone chapter for now—each other chapter
will grow its own `_non_tcgen05` / `_tcgen05` implementation, instead of wrapping
capstone logic, so their narratives stay self-contained.

For automated benchmarking, the harness sees
`baseline_cluster_gemm_tcgen05.py`, `optimized_cluster_gemm_tcgen05.py`, and
`optimized_cluster_gemm_tcgen05_cta2.py`; the first reuses the scalar baseline
while the latter two exercise the CTA-group::1 and CTA-group::2 tcgen05 paths,
respectively.

## How the optimized kernel works

See `capstone/capstone_kernels.cu` for the annotated implementation.  The high
level ingredients are:

1. **Thread-block clusters** (`__cluster_dims__(2,2,1)` + `cudaFuncSetAttribute`)
   keep four CTAs resident on neighboring SMs.  Cluster-level barriers
   (`cg::cluster_group`) let us synchronize the persistent schedule.
2. **Distributed shared memory** via `cluster.map_shared_rank` lets the top-row
   CTAs load each B tile once and expose it to the row beneath them over DSMEM.
   That halves the global-memory traffic for the column panels that dominate MoE
   expert matmuls.
3. **Tensor Memory Accelerator (TMA)** descriptors (created with
   `cuTensorMapEncodeTiled`) feed `cuda::device::experimental::cp_async_bulk_tensor_*`
   to multicast 128×64 tiles from GMEM→SMEM asynchronously.  Dual barriers keep
   a 3-stage circular buffer alive while compute warps chew through the previous tile.
4. **Warp specialization**: two warps dedicate themselves to TMA loads, eight
   warps run the 128×128 MMA on tensor-memory backed fragments, and two warps
   handle TMEM→SMEM→GMEM stores.
5. **Shared-memory accumulator ping-pong**: the FP32 accumulators live in
   staged shared-memory tiles so we can overlap the next set of TMA loads with
   the epilogue.
6. **Cluster-cooperative epilogue**: the store warps convert the FP32 accumulators
   into FP16 tiles, fence them, then launch `cp_async_bulk_tensor_2d_shared_to_global`
   writes so the entire C tile drains with a single TMA transaction.

These choices track the Part 3 + Part 4 optimizations from Modular’s
“Matrix Multiplication on NVIDIA Blackwell” series, but the code is native
PyTorch/CUDA C++ instead of Mojo.

## Using the kernels from Python

```python
import torch
from capstone import baseline_matmul, optimized_matmul

a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)

ref = baseline_matmul(a, b)
fast = optimized_matmul(a, b)
print((ref - fast).abs().max())
```

Remember that the optimized kernel assumes dimensions divisible by 128/64.
The CLI already validates this.
