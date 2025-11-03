# Pipeline-Parallel Inference Playbook

This walkthrough demonstrates how to bring the NVSHMEM-backed pipeline
implementation in `ch4/nvshmem_pipeline_parallel.py` online for production
inference. The same routines power the microbatch handoff observed in the
multi-GPU load tests.

## Prerequisites
- 8× NVIDIA B200 GPUs (NVLink 5.0 / NVSwitch topology).
- CUDA 13.0+, PyTorch 2.9+ with `torch.distributed.nn.SymmetricMemory`.
- NVSHMEM user-space components installed (`libnvshmem.so` visible inside the
  runtime container).
- Environment configured as in `docs/8xb200_load_testing_guide.md`
  (`NCCL_P2P_LEVEL=NVL`, NVLink mesh verified).

## 1F1B Schedule (Baseline)
The one-forward-one-backward schedule is the canonical configuration. Run it
with:

```bash
torchrun --nproc_per_node=8 ch4/nvshmem_pipeline_parallel.py --schedule 1f1b
```

Rank 0 prints the completion summary:

```
[1f1b] Completed 16 microbatches in 2.41s
[1f1b] Average loss: 0.5012
[1f1b] Symmetric memory: True
```

Key takeaways:
- NVSHMEM double-buffered activation transfers keep bubble time ≈10%.
- Latency for a single microbatch handoff is <5 µs (Nsight timelines show the
  NVLink copy overlapping compute on consecutive stages).
- Use this schedule for balanced latency/throughput without excess memory use.

## Interleaved Virtual Stages
When latency budget allows for higher VRAM usage, interleaving stages reduces
pipeline bubbles further by placing two virtual stages on each GPU.

```bash
torchrun --nproc_per_node=8 ch4/nvshmem_pipeline_parallel.py --schedule interleaved
```

Sample output:

```
[interleaved] Completed 16 microbatches in 1.86s
[interleaved] Virtual stages per rank: 2
[interleaved] Symmetric memory: True
```

Operational notes:
- Bubble time drops below 5 %; throughput improves 1.2–1.3× vs. 1F1B.
- Expect ~18 % higher activation memory (two stage buffers resident per GPU).
- This configuration matches the 14,960 tokens/sec MoE load test results.

## Combined Drill
Use `--schedule all` to run both schedules back-to-back—handy for benchmarking
after kernel or NCCL upgrades:

```bash
torchrun --nproc_per_node=8 ch4/nvshmem_pipeline_parallel.py --schedule all
```

## Instrumentation Hooks
- **NVTX ranges**: the script emits ranges for stage warmup, steady-state, and
  cooldown. Capture with `nsys profile --duration=60 --delay=10 …` to confirm
  overlap behaviour on the production topology.
- **Metrics**: each stage logs microbatch timings to stdout. Pipe the output
  into your observability stack or adapt the helper functions inside the file
  (`NVSHMEMPipelineEngine.run_1f1b_schedule`) to emit Prometheus counters.
- **Correctness guard**: run with `NCCL_P2P_DISABLE=1` once during staging—the
  fallback path (explicit `dist.send/recv`) ensures correctness even if
  SymmetricMemory is unavailable, albeit with ~10× higher latency.

## Integrating With Serving
1. Import the engine:
   ```python
   from ch4.nvshmem_pipeline_parallel import NVSHMEMPipelineEngine
   ```
2. Instantiate one engine per real or virtual stage; wire them into the
   request loop that currently calls tensor-parallel collectives.
3. Reuse the activation buffer design from the script (double-buffered,
   symmetric memory aware) so NVLink transfers overlap compute.
4. Verify combined throughput with the load-testing harness:
   ```bash
   ./tools/orchestrate_8xb200_load_test.sh 3600 180 nvshmem_pipeline_validation
   ```
   Expect ≥14,000 tokens/sec with the interleaved pipeline and power efficiency
   around 8.6 tokens/joule.

With 1F1B and interleaved schedules documented, the production playbook now
covers both the fallback single-GPU path and the high-throughput NVLink mesh
deployment.
