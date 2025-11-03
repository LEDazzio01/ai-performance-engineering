# Nsight Trace Capture: FP8 + FlexAttention (Large Models)

Use this runbook to generate Nsight Systems and Nsight Compute traces for the
40B-class FlexAttention + FP8 workload on Blackwell GPUs. The commands assume
an 8×B200 node with the validated NVLink mesh and the FlexAttention vmap fix.

## Directory Layout
Traces and reports are stored under `artifacts/nsight/fp8_flexattention/`
with the following structure:

```
artifacts/nsight/fp8_flexattention/
├── nsys/
│   ├── flexattention_fp8.qdrep
│   └── flexattention_fp8.sqlite
└── ncu/
    └── flexattention_fp8.ncu-rep
```

The helper script `scripts/capture_fp8_flexattention_traces.sh` creates the
directory tree automatically.

## 1. Warmup (Optional but recommended)
Run the optimized benchmark once so compile-time overhead does not bleed into
the Nsight capture.

```bash
python ch16/test_gpt_large_optimized.py \
    --tensor-parallel-gpus 8 \
    --attention-backend flex \
    --fp8-mode transformer-engine \
    --warmup 2 \
    --iters 4 \
    --output-json artifacts/nsight/fp8_flexattention/warmup.json
```

## 2. Nsight Systems
Profiles timeline-level interactions (CUDA kernel launches, NCCL calls, CPU
scheduling, NVLink throughput).

```bash
nsys profile --force-overwrite true \
  --output=artifacts/nsight/fp8_flexattention/nsys/flexattention_fp8 \
  --sample=none \
  --trace=cuda,nvtx,mpi \
  --cuda-memory-usage=true \
  bash -lc "
    PYTHONPATH=$PWD \
    torchrun --nproc_per_node=8 \
      ch16/inference_server_load_test.py \
      --duration 60 \
      --target-qps 28 \
      --prompt-len-min 512 \
      --prompt-len-max 1024 \
      --max-new-tokens 512 \
      --attention-backend flex \
      --fp8-mode transformer-engine"
```

- `--trace=cuda,nvtx,mpi` captures kernel launches, NVTX ranges emitted by the
  benchmark, and NCCL collectives.
- `--sample=none` avoids statistical sampling so NVLink bandwidth timelines are
  uncompressed.

## 3. Nsight Compute
Collects per-kernel roofline metrics, tensor-core utilization, and memory
traffic. Focus on FlexAttention kernels (`FusedDecodeAttn*`).

```bash
ncu --set full \
  --target-processes all \
  --force-overwrite \
  --kernel-name-base demangled \
  --launch-skip 10 \
  --launch-count 2 \
  --section Sleep,LaunchStats,MemoryWorkloadAnalysis,RooflineChart \
  --export artifacts/nsight/fp8_flexattention/ncu/flexattention_fp8 \
  bash -lc "
    PYTHONPATH=$PWD \
    torchrun --nproc_per_node=8 \
      ch16/inference_server_load_test.py \
      --duration 45 \
      --target-qps 28 \
      --prompt-len-min 512 \
      --prompt-len-max 1024 \
      --max-new-tokens 512 \
      --attention-backend flex \
      --fp8-mode transformer-engine \
      --skip-power-monitor"
```

- Skip the first 10 launches to avoid cold-start noise.
- Use `--skip-power-monitor` to reduce interference while Nsight Compute pauses
  the GPU between kernel captures.

## 4. Automated Script
The repository ships with `scripts/capture_fp8_flexattention_traces.sh`, which
wraps the commands above and enforces artifact layout. Example:

```bash
./scripts/capture_fp8_flexattention_traces.sh
```

Override defaults by exporting environment variables before invoking the
script:

```bash
export TRACE_DURATION=90
export TARGET_QPS=30
./scripts/capture_fp8_flexattention_traces.sh
```

## 5. Post-Processing
- Import the `.qdrep` file into Nsight Systems GUI to inspect cross-GPU NVLink
  utilization and verify kernels overlap with communication steps.
- Use `ncu --import` on the `.ncu-rep` artifact to generate CSV summaries:
  ```bash
  ncu --import artifacts/nsight/fp8_flexattention/ncu/flexattention_fp8.ncu-rep \
      --csv \
      --page raw \
      --output artifacts/nsight/fp8_flexattention/ncu/flexattention_fp8.csv
  ```
- Attach the CSV and Nsight reports to the performance baseline dashboard.

Following this workflow ensures every release of the FP8 + FlexAttention path
ships with reproducible traces that back the published efficiency numbers.
