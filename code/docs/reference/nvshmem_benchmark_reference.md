# NVSHMEM vs NCCL Benchmark Reference

Use `torchrun --nproc_per_node=8 ch4/nvshmem_vs_nccl_benchmark.py` to
collect latency/bandwidth pairs for both NCCL and symmetric memory paths.
The script writes tabulated output to stdout; redirect it to JSON/markdown
for tracking if desired.

## Suggested Workflow

1. Warm up NCCL/NVSHMEM fabric:
   ```bash
   torchrun --nproc_per_node=8 ch4/nvshmem_vs_nccl_benchmark.py --steps 1 --iterations 10
   ```
2. Run the full sweep and capture logs:
   ```bash
   torchrun --nproc_per_node=8 ch4/nvshmem_vs_nccl_benchmark.py \
     --min-bytes 1024 --max-bytes 67108864 --steps 8 --iterations 100 \
     | tee profiles/nvshmem_vs_nccl_$(date +%Y%m%d_%H%M%S).log
   ```
3. Record summary metrics (latency μs / bandwidth GB/s) and archive the
   log alongside the hardware manifest in `profiles/`.

| Message Size | NCCL Latency (μs) | NCCL BW (GB/s) | NVSHMEM Latency (μs) | NVSHMEM BW (GB/s) |
|--------------|-------------------|----------------|-----------------------|-------------------|
| 4 KB         | _fill after run_  | _fill_         | _fill_                | _fill_            |
| 1 MB         | _fill after run_  | _fill_         | _fill_                | _fill_            |
| 64 MB        | _fill after run_  | _fill_         | _fill_                | _fill_            |

Update the table with real measurements from the latest Blackwell run to
maintain the 100/100 coverage score.
