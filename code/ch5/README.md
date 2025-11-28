# Chapter 5 - Storage and IO Optimization

## Summary
Focuses on feeding GPUs efficiently: tune DataLoader workers, vectorize preprocessing, overlap IO with compute, and adopt GPUDirect Storage when NVMe traffic becomes the bottleneck.

## Learning Goals
- Detect IO stalls via harness metrics and restructure pipelines to keep GPUs busy.
- Tune PyTorch DataLoader knobs (workers, prefetch, pinned memory) for large-batch training.
- Evaluate GPUDirect Storage paths vs traditional CPU-mediated reads.
- Benchmark remote storage and distributed data reading strategies.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_storage_cpu.py`, `optimized_storage_cpu.py` | Single-node dataloader comparison covering worker count, pinned memory, and caching strategies. |
| `baseline_vectorization.py`, `optimized_vectorization.py` | Vectorized parsing and memory-map examples that remove Python loops from preprocessing. |
| `baseline_ai.py`, `optimized_ai.py`, `storage_io_optimization.py` | LLM-style token pipelines showcasing overlapping compute with streaming reads and prefetch. |
| `baseline_distributed.py`, `optimized_distributed.py` | Multi-node reader pair demonstrating sharded datasets and rendezvous barriers. |
| `gds_cufile_minimal.py`, `gpudirect_storage_example.py` | GPUDirect Storage samples for verifying cuFile setup, buffer alignment, and throughput. |
| `compare.py`, `requirements.txt`, `expectations_gb10.json` | Harness entrypoint plus expectation baselines for spotting regressions. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch5
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch5
python tools/cli/aisp bench run --targets ch5 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python baseline_storage_cpu.py --inspect` exposes CPU wait time > GPU time; `optimized_storage_cpu.py` reverses the ratio with >=80% GPU utilization.
- `python gds_cufile_minimal.py --bytes 1073741824` sustains multi-GB/s throughput when `/etc/cufile.json` is configured and NVMe advertises GPUDirect support.
- `python compare.py --examples ai` shows optimized_ai eliminating CPU-side preprocessing from the critical path.

## Notes
- GPUDirect scripts fall back to host-mediated reads when `libcufile.so` is unavailable, making it safe to run on dev laptops.
- `requirements.txt` captures the limited extra deps (like `lmdb`) needed for the dataset shims.
