# Benchmark Baseline – 2025-10-28

Date: October 28, 2025  
Command: `python3 benchmark_peak.py`  
Output JSON: `BENCHMARK_PEAK_RESULTS_20251028_062057.json`

## Summary
- **HBM3e bandwidth:** 2.73 TB/s (35.0 % of 7.8 TB/s theoretical); status `GOOD`
- **FP16 compute:** 1291 TFLOPS peak (matrix 8192×8192); status `PASS`
- **torch.compile speedup:** 1.52× versus eager; status `PASS`

## Detailed Metrics
### HBM3e Bandwidth
| Transfer Size | Bandwidth (TB/s) | Utilization (%) |
|---------------|------------------|-----------------|
| 8 GB          | 1.82             | 23.39           |
| 12 GB         | **2.73**         | **35.01**       |
| 16 GB         | 2.68             | 34.35           |

### FP16 Tensor Core Compute
| Matrix (M=N=K) | Time (ms) | TFLOPS |
|----------------|-----------|--------|
| 4096           | 0.134     | 1023   |
| 8192           | 0.851     | **1291** |

### torch.compile Configuration
- Eager latency: 1.528 ms  
- Compiled latency: 1.006 ms  
- Speedup: **1.52×**

## Notes
- HBM3e measurements plateau well below the 7.0 TB/s target; investigate vectorized copy kernels or larger batched transfers if higher utilization is required.
- FP16 compute exceeds the 1000 TFLOPS goal; keep matrix sizes as multiples of 128 to maintain tensor core efficiency.
- torch.compile configuration (cudagraphs + warmup) is validated; use this setup when evaluating further optimizations.

