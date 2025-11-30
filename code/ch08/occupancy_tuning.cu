// occupancy_tuning.cu -- occupancy tuning microbenchmark with tunable launch config and unrolling.

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

constexpr int N = 1 << 21;  // Moderate workload to surface occupancy effects without long runs.

__global__ __launch_bounds__(256, 4)
void kernel(const float* in, float* out, int n, int smem_bytes, int unroll, int inner_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ unsigned char smem[];
  // Touch shared memory when requested so occupancy accounts for it.
  if (smem_bytes > 0 && threadIdx.x == 0) {
    smem[0] = 0;
  }
  int stride = blockDim.x * gridDim.x;
  for (int base = idx; base < n; base += stride * unroll) {
    #pragma unroll
    for (int u = 0; u < 8; ++u) {  // cap unroll loop; runtime arg may be smaller
      if (u >= unroll) break;
      int i = base + u * stride;
      if (i < n) {
        float val = in[i];
        // Independent accumulators to surface ILP vs occupancy trade-offs.
        float a = val * 1.1f;
        float b = val * 0.9f;
        float c = val * 1.3f;
        float d = val * 0.7f;
        int iters = inner_iters;
        #pragma unroll 16
        for (int it = 0; it < 16; ++it) {
          if (it >= iters) break;
          a = fmaf(a, 1.0001f, b);
          b = fmaf(b, 0.9997f, c);
          c = fmaf(c, 1.0003f, d);
          d = fmaf(d, 0.9999f, a);
        }
        out[i] = a + b + c + d;
      }
    }
  }
}

int main(int argc, char** argv) {
  int block_size = 128;  // intentionally modest baseline
  int smem_bytes = 0;
  int unroll = 1;
  int inner_iters = 2;
  int reps = 50;  // number of kernel launches to amortize overhead
  // Basic flag parsing.
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
      block_size = std::max(1, atoi(argv[++i]));
    } else if (strcmp(argv[i], "--smem-bytes") == 0 && i + 1 < argc) {
      smem_bytes = std::max(0, atoi(argv[++i]));
    } else if (strcmp(argv[i], "--unroll") == 0 && i + 1 < argc) {
      unroll = std::max(1, atoi(argv[++i]));
    } else if (strcmp(argv[i], "--inner-iters") == 0 && i + 1 < argc) {
      inner_iters = std::max(1, atoi(argv[++i]));
    } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = std::max(1, atoi(argv[++i]));
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [--block-size N] [--smem-bytes N] [--unroll N] [--inner-iters N] [--reps N]\n", argv[0]);
      return 0;
    }
  }
  if (block_size <= 0 || block_size > 1024) {
    fprintf(stderr, "Invalid block size %d (must be 1..1024)\n", block_size);
    return 1;
  }
  unroll = std::min(unroll, 8);
  inner_iters = std::min(inner_iters, 16);

  float *h_in, *h_out;
  cudaMallocHost(&h_in, N * sizeof(float));
  cudaMallocHost(&h_out, N * sizeof(float));
  for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  cudaMalloc(&d_in, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));
  cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(block_size);
  // Grid-stride accounts for unroll
  dim3 grid((N + block.x * unroll - 1) / (block.x * unroll));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int r = 0; r < reps; ++r) {
    kernel<<<grid, block, smem_bytes>>>(d_in, d_out, N, smem_bytes, unroll, inner_iters);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  float avg_kernel_ms = total_ms / static_cast<float>(reps);

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf(
      "out[0]=%.1f (block=%d, smem=%d bytes, unroll=%d, inner_iters=%d, reps=%d) avg_kernel_ms=%.6f total_ms=%.3f\n",
      h_out[0],
      block_size,
      smem_bytes,
      unroll,
      inner_iters,
      reps,
      avg_kernel_ms,
      total_ms);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
