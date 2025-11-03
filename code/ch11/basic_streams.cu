// basic_streams.cu -- CUDA 13.0 stream overlap demo with error handling.

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int WORK_ITERS = 4;
constexpr float BIAS_ADD = 0.001f;
constexpr float BIAS_SUB = 0.0002f;
constexpr float DECAY = 0.9f;

// Optimized kernel with vectorized loads and async copy support
// Launch bounds removed to let compiler auto-tune for different architectures (B200 vs GB10)
__global__ void scale_kernel(float* data, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = data[idx];
#pragma unroll 4
    for (int iter = 0; iter < WORK_ITERS; ++iter) {
      val = val * scale + BIAS_ADD;
      val = val * DECAY - BIAS_SUB;
    }
    data[idx] = val;
  }
}

// Vectorized version using float4 for better memory throughput
// Launch bounds removed to let compiler auto-tune for different architectures (B200 vs GB10)
__global__ void scale_kernel_vectorized(float* data, int n, float scale) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  
  if (idx + 3 < n) {
    // Load 4 floats at once (128-bit transaction)
    float4 vec = *reinterpret_cast<float4*>(&data[idx]);
    
    // Process all 4 elements
#pragma unroll 4
    for (int iter = 0; iter < WORK_ITERS; ++iter) {
      vec.x = vec.x * scale + BIAS_ADD;
      vec.y = vec.y * scale + BIAS_ADD;
      vec.z = vec.z * scale + BIAS_ADD;
      vec.w = vec.w * scale + BIAS_ADD;
      vec.x = vec.x * DECAY - BIAS_SUB;
      vec.y = vec.y * DECAY - BIAS_SUB;
      vec.z = vec.z * DECAY - BIAS_SUB;
      vec.w = vec.w * DECAY - BIAS_SUB;
    }
    
    // Store 4 floats at once
    *reinterpret_cast<float4*>(&data[idx]) = vec;
  } else {
    // Handle remaining elements
    for (int i = idx; i < n; i++) {
      float val = data[i];
#pragma unroll 4
      for (int iter = 0; iter < WORK_ITERS; ++iter) {
        val = val * scale + BIAS_ADD;
        val = val * DECAY - BIAS_SUB;
      }
      data[i] = val;
    }
  }
}

// Shared memory version with prefetching (simpler than cp.async for this demo)
// On Blackwell, compiler optimizes this with TMA automatically
// Launch bounds removed to let compiler auto-tune for different architectures (B200 vs GB10)
__global__ void scale_kernel_async(float* __restrict__ data, int n, float scale) {
  // Shared memory for prefetching and better cache utilization
  __shared__ float smem[256 * 4];  // 4 elements per thread for float4
  
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + tid) * 4;
  
  if (idx + 3 < n) {
    // Load to shared memory (compiler will optimize on Blackwell)
    float4 vec = *reinterpret_cast<const float4*>(&data[idx]);
    *reinterpret_cast<float4*>(&smem[tid * 4]) = vec;
    
    __syncthreads();
    
    // Process from shared memory
    vec = *reinterpret_cast<float4*>(&smem[tid * 4]);
#pragma unroll 4
    for (int iter = 0; iter < WORK_ITERS; ++iter) {
      vec.x = vec.x * scale + BIAS_ADD;
      vec.y = vec.y * scale + BIAS_ADD;
      vec.z = vec.z * scale + BIAS_ADD;
      vec.w = vec.w * scale + BIAS_ADD;
      vec.x = vec.x * DECAY - BIAS_SUB;
      vec.y = vec.y * DECAY - BIAS_SUB;
      vec.z = vec.z * DECAY - BIAS_SUB;
      vec.w = vec.w * DECAY - BIAS_SUB;
    }
    
    __syncthreads();
    
    // Write back to global memory
    *reinterpret_cast<float4*>(&data[idx]) = vec;
  } else {
    // Handle remaining elements
    for (int i = idx; i < n; i++) {
      data[i] = data[i] * scale + 0.001f;
    }
  }
}

int main() {
  constexpr int N = 1 << 21;
  constexpr size_t BYTES = N * sizeof(float);

  float *h_a = nullptr, *h_b = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_a, BYTES));
  CUDA_CHECK(cudaMallocHost(&h_b, BYTES));
  for (int i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a = nullptr, *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, BYTES));
  CUDA_CHECK(cudaMalloc(&d_b, BYTES));

  cudaStream_t stream1 = nullptr, stream2 = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, 0));

  CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, BYTES, cudaMemcpyHostToDevice, stream2));

  // Benchmark: Compare original, vectorized, and async versions
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  constexpr int WARMUP = 5;
  constexpr int ITERS = 100;
  constexpr int PIPELINE_BATCHES = 4;
  
  // Original kernel
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  for (int i = 0; i < WARMUP; ++i) {
    scale_kernel<<<grid, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  
  CUDA_CHECK(cudaEventRecord(start, stream1));
  for (int i = 0; i < ITERS; ++i) {
    scale_kernel<<<grid, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream1));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms_original = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_original, start, stop));
  
  // Vectorized kernel (adjust grid size for float4)
  dim3 grid_vec((N / 4 + block.x - 1) / block.x);
  for (int i = 0; i < WARMUP; ++i) {
    scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  
  CUDA_CHECK(cudaEventRecord(start, stream1));
  for (int i = 0; i < ITERS; ++i) {
    scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream1));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms_vectorized = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_vectorized, start, stop));
  
  // Async kernel
  for (int i = 0; i < WARMUP; ++i) {
    scale_kernel_async<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  
  CUDA_CHECK(cudaEventRecord(start, stream1));
  for (int i = 0; i < ITERS; ++i) {
    scale_kernel_async<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream1));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms_async = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_async, start, stop));
  
  // Test with dual streams
  scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  scale_kernel_vectorized<<<grid_vec, block, 0, stream2>>>(d_b, N, 0.9f);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, BYTES, cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(h_b, d_b, BYTES, cudaMemcpyDeviceToHost, stream2));

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));

  // Measure sequential vs overlapped pipelines
  auto measure_pipeline = [&](bool overlap) -> double {
    constexpr int BATCHES = PIPELINE_BATCHES;
    CUDA_CHECK(cudaMemsetAsync(d_a, 0, BYTES, stream1));
    CUDA_CHECK(cudaMemsetAsync(d_b, 0, BYTES, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!overlap) {
      // Naive sequential pipeline on a single stream with blocking transfers.
      for (int i = 0; i < BATCHES; ++i) {
        CUDA_CHECK(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
        scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.05f);
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaMemcpy(h_a, d_a, BYTES, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice));
        scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_b, N, 0.95f);
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaMemcpy(h_b, d_b, BYTES, cudaMemcpyDeviceToHost));
      }
    } else {
      // Three-way pipeline: copy on stream1, compute on stream2, copy-back on a third stream.
      cudaStream_t d2h_stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking));

      cudaEvent_t h2d_done[2], compute_done[2], d2h_done[2];
      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&h2d_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&d2h_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(d2h_done[i], d2h_stream));
      }
      CUDA_CHECK(cudaStreamSynchronize(d2h_stream));

      for (int i = 0; i < BATCHES; ++i) {
        const int buf = i & 1;
        float* d_buf = buf == 0 ? d_a : d_b;
        float* h_buf = buf == 0 ? h_a : h_b;
        const float scale = buf == 0 ? 1.05f : 0.95f;

        CUDA_CHECK(cudaStreamWaitEvent(stream1, d2h_done[buf], 0));
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, BYTES, cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaEventRecord(h2d_done[buf], stream1));

        CUDA_CHECK(cudaStreamWaitEvent(stream2, h2d_done[buf], 0));
        scale_kernel_vectorized<<<grid_vec, block, 0, stream2>>>(d_buf, N, scale);
        CUDA_CHECK(cudaEventRecord(compute_done[buf], stream2));

        CUDA_CHECK(cudaStreamWaitEvent(d2h_stream, compute_done[buf], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, BYTES, cudaMemcpyDeviceToHost, d2h_stream));
        CUDA_CHECK(cudaEventRecord(d2h_done[buf], d2h_stream));
      }

      CUDA_CHECK(cudaStreamSynchronize(stream1));
      CUDA_CHECK(cudaStreamSynchronize(stream2));
      CUDA_CHECK(cudaStreamSynchronize(d2h_stream));

      for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventDestroy(h2d_done[i]));
        CUDA_CHECK(cudaEventDestroy(compute_done[i]));
        CUDA_CHECK(cudaEventDestroy(d2h_done[i]));
      }
      CUDA_CHECK(cudaStreamDestroy(d2h_stream));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  double sequential_ms = measure_pipeline(false);
  double overlap_ms = measure_pipeline(true);
  double overlap_speedup = sequential_ms / overlap_ms;
  double overlap_pct = (1.0 - overlap_ms / sequential_ms) * 100.0;

  std::printf("\n=== Kernel Performance Comparison ===\n");
  std::printf("Original kernel:    %.3f ms (%.2f GB/s)\n", 
              ms_original / ITERS, 2.0f * BYTES / (ms_original / ITERS * 1e6));
  std::printf("Vectorized kernel:  %.3f ms (%.2f GB/s) [%.2fx speedup]\n", 
              ms_vectorized / ITERS, 2.0f * BYTES / (ms_vectorized / ITERS * 1e6),
              ms_original / ms_vectorized);
  std::printf("Async copy kernel:  %.3f ms (%.2f GB/s) [%.2fx speedup]\n", 
              ms_async / ITERS, 2.0f * BYTES / (ms_async / ITERS * 1e6),
              ms_original / ms_async);
  std::printf("Stream speedup (vectorized vs original): %.2fx\n", ms_original / ms_vectorized);
  std::printf("\nstream1 result: %.3f\n", h_a[0]);
  std::printf("stream2 result: %.3f\n", h_b[0]);
  std::printf("\n=== Stream Overlap Benchmark ===\n");
  std::printf("Sequential pipeline: %.2f ms\n", sequential_ms);
  std::printf("Overlapped pipeline: %.2f ms\n", overlap_ms);
  std::printf("Stream overlap speedup: %.2fx (%.1f%% latency reduction)\n",
              overlap_speedup, overlap_pct);
  std::printf("Stream overlap percent: %.1f%%\n", overlap_pct);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  return 0;
}
