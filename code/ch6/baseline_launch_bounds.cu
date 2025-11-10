// baseline_launch_bounds.cu -- kernel without launch bounds (baseline).

#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(status));                                     \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

constexpr int kLaunchBoundsWorkIters = 96;
constexpr int kLaunchBoundsTransformRepeats = 3;
constexpr int kTransformPasses = 16;  // matches optimized version, but staged here
constexpr int kBaselineThreads = 64;
constexpr int kBaselineBlocks = 8;
constexpr int kBaselineChunkElements = kBaselineThreads * kBaselineBlocks;
constexpr float kLaunchBoundsEps = 1e-6f;

__device__ __forceinline__ float launch_bounds_workload(float value) {
    float acc0 = value * 1.0001f + 0.1f;
    float acc1 = value * 0.9997f - 0.05f;

    #pragma unroll
    for (int repeat = 0; repeat < kLaunchBoundsTransformRepeats; ++repeat) {
        #pragma unroll 4
        for (int iter = 0; iter < kLaunchBoundsWorkIters; ++iter) {
            const float coupled = (acc0 * acc1) * 0.00025f + (iter + 1 + repeat) * kLaunchBoundsEps;
            const float inv = rsqrtf(fabsf(acc0) + fabsf(acc1) + coupled + kLaunchBoundsEps);
            acc0 = fmaf(acc0, 1.00003f, inv * 0.0002f + coupled);
            acc1 = fmaf(acc1, 0.99991f, -inv * 0.00015f - coupled * 0.5f);
        }
        float mix = acc0 * 0.125f + acc1 * 0.875f;
        acc0 = mix * 1.00001f + acc1 * 0.0001f;
        acc1 = mix * 0.75f - acc0 * 0.00005f;
    }
    return acc0 + acc1;
}

// Kernel without launch bounds for comparison (baseline)
__global__ void myKernelNoLB(float* input, float* output, int N) {
    extern __shared__ float staging[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float value = input[idx];
#pragma unroll 1
        for (int pass = 0; pass < kTransformPasses; ++pass) {
            staging[threadIdx.x] = value;
            __syncthreads();
            value = launch_bounds_workload(staging[threadIdx.x]);
            __syncthreads();
        }
        output[idx] = value;
    }
}

int main() {
    const int N = 1024 * 1024;
    
    float *h_input, *h_output;
    CUDA_CHECK(cudaMallocHost(&h_input, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output, N * sizeof(float)));
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = float(i % 1000) / 1000.0f;
    }
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_input, kBaselineChunkElements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_output, kBaselineChunkElements * sizeof(float), stream));
    
    // Launch parameters
    const int threads = kBaselineThreads;
    const int blocks = kBaselineBlocks;
    const int chunk_capacity = kBaselineChunkElements;
    const int total_launches = (N + chunk_capacity - 1) / chunk_capacity;
    
    printf("========================================\n");
    printf("Launch Bounds (Baseline)\n");
    printf("========================================\n");
    printf("Problem size: %d elements\n", N);
    printf("Threads per block: %d\n", threads);
    printf("Blocks per launch: %d (micro-grid)\n", blocks);
    printf("Transform passes per element (staged reuse): %d\n", kTransformPasses);
    printf("Chunk capacity: %d elements (launches required: %d)\n\n", chunk_capacity, total_launches);
    
    // Time kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    cudaEventRecord(start, stream);
    int processed = 0;
    while (processed < N) {
        int chunk_elems = std::min(chunk_capacity, N - processed);
        size_t chunk_bytes = static_cast<size_t>(chunk_elems) * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_input,
                                   h_input + processed,
                                   chunk_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, chunk_bytes, stream));
        myKernelNoLB<<<blocks, threads, threads * sizeof(float), stream>>>(
            d_input,
            d_output,
            chunk_elems);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_output + processed,
                                   d_output,
                                   chunk_bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        processed += chunk_elems;
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Kernel without launch bounds: %.2f ms\n", ms);
    printf("First result: %.3f\n", h_output[0]);
    
    // Cleanup
    CUDA_CHECK(cudaFreeAsync(d_input, stream));
    CUDA_CHECK(cudaFreeAsync(d_output, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    
    return 0;
}
