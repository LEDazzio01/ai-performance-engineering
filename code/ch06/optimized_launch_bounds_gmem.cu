// optimized_launch_bounds_gmem.cu -- launch bounds + dummy gmem write to build on GB10.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

constexpr int kWorkIters = 96;
constexpr int kTransformPasses = 8;
constexpr int kThreads = 128;
constexpr int kBlocks = 8;
constexpr int kChunkElements = kThreads * kBlocks;
__device__ float gmem_sink_opt;

__device__ __forceinline__ float workload(float v) {
    #pragma unroll 4
    for (int i = 0; i < kWorkIters; ++i) {
        v = fmaf(v, 1.0003f, 0.0002f * (i + 1));
        v = fmaf(v, 0.9998f, -0.00015f * (i + 1));
    }
    return v;
}

// Kernel with launch bounds annotation (optimized) and explicit gmem write
__global__ __launch_bounds__(kThreads, 4)
void kernel_with_lb(float* input, float* output, int n) {
    extern __shared__ float staging[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        #pragma unroll 1
        for (int pass = 0; pass < kTransformPasses; ++pass) {
            staging[threadIdx.x] = v;
            __syncthreads();
            v = workload(staging[threadIdx.x]);
            __syncthreads();
        }
        output[idx] = v;
        gmem_sink_opt = v;  // force global write so ptxas emits gmem
    }
}

int main() {
    const int N = 1024 * 64;
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
    for (int i = 0; i < N; ++i) h_in[i] = float(i % 113) * 0.33f;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, kChunkElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, kChunkElements * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);
    int processed = 0;
    while (processed < N) {
        const int chunk = std::min(kChunkElements, N - processed);
        const size_t bytes = size_t(chunk) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_in, h_in + processed, bytes, cudaMemcpyHostToDevice));
        kernel_with_lb<<<kBlocks, kThreads, kThreads * sizeof(float)>>>(d_in, d_out, chunk);
        CUDA_CHECK(cudaMemcpy(h_out + processed, d_out, bytes, cudaMemcpyDeviceToHost));
        processed += chunk;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("Launch-bounds optimized (gmem forcing) time: %.3f ms\\n", ms);
    std::printf("First output: %.4f\\n", h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
