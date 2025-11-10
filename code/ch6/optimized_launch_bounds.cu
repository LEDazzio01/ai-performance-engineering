// optimized_launch_bounds.cu -- kernel with launch bounds annotation (optimized).

#include <cuda_runtime.h>
#include <math.h>
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
constexpr int kTransformPasses = 16;  // number of chained passes, matches baseline
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

// Kernel with launch bounds annotation (optimized)
__global__ __launch_bounds__(256, 4)
void myKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int offset = idx; offset < N; offset += stride) {
        float value = input[offset];
#pragma unroll 1
        for (int pass = 0; pass < kTransformPasses; ++pass) {
            value = launch_bounds_workload(value);
        }
        output[offset] = value;
    }
}

int main() {
    const int N = 1024 * 1024;
    
    float *h_input;
    float h_first = 0.0f;
    CUDA_CHECK(cudaMallocHost(&h_input, N * sizeof(float)));
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = float(i % 1000) / 1000.0f;
    }
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_input, N * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_output, N * sizeof(float), stream));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    
    // Launch parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Time kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    cudaEventRecord(start, stream);
    myKernel<<<blocks, threads, 0, stream>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Copy a single element back to host for verification (device stays resident)
    CUDA_CHECK(cudaMemcpyAsync(&h_first, d_output, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("Kernel with launch bounds: %.2f ms\n", ms);
    printf("First result: %.3f\n", h_first);
    
    // Cleanup
    CUDA_CHECK(cudaFreeAsync(d_input, stream));
    CUDA_CHECK(cudaFreeAsync(d_output, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFreeHost(h_input);
    
    return 0;
}
