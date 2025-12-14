// baseline_dynamic_parallelism_device.cu
// Device-initiated launches: baseline uses many mid-sized child launches and heavier math.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include <vector>

#include "../core/common/headers/cuda_verify.cuh"

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

__global__ void childKernel(float* data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float v = data[idx];
        // Moderate math applied in both baseline and optimized variants
        #pragma unroll 4
        for (int i = 0; i < 8; ++i) {
            v = fmaf(v, 1.0002f, 0.001f * (i + 1));
            v = tanhf(v);
        }
        data[idx] = v;
    }
}

__global__ void parentKernel(float* data, int N, int* launch_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Baseline: launch many tiny child grids to highlight launch overhead
        const int seg = 32;
        for (int offset = 0; offset < N; offset += seg) {
            int count = min(seg, N - offset);
            // Naive choice: tiny blocks (1 warp) for each segment
            dim3 child_grid((count + 31) / 32);
            dim3 child_block(32);
            childKernel<<<child_grid, child_block>>>(data + offset, count);
            atomicAdd(launch_count, 1);
        }
    }
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);

    const int N = 262144;  // larger workload to amortize launch overhead; matches optimized
    float* d_data = nullptr;
    int* d_launch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_launch_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_launch_count, 0, sizeof(int)));

    dim3 parent_grid(1);
    dim3 parent_block(64);
    // Single parent that launches many segmented child grids on the device
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    parentKernel<<<parent_grid, parent_block>>>(d_data, N, d_launch_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    int launches = 0;
    CUDA_CHECK(cudaMemcpy(&launches, d_launch_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("Device child launches (baseline): %d\n", launches);
    std::printf("Elapsed_ms: %.6f ms\n", elapsed_ms);

#ifdef VERIFY
    std::vector<float> h_data(N);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_data) {
        checksum += static_cast<double>(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_launch_count));
    return 0;
}
