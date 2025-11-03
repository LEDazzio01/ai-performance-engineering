// simple_kernel.cu
// Basic CUDA kernel demonstrating fundamental concepts:
// - Kernel launch configuration
// - Thread indexing
// - Memory management (pinned memory)
// - Element-wise operations

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//------------------------------------------------------
__global__ void myKernel(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        input[idx] *= 2.0f;           // scale in-place by 2
    }
}

int main() {
    const int N = 1'000'000;

    float* h_input = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    float* d_input = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_input, d_input, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_input[i] != 2.0f) {
            printf("Error at index %d: expected 2.0, got %f\n", i, h_input[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("✓ Simple kernel succeeded: %d elements scaled by 2.0f\n", N);
        printf("  Configuration: %d blocks × %d threads = %d total threads\n",
               blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);
    }

    cudaFree(d_input);
    cudaFreeHost(h_input);
    return 0;
}

