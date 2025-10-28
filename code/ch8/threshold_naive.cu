// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// threshold_naive.cu
// For CUDA 13 pipeline/TMA usage see ch7/async_prefetch_tma.cu or ch10/tma_2d_pipeline_blackwell.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void threshold_naive(const float* X, float* Y, float threshold, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        if (X[i] > threshold) {
            Y[i] = X[i]; // branch 1
        } else {
            Y[i] = 0.0f; // branch 2
        }
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const float threshold = 0.5f;
    const size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float* h_X = (float*)malloc(bytes);
    float* h_Y = (float*)malloc(bytes);
    
    // Initialize input data (mix of values above and below threshold)
    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;  // Random 0-1
    }
    
    // Allocate device memory
    float* d_X = nullptr;
    float* d_Y = nullptr;
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);
    
    // Copy to device
    cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
    
    // Launch configuration
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    threshold_naive<<<gridSize, blockSize>>>(d_X, d_Y, threshold, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Naive threshold kernel time: %.3f ms\n", time);
    printf("This kernel exhibits warp divergence due to if/else branching\n");
    printf("Profile with: ncu --metrics smsp__warp_execution_efficiency.avg.pct ./threshold_naive\n");
    
    // Verify results
    cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);
    
    // Count elements above threshold
    int count_input = 0, count_output = 0;
    for (int i = 0; i < N; ++i) {
        if (h_X[i] > threshold) count_input++;
        if (h_Y[i] > 0.0f) count_output++;
    }
    
    printf("Input elements > %.1f: %d\n", threshold, count_input);
    printf("Output elements > 0: %d\n", count_output);
printf("Results %s\n", (count_input == count_output) ? "CORRECT" : "INCORRECT");

// Cleanup
cudaEventDestroy(start);
cudaEventDestroy(stop);
free(h_X);
free(h_Y);
cudaFree(d_X);
cudaFree(d_Y);

return 0;
}
