// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// For an end-to-end CUDA 13 pipeline/TMA sample see ch7/async_prefetch_tma.cu or ch10/tma_2d_pipeline_blackwell.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Naive implementation: separate kernels (low arithmetic intensity)
__global__ void squareKernel(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = in[i] * in[i];
    }
}

__global__ void addKernel(const float* a, const float* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = a[i] + b[i];
    }
}

__global__ void sqrtKernel(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = sqrtf(in[i]);
    }
}

// Fused implementation: single kernel (higher arithmetic intensity)
__global__ void fusedL2Norm(const float *a, const float *b, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float ai = a[i];
        float bi = b[i];
        
        // Perform multiple arithmetic ops on ai and bi before storing:
        // 2 multiplications + 1 addition + 1 square root = 4 FLOPs
        // Memory: 2 reads (8 bytes) + 1 write (4 bytes) = 12 bytes
        // Arithmetic intensity: 4 FLOPs / 12 bytes = 0.33 FLOPs/byte
        float sumsq = ai * ai + bi * bi;
        out[i] = sqrtf(sumsq);
    }
}

void naiveL2Norm(const float* a, const float* b, float* out, int N, 
                 float* temp1, float* temp2, float* temp3) {
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    // 4 separate kernel launches
    squareKernel<<<gridSize, blockSize>>>(a, temp1, N);
    squareKernel<<<gridSize, blockSize>>>(b, temp2, N);
    addKernel<<<gridSize, blockSize>>>(temp1, temp2, temp3, N);
    sqrtKernel<<<gridSize, blockSize>>>(temp3, out, N);
    
    cudaDeviceSynchronize();
}

void fusedL2NormWrapper(const float* a, const float* b, float* out, int N) {
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    // Single kernel launch
    fusedL2Norm<<<gridSize, blockSize>>>(a, b, out, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out_naive = (float*)malloc(bytes);
    float* h_out_fused = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 100) / 100.0f;
        h_b[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }
    
    // Allocate device memory
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out_naive = nullptr;
    float* d_out_fused = nullptr;
    float* d_temp1 = nullptr;
    float* d_temp2 = nullptr;
    float* d_temp3 = nullptr;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out_naive, bytes);
    cudaMalloc(&d_out_fused, bytes);
    cudaMalloc(&d_temp1, bytes);  // For naive version
    cudaMalloc(&d_temp2, bytes);
    cudaMalloc(&d_temp3, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Time naive implementation
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        naiveL2Norm(d_a, d_b, d_out_naive, N, d_temp1, d_temp2, d_temp3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Time fused implementation
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        fusedL2NormWrapper(d_a, d_b, d_out_fused, N);
    }
    end = std::chrono::high_resolution_clock::now();
    auto fused_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    printf("=== Arithmetic Intensity Comparison ===\n");
    printf("Problem size: %d elements\n", N);
    printf("Naive implementation time: %.3f ms\n", naive_time);
    printf("Fused implementation time: %.3f ms\n", fused_time);
    printf("Fusion speedup: %.2fx\n", naive_time / fused_time);
    
    // Arithmetic intensity analysis
    printf("\n=== Arithmetic Intensity Analysis ===\n");
    printf("Naive version:\n");
    printf("  - 4 kernels, each with ~1 FLOP per element\n");
    printf("  - Each kernel: 1 FLOP / 12 bytes = 0.083 FLOPs/byte\n");
    printf("  - Total memory traffic: ~36 bytes per element (intermediates)\n");
    printf("Fused version:\n");
    printf("  - 1 kernel with 4 FLOPs per element\n");
    printf("  - 4 FLOPs / 12 bytes = 0.33 FLOPs/byte\n");
    printf("  - 4x higher arithmetic intensity\n");
    
    // Verify results
    cudaMemcpy(h_out_naive, d_out_naive, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_fused, d_out_fused, bytes, cudaMemcpyDeviceToHost);
    
    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        float error = fabs(h_out_naive[i] - h_out_fused[i]);
        max_error = fmax(max_error, error);
        if (error > 1e-5) {
            correct = false;
        }
    }
    
    printf("\nResults: %s (max error: %.2e)\n", 
           correct ? "CORRECT" : "INCORRECT", max_error);
    
printf("\nProfile with Nsight Compute roofline analysis:\n");
printf("ncu --metrics dram__throughput.avg.pct_of_peak_sustained,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./fused_l2norm\n");

// Cleanup
free(h_a);
    free(h_b);
    free(h_out_naive);
    free(h_out_fused);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out_naive);
    cudaFree(d_out_fused);
    cudaFree(d_temp1);
cudaFree(d_temp2);
cudaFree(d_temp3);

return 0;
}
