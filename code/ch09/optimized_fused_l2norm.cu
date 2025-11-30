// optimized_fused_l2norm.cu -- Fused L2 norm single kernel (optimized).

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

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
    
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 100) / 100.0f;
        h_b[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }
    
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fusedL2NormWrapper(d_a, d_b, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_total;
    cudaEventElapsedTime(&time_total, start, stop);
    float time_avg = time_total / iterations;
    
    printf("Fused L2 norm (single kernel, optimized): %.3f ms\n", time_avg);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_a);
    free(h_b);
    free(h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    
    return 0;
}

