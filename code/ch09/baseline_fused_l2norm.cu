// baseline_fused_l2norm.cu -- Naive L2 norm with 4 separate kernels (baseline).

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
    float* d_temp1 = nullptr;
    float* d_temp2 = nullptr;
    float* d_temp3 = nullptr;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_temp1, bytes);
    cudaMalloc(&d_temp2, bytes);
    cudaMalloc(&d_temp3, bytes);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        naiveL2Norm(d_a, d_b, d_out, N, d_temp1, d_temp2, d_temp3);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_total;
    cudaEventElapsedTime(&time_total, start, stop);
    float time_avg = time_total / iterations;
    
    printf("Naive L2 norm (4 kernels, baseline): %.3f ms\n", time_avg);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_a);
    free(h_b);
    free(h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    cudaFree(d_temp3);
    
    return 0;
}

