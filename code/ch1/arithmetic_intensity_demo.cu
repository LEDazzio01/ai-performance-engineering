/*
 * Arithmetic Intensity Optimization Demo
 * 
 * Demonstrates various techniques to increase arithmetic intensity:
 * 1. Baseline: Simple operation (AI ~0.125 FLOP/Byte)
 * 2. Unrolled: Loop unrolling (enables other optimizations)
 * 3. Vectorized: float4 loads (reduces memory transactions)
 * 4. Optimized: Unroll + vectorize + more FLOPs (AI ~2.5 FLOP/Byte)
 *
 * Compile: make arithmetic_intensity_demo
 * Run: ./arithmetic_intensity_demo
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (10 * 1024 * 1024)  // 10M elements
#define BLOCK_SIZE 256

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Timing helper
class CudaTimer {
public:
    cudaEvent_t start, stop;
    
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void start_timer() {
        CUDA_CHECK(cudaEventRecord(start));
    }
    
    float stop_timer() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};

// ============================================================================
// 1. BASELINE: Simple multiply (low AI)
// ============================================================================
__global__ void baseline_kernel(const float* __restrict__ a, 
                                const float* __restrict__ b, 
                                float* __restrict__ out, 
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

// AI = 1 FLOP / (2 reads + 1 write) × 4 bytes = 1 / 12 = 0.083 FLOP/Byte
// Actually: 1 FLOP / 8 bytes (2 loads) = 0.125 FLOP/Byte (ignoring store)

// ============================================================================
// 2. UNROLLED: Loop unrolling (exposes parallelism)
// ============================================================================
__global__ void unrolled_kernel(const float* __restrict__ a, 
                                const float* __restrict__ b, 
                                float* __restrict__ out, 
                                int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n) {
        // Unroll by 4
        out[idx]   = a[idx]   * b[idx];
        out[idx+1] = a[idx+1] * b[idx+1];
        out[idx+2] = a[idx+2] * b[idx+2];
        out[idx+3] = a[idx+3] * b[idx+3];
    } else {
        // Handle remainder
        for (int i = idx; i < n && i < idx + 4; i++) {
            out[i] = a[i] * b[i];
        }
    }
}

// AI = 4 FLOP / 32 bytes = 0.125 FLOP/Byte (same, but better ILP)

// ============================================================================
// 3. VECTORIZED: float4 loads (reduces memory transactions)
// ============================================================================
__global__ void vectorized_kernel(const float* __restrict__ a, 
                                  const float* __restrict__ b, 
                                  float* __restrict__ out, 
                                  int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n) {
        // Vectorized loads (16-byte aligned)
        float4 a_vec = *reinterpret_cast<const float4*>(&a[idx]);
        float4 b_vec = *reinterpret_cast<const float4*>(&b[idx]);
        
        float4 result;
        result.x = a_vec.x * b_vec.x;
        result.y = a_vec.y * b_vec.y;
        result.z = a_vec.z * b_vec.z;
        result.w = a_vec.w * b_vec.w;
        
        // Vectorized store
        *reinterpret_cast<float4*>(&out[idx]) = result;
    }
}

// AI = 4 FLOP / (2×16 + 16) bytes = 4 / 48 = 0.083 FLOP/Byte
// But: Fewer memory transactions (coalesced) → Better throughput

// ============================================================================
// 4. OPTIMIZED: Unroll + vectorize + more FLOPs per load
// ============================================================================
__global__ void optimized_kernel(const float* __restrict__ a, 
                                 const float* __restrict__ b, 
                                 float* __restrict__ out, 
                                 int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n) {
        // Vectorized loads
        float4 a_vec = *reinterpret_cast<const float4*>(&a[idx]);
        float4 b_vec = *reinterpret_cast<const float4*>(&b[idx]);
        
        // More computation per load (polynomial evaluation)
        // f(x) = exp(a*b) = exp(x)
        // Much more FLOPs per memory access!
        float4 result;
        float mul_x = a_vec.x * b_vec.x;
        float mul_y = a_vec.y * b_vec.y;
        float mul_z = a_vec.z * b_vec.z;
        float mul_w = a_vec.w * b_vec.w;
        
        // expf is ~20 FLOPs
        result.x = expf(mul_x);
        result.y = expf(mul_y);
        result.z = expf(mul_z);
        result.w = expf(mul_w);
        
        *reinterpret_cast<float4*>(&out[idx]) = result;
    }
}

// AI = 4 × (1 mul + 20 expf) FLOPs / (2×16 bytes loaded)
//    = 84 FLOPs / 32 bytes = 2.625 FLOP/Byte
// 20x improvement over baseline!

// ============================================================================
// 5. FUSED: Multiple operations in one kernel
// ============================================================================
__global__ void fused_kernel(const float* __restrict__ a, 
                             const float* __restrict__ b, 
                             const float* __restrict__ c, 
                             float* __restrict__ out, 
                             int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n) {
        float4 a_vec = *reinterpret_cast<const float4*>(&a[idx]);
        float4 b_vec = *reinterpret_cast<const float4*>(&b[idx]);
        float4 c_vec = *reinterpret_cast<const float4*>(&c[idx]);
        
        // Fused: (a + b) * (a - c) + expf(b)
        // Much better than 3 separate kernels!
        float4 result;
        result.x = (a_vec.x + b_vec.x) * (a_vec.x - c_vec.x) + expf(b_vec.x);
        result.y = (a_vec.y + b_vec.y) * (a_vec.y - c_vec.y) + expf(b_vec.y);
        result.z = (a_vec.z + b_vec.z) * (a_vec.z - c_vec.z) + expf(b_vec.z);
        result.w = (a_vec.w + b_vec.w) * (a_vec.w - c_vec.w) + expf(b_vec.w);
        
        *reinterpret_cast<float4*>(&out[idx]) = result;
    }
}

// AI = 4 × (2 add + 1 sub + 1 mul + 20 expf) FLOPs / (3×16 bytes loaded)
//    = 96 FLOPs / 48 bytes = 2.0 FLOP/Byte
// Without fusion: Would need 3 separate kernels = 3× memory traffic!

// ============================================================================
// Benchmarking
// ============================================================================

void init_array(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 100) / 100.0f + 0.01f;  // Avoid 0
    }
}

void benchmark_kernel(const char* name,
                     void* kernel_func,
                     dim3 grid, dim3 block,
                     void** args,
                     double flops_per_element,
                     int iterations) {
    
    CudaTimer timer;
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaLaunchKernel(kernel_func, grid, block, args, 0, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    timer.start_timer();
    for (int i = 0; i < iterations; i++) {
        cudaLaunchKernel(kernel_func, grid, block, args, 0, 0);
    }
    float time_ms = timer.stop_timer();
    time_ms /= iterations;
    
    // Calculate performance
    double total_flops = flops_per_element * N;
    double gflops = (total_flops / (time_ms / 1000.0)) / 1e9;
    
    // Estimate AI (approximate)
    double bytes_loaded = 8.0 * N;  // 2 arrays × 4 bytes (ignoring stores)
    double ai = total_flops / bytes_loaded;
    
    printf("%s:\n", name);
    printf("  Time: %.3f ms\n", time_ms);
    printf("  Performance: %.1f GFLOPS\n", gflops);
    printf("  Estimated AI: %.3f FLOP/Byte\n", ai);
    
    // Roofline position
    double ridge_point = 250.0;  // B200: 2000 TFLOPS / 8 TB/s
    if (ai < ridge_point) {
        printf("  Status: Memory-bound (AI < %.0f)\n", ridge_point);
    } else {
        printf("  Status: Compute-bound (AI > %.0f)\n", ridge_point);
    }
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("Arithmetic Intensity Optimization Demo\n");
    printf("========================================\n\n");
    
    printf("Array size: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("B200 Ridge Point: 250 FLOP/Byte\n\n");
    
    // Allocate host memory
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    
    // Initialize
    init_array(h_a, N);
    init_array(h_b, N);
    init_array(h_c, N);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice));
    
    int iterations = 100;
    int n = N;  // Create variable for kernel args
    
    printf("Benchmarking (averaged over %d iterations)...\n\n", iterations);
    
    // 1. Baseline
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        void* args[] = {&d_a, &d_b, &d_out, &n};
        benchmark_kernel("1. Baseline (Simple Multiply)", 
                        (void*)baseline_kernel, grid, block, args, 1.0, iterations);
    }
    
    // 2. Unrolled
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((N/4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        void* args[] = {&d_a, &d_b, &d_out, &n};
        benchmark_kernel("2. Unrolled (4×)", 
                        (void*)unrolled_kernel, grid, block, args, 1.0, iterations);
    }
    
    // 3. Vectorized
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((N/4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        void* args[] = {&d_a, &d_b, &d_out, &n};
        benchmark_kernel("3. Vectorized (float4 loads)", 
                        (void*)vectorized_kernel, grid, block, args, 1.0, iterations);
    }
    
    // 4. Optimized (more FLOPs)
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((N/4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        void* args[] = {&d_a, &d_b, &d_out, &n};
        benchmark_kernel("4. Optimized (vectorized + expf)", 
                        (void*)optimized_kernel, grid, block, args, 21.0, iterations);
    }
    
    // 5. Fused
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((N/4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        void* args[] = {&d_a, &d_b, &d_c, &d_out, &n};
        benchmark_kernel("5. Fused (multi-op kernel)", 
                        (void*)fused_kernel, grid, block, args, 24.0, iterations);
    }
    
    printf("========================================\n");
    printf("Summary:\n");
    printf("  • Baseline:    AI ~0.125 FLOP/Byte (deep memory-bound)\n");
    printf("  • Unrolled:    AI ~0.125 FLOP/Byte (better ILP)\n");
    printf("  • Vectorized:  AI ~0.125 FLOP/Byte (fewer transactions)\n");
    printf("  • Optimized:   AI ~2.6 FLOP/Byte   (20× better!)\n");
    printf("  • Fused:       AI ~2.0 FLOP/Byte   (reduced traffic)\n");
    printf("\n");
    printf("Key insights:\n");
    printf("  1. Unrolling enables vectorization and ILP\n");
    printf("  2. Vectorization reduces memory transactions\n");
    printf("  3. Adding FLOPs per load increases AI dramatically\n");
    printf("  4. Fusion combines ops → fewer global memory passes\n");
    printf("\n");
    printf("All optimizations move kernels right on roofline!\n");
    printf("========================================\n");
    
    // Cleanup
    free(h_a); free(h_b); free(h_c); free(h_out);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_out));
    
    return 0;
}

