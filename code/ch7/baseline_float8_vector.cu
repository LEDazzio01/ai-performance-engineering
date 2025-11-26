// baseline_float8_vector.cu - Scalar/float4 Loads (Ch7)
//
// WHAT: Standard scalar and float4 (16-byte) vectorized loads.
// No 32-byte vectorization.
//
// WHY THIS IS SLOWER:
//   - Scalar: 1 float (4 bytes) per load instruction
//   - float4: 4 floats (16 bytes) per load instruction
//   - On Blackwell with HBM3e (~8 TB/s), wider loads needed to saturate bandwidth
//
// COMPARE WITH: optimized_float8_vector.cu
//   - Optimized uses 32-byte (float8) loads
//   - 8 floats per load instruction
//   - Better bandwidth utilization on Blackwell

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;

//============================================================================
// Baseline: Scalar Loads (4 bytes)
//============================================================================

__global__ void baseline_vector_add_scalar(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // Two 4-byte loads, one 4-byte store
    }
}

//============================================================================
// Baseline: float4 Loads (16 bytes)
//============================================================================

__global__ void baseline_vector_add_float4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];  // 16-byte load
        float4 vb = b[idx];  // 16-byte load
        
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        
        c[idx] = vc;  // 16-byte store
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Baseline Vectorized Loads (Scalar + float4)\n");
    printf("===========================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("\n");
    
    // Large array to saturate bandwidth
    const int N = 128 * 1024 * 1024;  // 128M floats = 512 MB per array
    const size_t bytes = N * sizeof(float);
    
    printf("Array size: %zu MB per array\n", bytes / (1024 * 1024));
    printf("Total data movement: %zu MB (2 reads + 1 write)\n\n", 
           3 * bytes / (1024 * 1024));
    
    // Allocate
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Initialize
    std::vector<float> h_data(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_a, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 20;
    
    //========================================================================
    // Benchmark Scalar
    //========================================================================
    printf("%-20s %8s  %10s\n", "Variant", "Time", "Bandwidth");
    printf("%-20s %8s  %10s\n", "-------", "----", "---------");
    
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Clear L2
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            baseline_vector_add_scalar<<<grid, block>>>(d_a, d_b, d_c, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            baseline_vector_add_scalar<<<grid, block>>>(d_a, d_b, d_c, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        
        float bandwidth = (3.0f * bytes) / (ms / 1000.0f) / 1e9f;
        printf("%-20s %8.3f ms  %8.1f GB/s\n", "Scalar (4B)", ms, bandwidth);
    }
    
    //========================================================================
    // Benchmark float4
    //========================================================================
    {
        int num_float4 = N / 4;
        dim3 block(BLOCK_SIZE);
        dim3 grid((num_float4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Clear L2
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            baseline_vector_add_float4<<<grid, block>>>(
                reinterpret_cast<float4*>(d_a),
                reinterpret_cast<float4*>(d_b),
                reinterpret_cast<float4*>(d_c),
                num_float4
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            baseline_vector_add_float4<<<grid, block>>>(
                reinterpret_cast<float4*>(d_a),
                reinterpret_cast<float4*>(d_b),
                reinterpret_cast<float4*>(d_c),
                num_float4
            );
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        
        float bandwidth = (3.0f * bytes) / (ms / 1000.0f) / 1e9f;
        printf("%-20s %8.3f ms  %8.1f GB/s\n", "float4 (16B)", ms, bandwidth);
    }
    
    printf("\nNotes:\n");
    printf("  - Scalar: 4 bytes per load (1 float)\n");
    printf("  - float4: 16 bytes per load (4 floats)\n");
    printf("  - Compare with optimized_float8_vector for 32-byte loads\n");
    printf("  - 32-byte loads are optimal on Blackwell (SM 100+)\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return 0;
}

