// optimized_cache_hints.cu - L2 Cache Hints and Cache Policies (Ch7)
//
// WHAT: Cache hints tell the GPU how to handle data in the cache hierarchy:
//   - CACHE ALL (.ca): Cache at all levels (default)
//   - CACHE GLOBAL (.cg): Cache only in L2, bypass L1
//   - CACHE STREAMING (.cs): Streaming load, evict first from L2
//   - CACHE VOLATILE (.cv): Don't cache, always fetch from memory
//
// WHY: Different access patterns benefit from different caching strategies:
//   - Reused data: .ca (keep in cache)
//   - Large sequential scans: .cs (don't pollute cache)
//   - Data written once: .cs/.cv (streaming store)
//   - Multi-GPU coherence: .cv (always see latest)
//
// WHEN TO USE:
//   - .cs for large arrays processed once (e.g., input tensors in inference)
//   - .cg when L1 space is precious for shared memory
//   - .cv for data that changes between kernel launches
//
// IMPACT:
//   - Proper hints can improve effective bandwidth by 10-30%
//   - Wrong hints can thrash cache and hurt performance
//
// NOTE: SM 90+ (Hopper/Blackwell) has improved L2 cache policies.
//       Use CUDA 12+ for best results.

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
// Different Load Cache Policies
//============================================================================

// Default: Cache All (L1 + L2)
__global__ void vector_add_cache_all(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Default loads - cache at all levels
        float va = a[idx];
        float vb = b[idx];
        c[idx] = va + vb;
    }
}

// Cache Streaming (.cs) - for data that won't be reused
__global__ void vector_add_cache_streaming(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va, vb;
        
        // Streaming loads - evict early from L2
        // Good for large arrays processed once
        asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(va) : "l"(a + idx));
        asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(vb) : "l"(b + idx));
        
        // Streaming store - don't allocate in L2
        asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(c + idx), "f"(va + vb));
    }
}

// Cache Global (.cg) - bypass L1, cache only in L2
__global__ void vector_add_cache_global(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va, vb;
        
        // L2-only caching - useful when L1 is precious
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(va) : "l"(a + idx));
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(vb) : "l"(b + idx));
        
        c[idx] = va + vb;
    }
}

//============================================================================
// Vectorized with Cache Hints (float4 = 16 bytes)
//============================================================================

__global__ void vector_add_cache_streaming_vec4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n  // number of float4 elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va, vb, vc;
        
        // 128-bit streaming loads
        asm volatile(
            "ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(va.x), "=f"(va.y), "=f"(va.z), "=f"(va.w)
            : "l"(a + idx)
        );
        asm volatile(
            "ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(vb.x), "=f"(vb.y), "=f"(vb.z), "=f"(vb.w)
            : "l"(b + idx)
        );
        
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        
        // 128-bit streaming store
        asm volatile(
            "st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
            :: "l"(c + idx), "f"(vc.x), "f"(vc.y), "f"(vc.z), "f"(vc.w)
        );
    }
}

//============================================================================
// L2 Cache Set-Aside (for Hopper/Blackwell)
//============================================================================
// On SM 90+, you can reserve a portion of L2 for persistent data.
// This is useful for data that should stay in L2 across kernel launches.

void configure_l2_cache_set_aside(void* data_ptr, size_t data_size) {
#if CUDART_VERSION >= 11040
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    if (prop.major >= 8) {
        cudaStreamAttrValue attr;
        
        // Set up access policy window
        attr.accessPolicyWindow.base_ptr = data_ptr;
        attr.accessPolicyWindow.num_bytes = data_size;
        attr.accessPolicyWindow.hitRatio = 1.0f;  // Try to keep 100% in L2
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        
        // Apply to default stream
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        
        printf("L2 cache set-aside configured for %zu bytes\n", data_size);
    } else {
        printf("L2 set-aside requires SM 80+ (Ampere or newer)\n");
    }
#else
    printf("L2 set-aside requires CUDA 11.4+\n");
#endif
}

void reset_l2_cache_policy() {
#if CUDART_VERSION >= 11040
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.num_bytes = 0;  // Disable window
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
#endif
}

//============================================================================
// Benchmark
//============================================================================

void benchmark_cache_policies() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("L2 Cache Hints Benchmark\n");
    printf("========================\n");
    printf("Device: %s\n", prop.name);
    printf("L2 Cache Size: %d KB\n\n", prop.l2CacheSize / 1024);
    
    // Use size larger than L2 to see cache effects
    const int n = 64 * 1024 * 1024;  // 256 MB total (64M floats × 4 bytes × 3 arrays)
    const size_t bytes = n * sizeof(float);
    
    printf("Array size: %zu MB (larger than L2 to show cache effects)\n\n", bytes / (1024 * 1024));
    
    // Allocate
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Initialize
    std::vector<float> h_a(n, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_a.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_vec4((n / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 20;
    
    // Results
    struct Result {
        const char* name;
        float ms;
        float bandwidth;
    };
    
    std::vector<Result> results;
    
    // Benchmark each variant
    auto benchmark = [&](const char* name, auto kernel, dim3 g, int elem_per_thread) {
        // Clear L2 cache between tests
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            kernel<<<g, block>>>(d_a, d_b, d_c, n / elem_per_thread);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            kernel<<<g, block>>>(d_a, d_b, d_c, n / elem_per_thread);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        
        // Bandwidth: 2 reads + 1 write = 3 × n × sizeof(float)
        float bandwidth = (3.0f * n * sizeof(float)) / (ms / 1000.0f) / 1e9f;
        
        results.push_back({name, ms, bandwidth});
    };
    
    benchmark("Cache All (default)", vector_add_cache_all, grid, 1);
    benchmark("Cache Streaming (.cs)", vector_add_cache_streaming, grid, 1);
    benchmark("Cache Global (.cg)", vector_add_cache_global, grid, 1);
    
    // Special case for vec4 kernel - needs float4* pointers
    {
        const char* name = "Streaming + Vec4";
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        for (int i = 0; i < warmup; ++i) {
            vector_add_cache_streaming_vec4<<<grid_vec4, block>>>(
                reinterpret_cast<const float4*>(d_a),
                reinterpret_cast<const float4*>(d_b),
                reinterpret_cast<float4*>(d_c),
                n / 4
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            vector_add_cache_streaming_vec4<<<grid_vec4, block>>>(
                reinterpret_cast<const float4*>(d_a),
                reinterpret_cast<const float4*>(d_b),
                reinterpret_cast<float4*>(d_c),
                n / 4
            );
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;
        float bandwidth = (3.0f * n * sizeof(float)) / (ms / 1000.0f) / 1e9f;
        results.push_back({name, ms, bandwidth});
    }
    
    // Print results
    printf("%-25s %10s %15s\n", "Variant", "Time (ms)", "Bandwidth (GB/s)");
    printf("%-25s %10s %15s\n", "-------", "---------", "----------------");
    
    for (const auto& r : results) {
        printf("%-25s %10.3f %15.1f\n", r.name, r.ms, r.bandwidth);
    }
    
    // Speedup
    printf("\nSpeedup vs default:\n");
    float baseline = results[0].ms;
    for (const auto& r : results) {
        printf("  %-25s %.2fx\n", r.name, baseline / r.ms);
    }
    
    printf("\nNotes:\n");
    printf("  - .cs (streaming) is best for large arrays processed once\n");
    printf("  - .cg is useful when L1 space is needed for shared memory\n");
    printf("  - Vec4 + streaming combines vectorization with cache hints\n");
    printf("  - Effect varies by GPU and working set size\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    benchmark_cache_policies();
    return 0;
}



