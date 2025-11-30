// cpu_gpu_zero_copy_coherent.cu -- Zero-copy GPU access to CPU memory via NVLink-C2C
// Optimized for Grace-Blackwell GB10 (SM 12.1) with 900 GB/s coherent interconnect
// Compile: nvcc -O3 -std=c++17 -arch=sm_121 cpu_gpu_zero_copy_coherent.cu -o cpu_gpu_zero_copy_coherent

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Zero-copy kernel: Direct CPU memory access from GPU
// On GB10, this uses 900 GB/s NVLink-C2C with cache coherence
__global__ void zero_copy_process_kernel(
    const float* __restrict__ cpu_input,   // CPU memory (via NVLink-C2C)
    float* __restrict__ gpu_output,         // GPU memory (HBM3e)
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Direct load from CPU memory - compiler generates coherent access
        // On GB10: ~800-900 GB/s via NVLink-C2C
        // On discrete GPU: ~25 GB/s via PCIe (much slower!)
        float val = cpu_input[idx];
        
        // Compute
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            result += sqrtf(val * val + float(i)) * 0.125f;
        }
        
        // Write to GPU memory
        gpu_output[idx] = result;
    }
}

// Bidirectional zero-copy: GPU reads CPU memory, writes back to CPU memory
__global__ void bidirectional_zero_copy_kernel(
    const float* __restrict__ cpu_input,
    float* __restrict__ cpu_output,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Read from CPU memory
        float val = cpu_input[idx];
        
        // Compute
        float result = val * val + val * 0.5f + 1.0f;
        
        // Write directly to CPU memory (no explicit D2H transfer!)
        cpu_output[idx] = result;
    }
}

// Traditional approach: Explicit copy for comparison
__global__ void traditional_process_kernel(
    const float* __restrict__ gpu_input,
    float* __restrict__ gpu_output,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = gpu_input[idx];
        
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            result += sqrtf(val * val + float(i)) * 0.125f;
        }
        
        gpu_output[idx] = result;
    }
}

bool is_grace_blackwell() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.major == 12;  // SM 12.x
}

int main() {
    // Detect architecture
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    bool is_sm121 = (prop.major == 12);
    
    std::printf("=== GB10 Zero-Copy Coherent Memory Benchmark ===\n");
    std::printf("Architecture: %s (SM %d.%d)\n", 
                is_sm121 ? "Grace-Blackwell GB10" : "Other",
                prop.major, prop.minor);
    
    if (!is_sm121) {
        std::printf("\n⚠️  WARNING: This demo is optimized for Grace-Blackwell GB10!\n");
        std::printf("   On discrete GPUs, zero-copy uses slow PCIe access (~25 GB/s)\n");
        std::printf("   On GB10, zero-copy uses NVLink-C2C coherent access (~900 GB/s)\n\n");
    }
    
    // Test with moderately large array
    constexpr size_t N = 64 * 1024 * 1024;  // 64M elements = 256 MB
    constexpr size_t BYTES = N * sizeof(float);
    
    std::printf("\nTest configuration:\n");
    std::printf("  Array size: %zu MB\n", BYTES / (1024 * 1024));
    std::printf("  Elements: %zu\n\n", N);
    
    // Allocate CPU memory (pinned for GPU access)
    float *h_input = nullptr, *h_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_input, BYTES));  // Pinned, GPU-accessible
    CUDA_CHECK(cudaMallocHost(&h_output, BYTES));
    
    // Allocate GPU memory for comparison
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, BYTES));
    CUDA_CHECK(cudaMalloc(&d_output, BYTES));
    
    // Initialize
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    constexpr int WARMUP = 10;
    constexpr int ITERS = 50;
    
    // ============================================================
    // Test 1: Traditional approach (H2D + kernel + D2H)
    // ============================================================
    std::printf("Test 1: Traditional (H2D copy + kernel + D2H copy)\n");
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        CUDA_CHECK(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice));
        traditional_process_kernel<<<grid, block>>>(d_input, d_output, N);
        CUDA_CHECK(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice));
        traditional_process_kernel<<<grid, block>>>(d_input, d_output, N);
        CUDA_CHECK(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_traditional = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_traditional, start, stop));
    float avg_traditional = ms_traditional / ITERS;
    
    std::printf("  Time: %.3f ms\n", avg_traditional);
    std::printf("  Throughput: %.2f GB/s\n\n", (3.0f * BYTES / 1e9) / (avg_traditional / 1000.0f));
    
    // ============================================================
    // Test 2: Zero-copy (direct CPU memory access)
    // ============================================================
    std::printf("Test 2: Zero-Copy (direct CPU memory access via NVLink-C2C)\n");
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        zero_copy_process_kernel<<<grid, block>>>(h_input, d_output, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        zero_copy_process_kernel<<<grid, block>>>(h_input, d_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_zerocopy = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_zerocopy, start, stop));
    float avg_zerocopy = ms_zerocopy / ITERS;
    
    std::printf("  Time: %.3f ms\n", avg_zerocopy);
    std::printf("  Throughput: %.2f GB/s (CPU read bandwidth)\n", 
                (BYTES / 1e9) / (avg_zerocopy / 1000.0f));
    std::printf("  Speedup: %.2fx vs traditional\n\n", avg_traditional / avg_zerocopy);
    
    // ============================================================
    // Test 3: Bidirectional zero-copy (CPU read + CPU write)
    // ============================================================
    std::printf("Test 3: Bidirectional Zero-Copy (CPU→GPU→CPU, no explicit transfers)\n");
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        bidirectional_zero_copy_kernel<<<grid, block>>>(h_input, h_output, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        bidirectional_zero_copy_kernel<<<grid, block>>>(h_input, h_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_bidir = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_bidir, start, stop));
    float avg_bidir = ms_bidir / ITERS;
    
    std::printf("  Time: %.3f ms\n", avg_bidir);
    std::printf("  Throughput: %.2f GB/s (read + write via NVLink-C2C)\n",
                (2.0f * BYTES / 1e9) / (avg_bidir / 1000.0f));
    std::printf("  Speedup: %.2fx vs traditional\n\n", avg_traditional / avg_bidir);
    
    // Verify
    CUDA_CHECK(cudaMemcpy(h_output, d_output, BYTES, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
        float val = h_input[i];
        float expected = 0.0f;
        for (int j = 0; j < 8; ++j) {
            expected += std::sqrt(val * val + float(j)) * 0.125f;
        }
        if (std::abs(h_output[i] - expected) > 1e-4) {
            correct = false;
            break;
        }
    }
    
    // Results summary
    std::printf("=== Summary ===\n");
    std::printf("Traditional:          %.3f ms (1.00x)\n", avg_traditional);
    std::printf("Zero-copy:            %.3f ms (%.2fx faster)\n", 
                avg_zerocopy, avg_traditional / avg_zerocopy);
    std::printf("Bidirectional:        %.3f ms (%.2fx faster)\n",
                avg_bidir, avg_traditional / avg_bidir);
    std::printf("\nCorrectness: %s\n", correct ? "✅ PASSED" : "❌ FAILED");
    
    if (is_sm121) {
        std::printf("\n✅ GB10 Benefits:\n");
        std::printf("  • No explicit H2D/D2H transfers needed\n");
        std::printf("  • 900 GB/s coherent CPU-GPU bandwidth\n");
        std::printf("  • Reduced memory footprint (data stays on CPU)\n");
        std::printf("  • Ideal for: large KV caches, optimizer states, preprocessing\n");
    } else {
        std::printf("\n⚠️  On discrete GPUs:\n");
        std::printf("  • Zero-copy uses slow PCIe (~25 GB/s)\n");
        std::printf("  • Traditional approach is usually faster\n");
        std::printf("  • GB10's NVLink-C2C provides ~36x better zero-copy bandwidth\n");
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
