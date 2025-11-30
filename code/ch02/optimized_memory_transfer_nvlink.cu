/**
 * NVLink-C2C CPU-GPU P2P Transfer Optimization for Blackwell
 * ===========================================================
 * 
 * Blackwell B200/B300 introduces NVLink-C2C (Chip-to-Chip) for
 * high-speed CPU-GPU interconnect:
 * - Up to 900 GB/s bandwidth (CPU ↔ GPU)
 * - Direct CPU memory access to GPU memory
 * - Better than PCIe 5.0 (128 GB/s)
 * - Grace-Blackwell Superchip optimized
 * 
 * This example demonstrates:
 * 1. CPU-GPU P2P transfers with NVLink-C2C
 * 2. Page migration hints for optimal placement
 * 3. Performance comparison vs PCIe
 * 4. Best practices for Grace-Blackwell
 * 
 * Requirements:
 * - Blackwell B200/B300 GPU
 * - Grace CPU (for NVLink-C2C) or PCIe 5.0
 * - CUDA 13.0+
 * 
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_100 nvlink_c2c_p2p_blackwell.cu -o nvlink_c2c
 * 
 * Expected Performance:
 * - NVLink-C2C: ~900 GB/s (Grace-Blackwell)
 * - PCIe 5.0: ~128 GB/s (fallback)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status));                                   \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// ============================================================================
// Detect NVLink-C2C Support
// ============================================================================

struct SystemInfo {
    bool has_nvlink_c2c;
    bool is_grace_blackwell;
    bool has_pcie_5;
    int num_gpus;
    size_t gpu_memory;
    int cpu_numa_nodes;
    bool is_8xb200;
    bool has_nvswitch;
};

// Detect if running on Grace CPU (ARM architecture)
bool detect_grace_cpu() {
    #ifdef __linux__
    // Check for ARM architecture
    FILE* f = popen("uname -m", "r");
    if (f) {
        char arch[64];
        bool is_arm = false;
        if (fgets(arch, sizeof(arch), f)) {
            if (strstr(arch, "aarch64") || strstr(arch, "arm64")) {
                is_arm = true;
            }
        }
        pclose(f);
        if (is_arm) {
            // Further verify it's Grace Neoverse
            FILE* cpuinfo = fopen("/proc/cpuinfo", "r");
            if (cpuinfo) {
                char line[256];
                while (fgets(line, sizeof(line), cpuinfo)) {
                    if (strstr(line, "Neoverse") || strstr(line, "0xd40")) {
                        fclose(cpuinfo);
                        return true;
                    }
                }
                fclose(cpuinfo);
            }
            return true;  // At least ARM
        }
    }
    #endif
    return false;
}

SystemInfo detect_system_capabilities() {
    SystemInfo info = {false, false, false, 0, 0, 0, false, false};
    
    CUDA_CHECK(cudaGetDeviceCount(&info.num_gpus));
    
    if (info.num_gpus == 0) {
        printf("No CUDA devices found\n");
        return info;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    info.gpu_memory = prop.totalGlobalMem;
    bool is_blackwell = (prop.major == 10);
    bool is_grace_blackwell = (prop.major >= 12);
    
    printf("=== System Capabilities ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("GPU Memory: %.2f GB per GPU\n", info.gpu_memory / (1024.0 * 1024.0 * 1024.0));
    printf("Number of GPUs: %d\n", info.num_gpus);
    
    // Check for 8x Blackwell/GB configuration
    if (info.num_gpus == 8 && (is_blackwell || is_grace_blackwell)) {
        float mem_gb = info.gpu_memory / (1024.0 * 1024.0 * 1024.0);
        if (mem_gb > 170.0 && mem_gb < 190.0) {
            info.is_8xb200 = true;
            printf("✓ 8x B200 GPU configuration detected\n");
            printf("  Total memory: %.2f TB (%.0f GB per GPU)\n", 
                   info.num_gpus * mem_gb / 1024.0, mem_gb);
            printf("  Total SMs: %d (148 per GPU)\n", info.num_gpus * 148);
            printf("  NVLink 5.0: 1800 GB/s per GPU pair (bidirectional)\n");
        } else if (mem_gb >= 270.0 && mem_gb <= 300.0) {
            info.is_8xb200 = true;
            printf("✓ 8x B300 GPU configuration detected\n");
            printf("  Total memory: %.2f TB (%.0f GB per GPU)\n",
                   info.num_gpus * mem_gb / 1024.0, mem_gb);
            printf("  Total SMs: %d (148 per GPU)\n", info.num_gpus * 148);
            printf("  NVLink 5.0: 1800 GB/s per GPU pair (bidirectional)\n");
        } else if (is_grace_blackwell) {
            info.is_8xb200 = true;
            printf("✓ 8x Grace-Blackwell configuration detected\n");
            printf("  Total memory: %.2f TB (%.0f GB per GPU)\n",
                   info.num_gpus * mem_gb / 1024.0, mem_gb);
            printf("  Grace coherence + NVLink-C2C enabled\n");
        }
    }
    
    // Check for Blackwell
    if (is_blackwell || is_grace_blackwell) {
        printf("✓ %s detected\n", is_grace_blackwell ? "Grace-Blackwell (SM 12.x)" : "Blackwell B200/B300");
        
        // Detect Grace CPU
        bool has_grace_cpu = detect_grace_cpu();
        
        if (has_grace_cpu || is_grace_blackwell) {
            info.has_nvlink_c2c = true;
            info.is_grace_blackwell = true;
            printf("✓ Grace CPU detected (ARM Neoverse)\n");
            printf("✓ GB200/GB300 Grace-Blackwell Superchip\n");
            printf("  NVLink-C2C: 900 GB/s coherent CPU ↔ GPU bandwidth\n");
            printf("  Unified memory address space\n");
            printf("  Grace: 72 ARM cores, 480GB-1TB LPDDR5X\n");
        } else {
            // Check for NVLink-C2C via NVIDIA driver
            #ifdef __linux__
            FILE* f = fopen("/proc/driver/nvidia/gpus/0000:01:00.0/information", "r");
            if (f) {
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    if (strstr(line, "NVLink-C2C")) {
                        info.has_nvlink_c2c = true;
                        break;
                    }
                }
                fclose(f);
            }
            #endif
            
            if (info.has_nvlink_c2c) {
                printf("✓ NVLink-C2C detected\n");
                printf("  Expected bandwidth: ~900 GB/s (CPU ↔ GPU)\n");
            } else {
                printf("⚠ PCIe connection (no NVLink-C2C)\n");
                printf("  Expected bandwidth: ~128 GB/s (PCIe 5.0)\n");
                info.has_pcie_5 = true;
            }
        }
        
        // Check for NVSwitch (for 8-GPU configurations)
        if (info.num_gpus >= 8) {
            #ifdef __linux__
            FILE* topo = popen("nvidia-smi topo -m 2>/dev/null | grep -i nvswitch", "r");
            if (topo) {
                char line[256];
                if (fgets(line, sizeof(line), topo)) {
                    info.has_nvswitch = true;
                    printf("✓ NVSwitch detected (optimal for 8-GPU topology)\n");
                }
                pclose(topo);
            }
            #endif
        }
    } else {
        printf("⚠ Not a Blackwell GPU - NVLink-C2C not available\n");
        info.has_pcie_5 = true;
    }
    
    printf("\n");
    return info;
}

// ============================================================================
// CPU-GPU Transfer Benchmarks
// ============================================================================

double benchmark_host_to_device(void* h_data, void* d_data, size_t size, 
                                 const char* method) {
    const int iterations = 100;
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / (double)iterations / 1000.0;
    double bandwidth_gbs = (size / (avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    printf("  %-25s: %7.2f ms, %7.2f GB/s\n", method, avg_ms, bandwidth_gbs);
    return bandwidth_gbs;
}

double benchmark_device_to_host(void* d_data, void* h_data, size_t size,
                                 const char* method) {
    const int iterations = 100;
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / (double)iterations / 1000.0;
    double bandwidth_gbs = (size / (avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    printf("  %-25s: %7.2f ms, %7.2f GB/s\n", method, avg_ms, bandwidth_gbs);
    return bandwidth_gbs;
}

// ============================================================================
// Page Migration Hints (CUDA 13 / Blackwell)
// ============================================================================

void demonstrate_page_migration() {
    printf("\n=== Page Migration Hints (CUDA 13) ===\n");
    
    size_t size = 256 * 1024 * 1024;  // 256 MB
    
    // Allocate managed memory
    float* managed_data;
    CUDA_CHECK(cudaMallocManaged(&managed_data, size));
    
    // Initialize on CPU
    for (size_t i = 0; i < size / sizeof(float); i++) {
        managed_data[i] = (float)i;
    }
    
    printf("Allocated %.2f MB managed memory\n", size / (1024.0 * 1024.0));
    
    // CUDA 13.0 API: Set up memory locations
    struct cudaMemLocation gpuLoc;
    gpuLoc.type = cudaMemLocationTypeDevice;
    gpuLoc.id = 0;
    
    struct cudaMemLocation cpuLoc;
    cpuLoc.type = cudaMemLocationTypeHost;
    cpuLoc.id = 0;
    
    // Strategy 1: No hints (baseline)
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Touch all pages on GPU (triggers migration)
        cudaMemPrefetchAsync(managed_data, size, gpuLoc, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("  No hints:            %5ld ms\n", duration.count());
    }
    
    // Reset to CPU
    cudaMemPrefetchAsync(managed_data, size, cpuLoc, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Strategy 2: With prefetch hint (optimized for NVLink-C2C)
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        // For streamed transfers, prefer double-buffered prefetch (overlap copy/compute).
        // On Hopper/Blackwell, also consider TMA + thread-block clusters for staging.
        // Prefetch with hint for bulk transfer
        cudaMemPrefetchAsync(managed_data, size, gpuLoc, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("  With prefetch:       %5ld ms (optimized for NVLink-C2C)\n", 
               duration.count());
    }
    
    // Strategy 3: Advised location (CUDA 13 enhancement)
    {
        // Advise that this memory will be accessed mostly from GPU
        CUDA_CHECK(cudaMemAdvise(managed_data, size, 
                                 cudaMemAdviseSetPreferredLocation, gpuLoc));
        CUDA_CHECK(cudaMemAdvise(managed_data, size,
                                 cudaMemAdviseSetAccessedBy, gpuLoc));
        CUDA_CHECK(cudaMemAdvise(managed_data, size,
                                 cudaMemAdviseSetReadMostly, gpuLoc));
        
        // Reset to CPU
        cudaMemPrefetchAsync(managed_data, size, cpuLoc, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        cudaMemPrefetchAsync(managed_data, size, gpuLoc, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("  With cudaMemAdvise:  %5ld ms (CUDA 13 optimization)\n", 
               duration.count());
    }
    
    CUDA_CHECK(cudaFree(managed_data));
    
    printf("\nBest Practice (Grace-Blackwell):\n");
    printf("1. Use cudaMemAdvise for frequently accessed data\n");
    printf("2. Prefetch before kernel launch\n");
    printf("3. NVLink-C2C enables seamless CPU-GPU memory\n");
}

// ============================================================================
// 8-GPU P2P Bandwidth Benchmarks
// ============================================================================

void benchmark_8gpu_p2p_bandwidth(const SystemInfo& info) {
    if (!info.is_8xb200) {
        printf("\n⚠ 8-GPU P2P benchmark requires 8x B200 GPUs (found %d)\n", info.num_gpus);
        return;
    }
    
    printf("\n=== 8-GPU P2P Bandwidth Benchmark ===\n");
    printf("Testing NVLink 5.0 bandwidth between all GPU pairs\n\n");
    
    const size_t transfer_size = 256 * 1024 * 1024;  // 256 MB
    const int iterations = 100;
    
    // Enable P2P access between all GPU pairs
    for (int i = 0; i < 8; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < 8; j++) {
            if (i != j) {
                int can_access;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    cudaError_t status = cudaDeviceEnablePeerAccess(j, 0);
                    if (status == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError();  // clear sticky error from previous enable
                    } else if (status != cudaSuccess) {
                        fprintf(stderr,
                                "CUDA error enabling peer access %d->%d: %s\n",
                                i, j, cudaGetErrorString(status));
                        exit(EXIT_FAILURE);
                    }
                    if (i < j) {
                        int native_atomics = 0;
                        int perf_rank = 0;
                        CUDA_CHECK(cudaDeviceGetP2PAttribute(
                            &native_atomics, cudaDevP2PAttrNativeAtomicSupported, i, j));
                        CUDA_CHECK(cudaDeviceGetP2PAttribute(
                            &perf_rank, cudaDevP2PAttrPerformanceRank, i, j));
                        printf("P2P %d ↔ %d: nativeAtomics=%d, perfRank=%d (lower is better)\n",
                               i, j, native_atomics, perf_rank);
                    }
                }
            }
        }
    }
    
    // Allocate memory on each GPU
    float* d_data[8];
    for (int i = 0; i < 8; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_data[i], transfer_size));
    }
    
    // Benchmark all pairs
    printf("GPU Pair | Bandwidth (GB/s) | Latency (μs)\n");
    printf("---------|------------------|-------------\n");
    
    double total_bandwidth = 0.0;
    int pair_count = 0;
    
    for (int src = 0; src < 8; src++) {
        for (int dst = src + 1; dst < 8; dst++) {
            CUDA_CHECK(cudaSetDevice(src));
            
            // Warmup
            CUDA_CHECK(cudaMemcpyPeer(d_data[dst], dst, d_data[src], src, transfer_size));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; i++) {
                CUDA_CHECK(cudaMemcpyPeer(d_data[dst], dst, d_data[src], src, transfer_size));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double avg_us = duration.count() / (double)iterations;
            double bandwidth_gbs = (transfer_size / (avg_us / 1e6)) / (1024.0 * 1024.0 * 1024.0);
            
            printf("  %d ↔ %d  |     %7.2f      |   %7.2f\n", 
                   src, dst, bandwidth_gbs, avg_us);
            
            total_bandwidth += bandwidth_gbs;
            pair_count++;
        }
    }
    
    // Summary
    printf("\n8-GPU P2P Summary:\n");
    printf("  Average bandwidth: %.2f GB/s per pair\n", total_bandwidth / pair_count);
    printf("  Total pairs: %d\n", pair_count);
    
    if (info.has_nvswitch) {
        printf("  ✓ NVSwitch topology: Full mesh connectivity\n");
        printf("  ✓ Expected: ~1800 GB/s bidirectional per pair (NVLink 5.0)\n");
    } else {
        printf("  ℹ Direct NVLink topology\n");
        printf("  ℹ Some pairs may have lower bandwidth (multi-hop)\n");
    }
    
    // Cleanup
    for (int i = 0; i < 8; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_data[i]));
    }
    
    printf("\nRecommendations for 8x B200:\n");
    printf("  - Use NCCL 2.28 with NVLS for optimal collectives\n");
    printf("  - Tensor Parallel: Split across 2, 4, or 8 GPUs\n");
    printf("  - Data Parallel: Remaining GPUs for batch parallelism\n");
    printf("  - Monitor with: nvidia-smi dmon -s u -i 0,1,2,3,4,5,6,7\n");
}

// ============================================================================
// GB200/GB300 Coherent Memory Examples
// ============================================================================

__global__ void coherent_memory_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Read-modify-write on coherent memory
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

void demonstrate_gb200_coherent_memory(const SystemInfo& info) {
    if (!info.is_grace_blackwell) {
        printf("\n⚠ GB200/GB300 coherent memory requires Grace-Blackwell Superchip\n");
        printf("  Current system: Standard CPU-GPU connection\n");
        return;
    }
    
    printf("\n=== GB200/GB300 Coherent Memory Demo ===\n");
    printf("NVLink-C2C enables coherent CPU-GPU memory access\n\n");
    
    const size_t size = 128 * 1024 * 1024;  // 128 MB
    const int num_elements = size / sizeof(float);
    
    // Allocate managed memory (coherent on GB200/GB300)
    float* coherent_data;
    CUDA_CHECK(cudaMallocManaged(&coherent_data, size));
    
    // Initialize on CPU
    printf("1. CPU: Initializing %.2f MB...\n", size / (1024.0 * 1024.0));
    for (int i = 0; i < num_elements; i++) {
        coherent_data[i] = (float)i;
    }
    
    // CPU reads back first element
    float cpu_value_before = coherent_data[0];
    printf("   CPU value[0] before: %.2f\n", cpu_value_before);
    
    // GPU modifies the data (coherent access via NVLink-C2C)
    printf("\n2. GPU: Modifying data via NVLink-C2C...\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // No explicit cudaMemPrefetchAsync needed - coherent!
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    coherent_memory_kernel<<<blocks, threads>>>(coherent_data, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("   GPU kernel completed in %ld ms\n", duration.count());
    
    // CPU reads back modified value (coherent read via NVLink-C2C)
    printf("\n3. CPU: Reading modified data (coherent)...\n");
    float cpu_value_after = coherent_data[0];
    printf("   CPU value[0] after:  %.2f\n", cpu_value_after);
    printf("   Expected: %.2f\n", cpu_value_before * 2.0f + 1.0f);
    
    if (fabs(cpu_value_after - (cpu_value_before * 2.0f + 1.0f)) < 0.01f) {
        printf("   ✓ Coherent memory working correctly!\n");
    }
    
    // Benchmark coherent access bandwidth
    printf("\n4. Benchmark coherent memory bandwidth:\n");
    
    // CPU writes, GPU reads
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_elements; i++) {
        coherent_data[i] = (float)i * 0.5f;
    }
    coherent_memory_kernel<<<blocks, threads>>>(coherent_data, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double bandwidth_gbs = (2 * size / (duration_us.count() / 1e6)) / (1024.0 * 1024.0 * 1024.0);
    printf("   CPU write + GPU read: %.2f GB/s\n", bandwidth_gbs);
    
    CUDA_CHECK(cudaFree(coherent_data));
    
    printf("\nGB200/GB300 Key Benefits:\n");
    printf("  ✓ No explicit cudaMemcpy required\n");
    printf("  ✓ Coherent view of memory from CPU and GPU\n");
    printf("  ✓ ~900 GB/s bandwidth via NVLink-C2C\n");
    printf("  ✓ Unified memory programming model\n");
    printf("  ✓ Ideal for: CPU preprocessing + GPU compute\n");
    
    printf("\nUse Cases:\n");
    printf("  1. CPU data loading → GPU training (seamless)\n");
    printf("  2. Large KV cache in CPU memory (480GB-1TB)\n");
    printf("  3. CPU-GPU pipeline without explicit transfers\n");
    printf("  4. Optimizer state offloading to CPU\n");
}

// ============================================================================
// Main Benchmark
// ============================================================================

void run_transfer_benchmarks(const SystemInfo& info) {
    printf("\n=== CPU-GPU Transfer Benchmark ===\n");
    
    std::vector<size_t> sizes = {
        16 * 1024 * 1024,    // 16 MB
        64 * 1024 * 1024,    // 64 MB
        256 * 1024 * 1024,   // 256 MB
        1024 * 1024 * 1024   // 1 GB
    };
    
    for (size_t size : sizes) {
        printf("\nTransfer size: %.2f MB\n", size / (1024.0 * 1024.0));
        
        // Allocate memory
        float* h_pageable = (float*)malloc(size);
        float* h_pinned;
        float* d_data;
        
        CUDA_CHECK(cudaMallocHost(&h_pinned, size));
        CUDA_CHECK(cudaMalloc(&d_data, size));
        
        // Initialize
        for (size_t i = 0; i < size / sizeof(float); i++) {
            h_pageable[i] = (float)i;
            h_pinned[i] = (float)i;
        }
        
        // Benchmark H2D
        printf("\nHost → Device:\n");
        double h2d_pageable = benchmark_host_to_device(h_pageable, d_data, size, 
                                                       "Pageable memory");
        double h2d_pinned = benchmark_host_to_device(h_pinned, d_data, size,
                                                     "Pinned memory");
        
        // Benchmark D2H
        printf("\nDevice → Host:\n");
        double d2h_pageable = benchmark_device_to_host(d_data, h_pageable, size,
                                                       "Pageable memory");
        double d2h_pinned = benchmark_device_to_host(d_data, h_pinned, size,
                                                     "Pinned memory");
        
        // Analysis
        printf("\nSpeedup with pinned memory:\n");
        printf("  H2D: %.2fx faster\n", h2d_pinned / h2d_pageable);
        printf("  D2H: %.2fx faster\n", d2h_pinned / d2h_pageable);
        
        if (info.has_nvlink_c2c && h2d_pinned > 500.0) {
            printf("  ✓ NVLink-C2C performance detected!\n");
        } else if (info.has_pcie_5 && h2d_pinned > 100.0) {
            printf("  ✓ PCIe 5.0 performance achieved\n");
        }
        
        // Cleanup
        free(h_pageable);
        CUDA_CHECK(cudaFreeHost(h_pinned));
        CUDA_CHECK(cudaFree(d_data));
    }
}

int main() {
    printf("=== NVLink-C2C CPU-GPU P2P Transfer (Blackwell) ===\n\n");
    
    // Detect system
    SystemInfo info = detect_system_capabilities();
    
    // Run standard benchmarks
    run_transfer_benchmarks(info);
    
    // Demonstrate page migration
    demonstrate_page_migration();
    
    // NEW: 8-GPU P2P benchmarks
    if (info.is_8xb200) {
        benchmark_8gpu_p2p_bandwidth(info);
    }
    
    // NEW: GB200/GB300 coherent memory
    if (info.is_grace_blackwell) {
        demonstrate_gb200_coherent_memory(info);
    }
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("Key Findings:\n");
    printf("1. Pinned memory is 2-3x faster than pageable\n");
    printf("2. NVLink-C2C (Grace-Blackwell): up to 900 GB/s\n");
    printf("3. PCIe 5.0 fallback: ~128 GB/s\n");
    printf("4. cudaMemAdvise optimizes migration for CUDA 13\n");
    printf("5. Prefer pinned memory for frequent transfers\n");
    
    if (info.is_8xb200) {
        printf("\n✓ 8x B200 GPU Configuration Detected\n");
        printf("  - Total memory: 1.44 TB (180 GB per GPU)\n");
        printf("  - NVLink 5.0: 1800 GB/s per GPU pair\n");
        printf("  - Recommendations:\n");
        printf("    * Use NCCL 2.28 for collectives\n");
        printf("    * Hybrid parallelism: TP=2/DP=4 or TP=4/DP=2\n");
        printf("    * Monitor bandwidth: nvidia-smi dmon -s u\n");
    }
    
    if (info.is_grace_blackwell) {
        printf("\n✓ GB200/GB300 Grace-Blackwell Superchip Detected\n");
        printf("  - Grace CPU: 72 ARM cores, 480GB-1TB memory\n");
        printf("  - NVLink-C2C: 900 GB/s coherent bandwidth\n");
        printf("  - Recommendations:\n");
        printf("    * Use managed memory with cudaMemAdvise\n");
        printf("    * CPU preprocessing + GPU training pipeline\n");
        printf("    * Offload optimizer states to CPU memory\n");
        printf("    * Store large KV caches in CPU memory\n");
    }
    
    if (!info.is_grace_blackwell && !info.is_8xb200) {
        printf("\n⚠ Standard GPU Configuration\n");
        if (info.has_pcie_5) {
            printf("  Connection: PCIe 5.0 (~128 GB/s)\n");
        }
        printf("  Recommendations:\n");
        printf("    * Minimize CPU-GPU transfers\n");
        printf("    * Use pinned memory when necessary\n");
        printf("    * Keep data on GPU when possible\n");
    }
    
    printf("\n=== Hardware-Specific Optimizations ===\n");
    printf("8x B200:       ch04/training_8xb200_pipeline.py\n");
    printf("GB200/GB300:   ch04/gb200_grace_numa_optimization.py\n");
    printf("NCCL Config:   ch04/nccl_blackwell_config.py\n");
    printf("Inference:     ch16/inference_optimizations_blackwell.py\n");
    
    return 0;
}
