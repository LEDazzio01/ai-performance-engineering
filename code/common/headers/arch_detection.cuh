#pragma once

/**
 * arch_detection.cuh - GPU architecture detection and capability queries
 * 
 * Provides unified interface for querying hardware capabilities across
 * Ampere, Hopper, Blackwell, and Grace-Blackwell architectures.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <initializer_list>

namespace cuda_arch {

// TMA box size limits by architecture
struct TMALimits {
    uint32_t max_1d_box_size;
    uint32_t max_2d_box_width;
    uint32_t max_2d_box_height;
};

inline TMALimits get_tma_limits() {
    static TMALimits cached_limits = {0, 0, 0};
    
    // Return cached limits if already queried
    if (cached_limits.max_1d_box_size != 0) {
        return cached_limits;
    }
    
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
        // Conservative fallback
        cached_limits = {256, 64, 32};
        return cached_limits;
    }
    
    // Set limits based on compute capability
    if (props.major == 12 && props.minor == 1) {
        // Grace-Blackwell GB10 (SM 12.1) - tested limits
        cached_limits = {256, 64, 32};
    } else if (props.major == 10 && props.minor == 0) {
        // Blackwell B200 (SM 10.0) - larger limits
        cached_limits = {1024, 128, 128};
    } else if (props.major == 9 && props.minor == 0) {
        // Hopper H100 (SM 9.0) - similar to Blackwell
        cached_limits = {1024, 128, 128};
    } else {
        // Conservative fallback for unknown architectures
        cached_limits = {256, 64, 32};
    }
    
    return cached_limits;
}

inline int get_max_shared_mem_per_block() {
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }

    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) != cudaSuccess ||
        value == 0) {
        cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    }

    if (value == 0) {
        // Fallback to 48 KB – supported on all post-Volta GPUs.
        value = 48 * 1024;
    }

    cached = value;
    return cached;
}

struct ArchitectureLimits {
    enum class TensorCoreGeneration {
        None,
        Ampere,
        Hopper,
        Blackwell
    };

    TMALimits tma{};
    int max_shared_mem_per_block = 48 * 1024;
    int max_shared_mem_per_sm = 0;
    int max_threads_per_block = 1024;
    int warp_size = 32;

    bool supports_clusters = false;
    int max_cluster_size = 1;

    TensorCoreGeneration tensor_core_gen = TensorCoreGeneration::None;
    int tensor_tile_m = 64;
    int tensor_tile_n = 64;
    int tensor_tile_k = 16;

    bool has_grace_coherence = false;
    bool has_nvlink_c2c = false;
    size_t kernel_parameter_limit = 4096;
};

struct TensorCoreTile {
    int m;
    int n;
    int k;
};

inline const ArchitectureLimits& get_architecture_limits() {
    static ArchitectureLimits cached{};
    static bool initialized = false;
    if (initialized) {
        return cached;
    }

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
        cached.tma = get_tma_limits();
        initialized = true;
        return cached;
    }

    cached.tma = get_tma_limits();
    cached.max_shared_mem_per_block = get_max_shared_mem_per_block();
    cudaDeviceGetAttribute(&cached.max_shared_mem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&cached.max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&cached.warp_size, cudaDevAttrWarpSize, 0);
    cached.kernel_parameter_limit = (props.major == 12 && props.minor >= 1) ? 32768 : 4096;

    #ifdef cudaDevAttrClusterLaunch
    int cluster_launch = 0;
    if (cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0) == cudaSuccess &&
        cluster_launch > 0) {
        cached.supports_clusters = true;
    }
    #endif

    if (!cached.supports_clusters) {
        if (props.major >= 10) {
            cached.max_cluster_size = 8;
            cached.supports_clusters = true;
        } else if (props.major == 9) {
            cached.max_cluster_size = 4;
            cached.supports_clusters = true;
        } else {
            cached.max_cluster_size = 1;
            cached.supports_clusters = false;
        }
    }

    if (props.major >= 12) {
        cached.tensor_core_gen = ArchitectureLimits::TensorCoreGeneration::Blackwell;
        cached.tensor_tile_m = 128;
        cached.tensor_tile_n = 128;
        cached.tensor_tile_k = 64;
    } else if (props.major >= 10) {
        cached.tensor_core_gen = ArchitectureLimits::TensorCoreGeneration::Blackwell;
        cached.tensor_tile_m = 128;
        cached.tensor_tile_n = 128;
        cached.tensor_tile_k = 64;
    } else if (props.major == 9) {
        cached.tensor_core_gen = ArchitectureLimits::TensorCoreGeneration::Hopper;
        cached.tensor_tile_m = 128;
        cached.tensor_tile_n = 128;
        cached.tensor_tile_k = 64;
    } else if (props.major >= 8) {
        cached.tensor_core_gen = ArchitectureLimits::TensorCoreGeneration::Ampere;
        cached.tensor_tile_m = 64;
        cached.tensor_tile_n = 64;
        cached.tensor_tile_k = 32;
    } else {
        cached.tensor_core_gen = ArchitectureLimits::TensorCoreGeneration::None;
        cached.tensor_tile_m = 64;
        cached.tensor_tile_n = 64;
        cached.tensor_tile_k = 16;
    }

    cached.has_grace_coherence = (props.major == 12 && props.minor >= 1);
    cached.has_nvlink_c2c = cached.has_grace_coherence;

    initialized = true;
    return cached;
}

inline TensorCoreTile select_tensor_core_tile() {
    const auto& limits = get_architecture_limits();
    using Gen = ArchitectureLimits::TensorCoreGeneration;

    switch (limits.tensor_core_gen) {
        case Gen::Blackwell:
            return {128, 128, 64};
        case Gen::Hopper:
            return {128, 128, 64};
        case Gen::Ampere:
            return {64, 64, 32};
        default:
            return {32, 32, 16};
    }
}

template <typename T>
inline int select_square_tile_size(int shared_tiles,
                                   std::initializer_list<int> candidates,
                                   bool enforce_thread_bound = false) {
    const auto& limits = get_architecture_limits();
    int fallback = *std::min_element(candidates.begin(), candidates.end());

    for (int candidate : candidates) {
        std::size_t shared_bytes =
            static_cast<std::size_t>(shared_tiles) *
            static_cast<std::size_t>(candidate) *
            static_cast<std::size_t>(candidate) *
            sizeof(T);

        bool fits_shared = shared_bytes <= static_cast<std::size_t>(limits.max_shared_mem_per_block);
        bool fits_threads = !enforce_thread_bound || (candidate * candidate <= limits.max_threads_per_block);

        if (fits_shared && fits_threads) {
            return candidate;
        }
    }

    return fallback;
}

}  // namespace cuda_arch

