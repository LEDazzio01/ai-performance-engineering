// async_prefetch_tma.cu -- double-buffered 1D streaming with CUDA 13 TMA

#include <algorithm>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "../common/headers/arch_detection.cuh"
#include "../common/headers/tma_helpers.cuh"

#if CUDART_VERSION >= 13000
#include <cuda.h>
#define TMA_CUDA13_AVAILABLE 1
#else
#define TMA_CUDA13_AVAILABLE 0
#endif

namespace cde = cuda::device::experimental;
using cuda_tma::check_cuda;
using cuda_tma::device_supports_tma;
using cuda_tma::load_cuTensorMapEncodeTiled;
using cuda_tma::make_1d_tensor_map;

constexpr int PIPELINE_STAGES = 2;

template <int TILE_SIZE>
__global__ void async_prefetch_fallback_kernel(
    const float* data,
    float* out,
    int tiles) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;

    for (int t = 0; t < tiles; ++t) {
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            smem[i] = data[t * TILE_SIZE + i];
        }
        __syncthreads();

        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            float v = smem[i] * 2.0f;
            smem[i] = v;
            out[t * TILE_SIZE + i] = v;
        }
        __syncthreads();
    }
}

#if TMA_CUDA13_AVAILABLE

template <int TILE_SIZE>
__global__ void async_prefetch_tma_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    int total_tiles) {
    __shared__ alignas(128) float stage_buffers[PIPELINE_STAGES][TILE_SIZE];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char stage_barrier_storage[PIPELINE_STAGES][sizeof(block_barrier)];

    const int pipeline_stages = PIPELINE_STAGES;

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < pipeline_stages; ++stage) {
            auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
            init(bar_ptr, blockDim.x);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    cuda::barrier<cuda::thread_scope_block>::arrival_token tokens[PIPELINE_STAGES];

    auto issue_tile = [&](int tile_idx) {
        if (tile_idx >= total_tiles) {
            return;
        }
        const int stage = tile_idx % pipeline_stages;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                tile_idx * TILE_SIZE,
                bar);
            tokens[stage] = cuda::device::barrier_arrive_tx(
                bar,
                1,
                static_cast<std::size_t>(TILE_SIZE) * sizeof(float));
        } else {
            tokens[stage] = bar.arrive();
        }
    };

    const int preload = std::min(total_tiles, pipeline_stages);
    for (int t = 0; t < preload; ++t) {
        issue_tile(t);
    }

    for (int tile = 0; tile < total_tiles; ++tile) {
        const int stage = tile % pipeline_stages;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        bar.wait(std::move(tokens[stage]));
        __syncthreads();

        float* tile_ptr = stage_buffers[stage];
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            tile_ptr[i] *= 2.0f;
        }
        cde::fence_proxy_async_shared_cta();
        __syncthreads();

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_shared_to_global(
                &out_desc,
                tile * TILE_SIZE,
                &stage_buffers[stage]);
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        const int next_tile = tile + pipeline_stages;
        if (next_tile < total_tiles) {
            issue_tile(next_tile);
        }
    }

}

template <int TILE_SIZE>
void launch_tma_prefetch_kernel(
    const CUtensorMap& in_desc,
    const CUtensorMap& out_desc,
    int tiles) {
    async_prefetch_tma_kernel<TILE_SIZE><<<1, 256>>>(in_desc, out_desc, tiles);
    check_cuda(cudaGetLastError(), "async_prefetch_tma_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");
}

template <int TILE_SIZE>
void launch_fallback_kernel(
    const float* d_in,
    float* d_out,
    int tiles) {
    const int threads = 256;
    const std::size_t smem_bytes = static_cast<std::size_t>(TILE_SIZE) * sizeof(float);
    async_prefetch_fallback_kernel<TILE_SIZE><<<1, threads, smem_bytes>>>(d_in, d_out, tiles);
    check_cuda(cudaGetLastError(), "async_prefetch_fallback_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "fallback kernel sync");
}

template <typename LaunchFn256, typename LaunchFn512, typename LaunchFn1024>
void dispatch_by_tile_size(int tile_size, LaunchFn256 launch256, LaunchFn512 launch512, LaunchFn1024 launch1024) {
    switch (tile_size) {
        case 1024:
            launch1024();
            break;
        case 512:
            launch512();
            break;
        case 256:
        default:
            launch256();
            break;
    }
}

int main() {
    std::printf("=== CUDA 13 TMA 1D Prefetch ===\n\n");

    bool enable_tma = std::getenv("ENABLE_BLACKWELL_TMA") != nullptr;
    const bool tma_supported = device_supports_tma();
    if (!tma_supported && enable_tma) {
        std::printf("⚠️  Device does not support Hopper/Blackwell TMA; using fallback path.\n");
        enable_tma = false;
    }

    PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    if (tma_supported) {
        encode = load_cuTensorMapEncodeTiled();
        if (!encode && enable_tma) {
            std::printf("⚠️  cuTensorMapEncodeTiled entry point unavailable; using fallback path.\n");
            enable_tma = false;
        }
    }

    if (!enable_tma) {
        std::printf("ℹ️  ENABLE_BLACKWELL_TMA not set (or descriptor API unavailable); using cuda::memcpy_async fallback.\n");
    }

    const cuda_arch::TMALimits limits = cuda_arch::get_tma_limits();
    const int tile_size = static_cast<int>(std::min<uint32_t>(limits.max_1d_box_size, 1024u));
    if (tile_size <= 0) {
        std::printf("⚠️  Invalid TMA limits reported; falling back to cuda::memcpy_async.\n");
        enable_tma = false;
    }
    std::printf("Selected TMA tile size: %d element(s)\n\n", tile_size);

    constexpr int kTiles = 64;
    const int total_elements = tile_size * kTiles;
    const std::size_t bytes = static_cast<std::size_t>(total_elements) * sizeof(float);

    float* h_in = nullptr;
    float* h_out = nullptr;
    check_cuda(cudaMallocHost(&h_in, bytes), "cudaMallocHost h_in");
    check_cuda(cudaMallocHost(&h_out, bytes), "cudaMallocHost h_out");
    for (int i = 0; i < total_elements; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "copy input");

    if (enable_tma && !encode) {
        enable_tma = false;
    }
    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    if (enable_tma) {
        enable_tma = make_1d_tensor_map(in_desc, encode, d_in, total_elements, tile_size) &&
                     make_1d_tensor_map(out_desc, encode, d_out, total_elements, tile_size);
        if (!enable_tma) {
            std::printf("⚠️  Falling back to cuda::memcpy_async implementation (TMA descriptor unavailable).\n");
        }
    }

    auto launch_tma = [&]() {
        dispatch_by_tile_size(
            tile_size,
            [&]() { launch_tma_prefetch_kernel<256>(in_desc, out_desc, kTiles); },
            [&]() { launch_tma_prefetch_kernel<512>(in_desc, out_desc, kTiles); },
            [&]() { launch_tma_prefetch_kernel<1024>(in_desc, out_desc, kTiles); });
    };

    auto launch_fallback = [&]() {
        dispatch_by_tile_size(
            tile_size,
            [&]() { launch_fallback_kernel<256>(d_in, d_out, kTiles); },
            [&]() { launch_fallback_kernel<512>(d_in, d_out, kTiles); },
            [&]() { launch_fallback_kernel<1024>(d_in, d_out, kTiles); });
    };

    if (enable_tma) {
        launch_tma();
    } else {
        launch_fallback();
    }

    check_cuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "copy output");
    std::printf("out[0]=%.1f (expected %.1f)\n", h_out[0], h_in[0] * 2.0f);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}

#else  // !TMA_CUDA13_AVAILABLE

int main() {
    std::printf("=== CUDA 13 TMA 1D Prefetch ===\n\n");
    std::printf("⚠️  CUDA 13.0+ required for TMA descriptor API (detected %d.%d)\n",
                CUDART_VERSION / 1000,
                (CUDART_VERSION % 100) / 10);
    return 0;
}

#endif  // TMA_CUDA13_AVAILABLE
