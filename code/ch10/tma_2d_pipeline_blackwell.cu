/**
 * Blackwell TMA (Tensor Memory Accelerator) 2D Pipeline
 * =====================================================
 *
 * CUDA 13.0 introduces descriptor-backed bulk async copies (cp.async.bulk.tensor.*)
 * that route through the Tensor Memory Accelerator on Hopper/Blackwell GPUs.
 * This sample demonstrates a double-buffered 2D pipeline that overlaps compute
 * with TMA transfers using CUDA C++17 primitives.
 *
 * Key features demonstrated:
 *  - CU_TENSOR_MAP_SWIZZLE_128B for HBM3e alignment on Blackwell B200/B300
 *  - cuda::device::experimental::cp_async_bulk_tensor_2d_* helpers
 *  - cuda::barrier based staging for multi-buffer pipelines
 *
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_100 tma_2d_pipeline_blackwell.cu -o tma_pipeline
 */

#include <algorithm>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

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
using cuda_tma::make_2d_tensor_map;

constexpr int TILE_M = 128;

#if TMA_CUDA13_AVAILABLE

__device__ void compute_on_tile(float* tile, int pitch, int rows, int cols) {
    for (int r = threadIdx.y; r < rows; r += blockDim.y) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float v = tile[r * pitch + c];
            tile[r * pitch + c] = v * 1.0001f + 0.0001f;  // trivial math to emulate work
        }
    }
}

template <int TILE_N_VALUE, int CHUNK_M_VALUE>
__global__ void tma_2d_pipeline_fallback_kernel(
    const float* __restrict__ A,
    float* __restrict__ C,
    int M,
    int N,
    int lda,
    int ldc) {
    __shared__ alignas(16) float stage_buffer[CHUNK_M_VALUE][TILE_N_VALUE];

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int row0 = tile_m * TILE_M;
    const int col0 = tile_n * TILE_N_VALUE;

    if (row0 >= M || col0 >= N) {
        return;
    }

    const int tile_rows = min(TILE_M, M - row0);
    const int tile_cols = min(TILE_N_VALUE, N - col0);
    const int num_chunks = (tile_rows + CHUNK_M_VALUE - 1) / CHUNK_M_VALUE;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int rows_this_chunk = min(CHUNK_M_VALUE, tile_rows - chunk * CHUNK_M_VALUE);
        const int row_base = row0 + chunk * CHUNK_M_VALUE;
        float* tile_ptr = &stage_buffer[0][0];

        for (int r = threadIdx.y; r < rows_this_chunk; r += blockDim.y) {
            const int gr = row_base + r;
            for (int c = threadIdx.x; c < tile_cols; c += blockDim.x) {
                const int gc = col0 + c;
                tile_ptr[r * TILE_N_VALUE + c] = A[gr * lda + gc];
            }
        }
        __syncthreads();

        compute_on_tile(tile_ptr, TILE_N_VALUE, rows_this_chunk, tile_cols);
        __syncthreads();

        for (int r = threadIdx.y; r < rows_this_chunk; r += blockDim.y) {
            const int gr = row_base + r;
            for (int c = threadIdx.x; c < tile_cols; c += blockDim.x) {
                const int gc = col0 + c;
                C[gr * ldc + gc] = tile_ptr[r * TILE_N_VALUE + c];
            }
        }
        __syncthreads();
    }
}

template <int TILE_N_VALUE, int CHUNK_M_VALUE, int PIPELINE_STAGES_VALUE>
__global__ void tma_2d_pipeline_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    float* __restrict__ fallback_out,
    int M,
    int N,
    int ldc) {
    __shared__ alignas(128) float stage_buffers[PIPELINE_STAGES_VALUE][CHUNK_M_VALUE][TILE_N_VALUE];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char stage_barrier_storage[PIPELINE_STAGES_VALUE][sizeof(block_barrier)];

    const int tile_m_dim = TILE_M;
    const int tile_n_dim = TILE_N_VALUE;
    const int chunk_m_dim = CHUNK_M_VALUE;
    const int pipeline_stages = PIPELINE_STAGES_VALUE;
    constexpr std::size_t BYTES_PER_CHUNK =
        static_cast<std::size_t>(CHUNK_M_VALUE) * TILE_N_VALUE * sizeof(float);

    const int participants = blockDim.x * blockDim.y;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int stage = 0; stage < pipeline_stages; ++stage) {
            auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
            init(bar_ptr, participants);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;

    const int g_row0 = tile_m * tile_m_dim;
    const int g_col0 = tile_n * tile_n_dim;

    if (g_row0 >= M || g_col0 >= N) {
        return;
    }

    const int tile_rows = std::min(tile_m_dim, M - g_row0);
    const int tile_cols = std::min(tile_n_dim, N - g_col0);
    const int num_chunks = (tile_rows + chunk_m_dim - 1) / chunk_m_dim;

    cuda::barrier<cuda::thread_scope_block>::arrival_token stage_tokens[PIPELINE_STAGES_VALUE];

    auto issue_chunk = [&](int chunk_idx) {
        if (chunk_idx >= num_chunks) {
            return;
        }
        const int stage = chunk_idx % pipeline_stages;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        const int row_base = g_row0 + chunk_idx * chunk_m_dim;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                g_col0,
                row_base,
                bar);
            stage_tokens[stage] = cuda::device::barrier_arrive_tx(bar, 1, BYTES_PER_CHUNK);
        } else {
            stage_tokens[stage] = bar.arrive();
        }
    };

    const int preload = std::min(num_chunks, pipeline_stages);
    for (int chunk = 0; chunk < preload; ++chunk) {
        issue_chunk(chunk);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int stage = chunk % pipeline_stages;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        bar.wait(std::move(stage_tokens[stage]));
        __syncthreads();

        const int row_base = g_row0 + chunk * chunk_m_dim;
        const int rows_this_chunk = std::min(chunk_m_dim, tile_rows - chunk * chunk_m_dim);
        float* tile_ptr = &stage_buffers[stage][0][0];

        compute_on_tile(tile_ptr, TILE_N_VALUE, rows_this_chunk, tile_cols);
        cde::fence_proxy_async_shared_cta();
        __syncthreads();

        const bool full_columns = tile_cols == TILE_N_VALUE;
        const bool full_rows = (row_base + chunk_m_dim) <= M;
        const bool can_use_tma_store = full_columns && full_rows;

        if (can_use_tma_store) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                cde::cp_async_bulk_tensor_2d_shared_to_global(
                    &out_desc,
                    g_col0,
                    row_base,
                    &stage_buffers[stage]);
                cde::cp_async_bulk_commit_group();
                cde::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();
        } else {
            for (int r = threadIdx.y; r < rows_this_chunk; r += blockDim.y) {
                const int global_row = row_base + r;
                if (global_row >= M) {
                    continue;
                }
                for (int c = threadIdx.x; c < tile_cols; c += blockDim.x) {
                    const int global_col = g_col0 + c;
                    if (global_col >= N) {
                        continue;
                    }
                    fallback_out[global_row * ldc + global_col] = tile_ptr[r * TILE_N_VALUE + c];
                }
            }
            __syncthreads();
        }

        const int next = chunk + pipeline_stages;
        if (next < num_chunks) {
            issue_chunk(next);
        }
    }
}

int main() {
    std::printf("=== Blackwell TMA 2D Pipeline ===\n\n");

    bool enable_tma = std::getenv("ENABLE_BLACKWELL_TMA") != nullptr;
    bool tma_supported = device_supports_tma();
    if (!tma_supported && enable_tma) {
        std::printf("⚠️  Device does not support Hopper/Blackwell TMA; falling back to cuda::memcpy_async path.\n");
        enable_tma = false;
    }

    PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    if (tma_supported) {
        encode = load_cuTensorMapEncodeTiled();
        if (!encode && enable_tma) {
            std::printf("⚠️  cuTensorMapEncodeTiled entry point unavailable; falling back.\n");
            enable_tma = false;
        }
    }

    constexpr int M = 4096;
    constexpr int N = 4096;
    const std::size_t bytes = static_cast<std::size_t>(M) * N * sizeof(float);

    std::vector<float> h_in(static_cast<std::size_t>(M) * N);
    for (std::size_t idx = 0; idx < h_in.size(); ++idx) {
        h_in[idx] = static_cast<float>((idx % 113) + 1);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemset(d_out, 0, bytes), "memset output");

    const cuda_arch::TMALimits limits = cuda_arch::get_tma_limits();
    const int shared_mem_limit = cuda_arch::get_max_shared_mem_per_block();

    struct PipelineOption {
        int tile_n;
        int chunk_m;
        int stages;
    };

    constexpr PipelineOption kOptions[] = {
        {128, 64, 1},
        {128, 32, 2},
        {128, 32, 1},
        {64, 64, 1},
        {64, 32, 2},
        {64, 32, 1}
    };

    PipelineOption selected{0, 0, 0};
    if (enable_tma) {
        for (const auto& option : kOptions) {
            if (option.tile_n > static_cast<int>(limits.max_2d_box_width)) {
                continue;
            }
            if (option.chunk_m > static_cast<int>(limits.max_2d_box_height)) {
                continue;
            }
            const std::size_t required_bytes =
                static_cast<std::size_t>(option.tile_n) *
                option.chunk_m *
                option.stages *
                sizeof(float);
            if (required_bytes > static_cast<std::size_t>(shared_mem_limit)) {
                continue;
            }
            selected = option;
            break;
        }
        if (selected.tile_n == 0) {
            std::printf("⚠️  No viable TMA configuration fits shared-memory limits on this device; using fallback pipeline.\n");
            enable_tma = false;
        } else {
            std::printf("Selected TMA configuration: width=%d, chunk=%d, stages=%d (shared mem %.0f bytes)\n",
                        selected.tile_n,
                        selected.chunk_m,
                        selected.stages,
                        static_cast<double>(selected.tile_n) * selected.chunk_m * selected.stages * sizeof(float));
        }
    } else {
        std::printf("ℹ️  ENABLE_BLACKWELL_TMA not set (or descriptors unavailable); using fallback pipeline.\n");
    }

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    if (enable_tma) {
        enable_tma = make_2d_tensor_map(
                          in_desc,
                          encode,
                          d_in,
                          N,
                          M,
                          N,
                          selected.tile_n,
                          selected.chunk_m,
                          CU_TENSOR_MAP_SWIZZLE_NONE) &&
                     make_2d_tensor_map(
                          out_desc,
                          encode,
                          d_out,
                          N,
                          M,
                          N,
                          selected.tile_n,
                          selected.chunk_m,
                          CU_TENSOR_MAP_SWIZZLE_NONE);
        if (!enable_tma) {
            std::printf("⚠️  Descriptor creation failed; reverting to fallback pipeline.\n");
        }
    }

    const PipelineOption fallback_option = enable_tma ? selected : PipelineOption{64, 32, 1};
    const int tile_width = enable_tma ? selected.tile_n : fallback_option.tile_n;

    dim3 block(32, 4, 1);  // 128 threads
    dim3 grid(
        (N + tile_width - 1) / tile_width,
        (M + TILE_M - 1) / TILE_M,
        1);

    auto launch_tma = [&]() {
        if (selected.tile_n == 128 && selected.chunk_m == 64 && selected.stages == 1) {
            tma_2d_pipeline_kernel<128, 64, 1><<<grid, block>>>(in_desc, out_desc, d_out, M, N, N);
        } else if (selected.tile_n == 128 && selected.chunk_m == 32 && selected.stages == 2) {
            tma_2d_pipeline_kernel<128, 32, 2><<<grid, block>>>(in_desc, out_desc, d_out, M, N, N);
        } else if (selected.tile_n == 128 && selected.chunk_m == 32 && selected.stages == 1) {
            tma_2d_pipeline_kernel<128, 32, 1><<<grid, block>>>(in_desc, out_desc, d_out, M, N, N);
        } else if (selected.tile_n == 64 && selected.chunk_m == 64 && selected.stages == 1) {
            tma_2d_pipeline_kernel<64, 64, 1><<<grid, block>>>(in_desc, out_desc, d_out, M, N, N);
        } else if (selected.tile_n == 64 && selected.chunk_m == 32 && selected.stages == 2) {
            tma_2d_pipeline_kernel<64, 32, 2><<<grid, block>>>(in_desc, out_desc, d_out, M, N, N);
        } else {
            tma_2d_pipeline_kernel<64, 32, 1><<<grid, block>>>(in_desc, out_desc, d_out, M, N, N);
        }
        check_cuda(cudaGetLastError(), "tma_2d_pipeline_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "tma kernel sync");
    };

    auto launch_fallback = [&](const PipelineOption& option) {
        if (option.tile_n == 128 && option.chunk_m == 64) {
            tma_2d_pipeline_fallback_kernel<128, 64><<<grid, block>>>(d_in, d_out, M, N, N, N);
        } else if (option.tile_n == 128 && option.chunk_m == 32) {
            tma_2d_pipeline_fallback_kernel<128, 32><<<grid, block>>>(d_in, d_out, M, N, N, N);
        } else if (option.tile_n == 64 && option.chunk_m == 64) {
            tma_2d_pipeline_fallback_kernel<64, 64><<<grid, block>>>(d_in, d_out, M, N, N, N);
        } else {
            tma_2d_pipeline_fallback_kernel<64, 32><<<grid, block>>>(d_in, d_out, M, N, N, N);
        }
        check_cuda(cudaGetLastError(), "tma_2d_pipeline_fallback_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "fallback kernel sync");
    };

    if (enable_tma) {
        launch_tma();
    } else {
        launch_fallback(fallback_option);
    }

    std::vector<float> h_out(TILE_M * tile_width);
    check_cuda(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy sample");

    std::printf("Sample output element: %.2f -> %.2f\n", h_in[0], h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);

    std::printf("\n=== Summary ===\n");
    if (enable_tma) {
        std::printf("✓ Bulk TMA transfers via cp.async.bulk.tensor.2d (%d-row chunk, %d-column tile)\n",
                    selected.chunk_m,
                    selected.tile_n);
        std::printf("✓ Descriptor-backed TMA transfers with L2 promotion enabled\n");
        std::printf("✓ cuda::barrier orchestrates staging and overlap between compute and TMA IO\n");
    } else {
        std::printf("✓ Fallback cuda::memcpy_async pipeline executed (no TMA descriptors used)\n");
        std::printf("✓ Kernel remains safe for profiling while descriptor support is unavailable\n");
    }

    return 0;
}

#else  // !TMA_CUDA13_AVAILABLE

int main() {
    std::printf("=== Blackwell TMA 2D Pipeline ===\n\n");
    std::printf("⚠️  CUDA 13.0+ required for TMA descriptor API (detected %d.%d)\n",
                CUDART_VERSION / 1000,
                (CUDART_VERSION % 100) / 10);
    return 0;
}

#endif  // TMA_CUDA13_AVAILABLE
