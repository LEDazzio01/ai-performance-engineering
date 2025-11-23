#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

#include "../../common/headers/tma_helpers.cuh"
#ifdef prefetch
#undef prefetch
#endif
#include <cute/algorithm/copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>

namespace {

namespace cde = cuda::device::experimental;
using cuda_tma::device_supports_tma;
using cuda_tma::load_cuTensorMapEncodeTiled;
using cute::Int;

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int CHUNK_M = 32;
constexpr int PIPELINE_STAGES = 2;

#if CUDART_VERSION >= 13000
#define CAPSTONE3_TMA_AVAILABLE 1
#else
#define CAPSTONE3_TMA_AVAILABLE 0
#endif

constexpr bool kTmemAvailable =
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    true;
#else
    false;
#endif

__device__ void decode_tile_math(float* tile, int pitch, int rows, int cols) {
    for (int r = threadIdx.y; r < rows; r += blockDim.y) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float v = tile[r * pitch + c];
            tile[r * pitch + c] = v * 1.0002f + 0.0001f;
        }
    }
}

#if CAPSTONE3_TMA_AVAILABLE

bool encode_tensor_map_silent(
    CUtensorMap& desc,
    PFN_cuTensorMapEncodeTiled_v12000 encode,
    void* base,
    int width,
    int height,
    int ld,
    int box_width,
    int box_height,
    CUtensorMapSwizzle swizzle_mode) {
    constexpr uint32_t rank = 2;
    std::uint64_t dims[rank] = {
        static_cast<std::uint64_t>(width),
        static_cast<std::uint64_t>(height)};
    std::uint64_t stride[rank - 1] = {
        static_cast<std::uint64_t>(ld * sizeof(float))};
    std::uint32_t box[rank] = {
        static_cast<uint32_t>(box_width),
        static_cast<uint32_t>(box_height)};
    std::uint32_t elem_stride[rank] = {1, 1};

    constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    auto fn = encode ? encode : cuTensorMapEncodeTiled;
    CUresult res = fn(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        base,
        dims,
        stride,
        box,
        elem_stride,
        interleave,
        swizzle_mode,
        promotion,
        oob_fill);
    return res == CUDA_SUCCESS;
}

#endif  // CAPSTONE3_TMA_AVAILABLE

#if CAPSTONE3_TMA_AVAILABLE

template <int TILE_N_VALUE, int CHUNK_M_VALUE, int PIPELINE_STAGES_VALUE>
__global__ void tma_decode_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    float* __restrict__ output,
    int rows,
    int cols,
    int ld_output) {
    __shared__ alignas(128) float stage_buffers[PIPELINE_STAGES_VALUE][CHUNK_M_VALUE][TILE_N_VALUE];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char stage_barrier_storage[PIPELINE_STAGES_VALUE][sizeof(block_barrier)];

    constexpr std::size_t BYTES_PER_CHUNK =
        static_cast<std::size_t>(CHUNK_M_VALUE) * TILE_N_VALUE * sizeof(float);
    const int participants = blockDim.x * blockDim.y;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int stage = 0; stage < PIPELINE_STAGES_VALUE; ++stage) {
            auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
            init(bar_ptr, participants);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int row0 = tile_m * TILE_M;
    const int col0 = tile_n * TILE_N_VALUE;

    if (row0 >= rows || col0 >= cols) {
        return;
    }

    const int tile_rows = min(TILE_M, rows - row0);
        const int tile_cols = min(TILE_N_VALUE, cols - col0);
        const int num_chunks = (tile_rows + CHUNK_M_VALUE - 1) / CHUNK_M_VALUE;

    cuda::barrier<cuda::thread_scope_block>::arrival_token stage_tokens[PIPELINE_STAGES_VALUE];

    auto issue_chunk = [&](int chunk_idx) {
        if (chunk_idx >= num_chunks) {
            return;
        }
        const int stage = chunk_idx % PIPELINE_STAGES_VALUE;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        const int row_base = row0 + chunk_idx * CHUNK_M_VALUE;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                col0,
                row_base,
                bar);
            stage_tokens[stage] = cuda::device::barrier_arrive_tx(bar, 1, BYTES_PER_CHUNK);
        } else {
            stage_tokens[stage] = bar.arrive();
        }
    };

    const int preload = min(num_chunks, PIPELINE_STAGES_VALUE);
    for (int chunk = 0; chunk < preload; ++chunk) {
        issue_chunk(chunk);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int stage = chunk % PIPELINE_STAGES_VALUE;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        bar.wait(std::move(stage_tokens[stage]));
        __syncthreads();

        const int row_base = row0 + chunk * CHUNK_M_VALUE;
        const int rows_this_chunk = min(CHUNK_M_VALUE, tile_rows - chunk * CHUNK_M_VALUE);
        float* tile_ptr = &stage_buffers[stage][0][0];

        decode_tile_math(tile_ptr, TILE_N_VALUE, rows_this_chunk, tile_cols);
        cde::fence_proxy_async_shared_cta();
        __syncthreads();

        const bool full_cols = tile_cols == TILE_N_VALUE;
        const bool full_rows = (row_base + CHUNK_M_VALUE) <= rows;
        const bool contiguous_out = (ld_output == TILE_N_VALUE);
        const bool can_use_tmem = kTmemAvailable && full_cols && full_rows && contiguous_out;
        const bool can_use_tma_store = full_cols && full_rows;

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        if (can_use_tmem) {
            __shared__ uint32_t tmem_base_ptr;
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                cute::TMEM::Allocator1Sm allocator{};
                allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base_ptr);
            }
            __syncthreads();

            auto tmem_tensor = cute::make_tensor(
                cute::make_tmem_ptr<float>(tmem_base_ptr),
                cute::make_layout(
                    cute::make_shape(Int<CHUNK_M_VALUE>{}, Int<TILE_N_VALUE>{}),
                    cute::make_stride(cute::TMEM::DP<float>{}, Int<1>{})));

            auto smem_tensor = cute::make_tensor(
                cute::make_smem_ptr(&stage_buffers[stage][0][0]),
                cute::make_layout(
                    cute::make_shape(Int<CHUNK_M_VALUE>{}, Int<TILE_N_VALUE>{}),
                    cute::make_stride(Int<TILE_N_VALUE>{}, Int<1>{})));

            auto gmem_tensor = cute::make_tensor(
                cute::make_gmem_ptr(output + row_base * ld_output + col0),
                cute::make_layout(
                    cute::make_shape(Int<CHUNK_M_VALUE>{}, Int<TILE_N_VALUE>{}),
                    cute::make_stride(Int<TILE_N_VALUE>{}, Int<1>{})));

            auto tmem_store = cute::make_tmem_copy(cute::SM100_TMEM_STORE_32dp32b4x{}, tmem_tensor);
            auto tmem_load = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b4x{}, tmem_tensor);

            if (threadIdx.y == 0 && threadIdx.x < 32) {
                auto store_thr = tmem_store.get_slice(threadIdx.x);
                auto src = store_thr.partition_S(smem_tensor);
                auto dst = store_thr.partition_D(tmem_tensor);
                cute::copy(tmem_store, src, dst);
            }
            __syncthreads();
            if (threadIdx.y == 0 && threadIdx.x < 32) {
                auto load_thr = tmem_load.get_slice(threadIdx.x);
                auto src = load_thr.partition_S(tmem_tensor);
                auto dst = load_thr.partition_D(gmem_tensor);
                cute::copy(tmem_load, src, dst);
            }
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                cute::TMEM::Allocator1Sm allocator{};
                allocator.release_allocation_lock();
                allocator.free(tmem_base_ptr, cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
            }
        } else
#endif  // CUTE_ARCH_TCGEN05_TMEM_ENABLED
        if (can_use_tma_store) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                cde::cp_async_bulk_tensor_2d_shared_to_global(
                    &out_desc,
                    col0,
                    row_base,
                    &stage_buffers[stage]);
                cde::cp_async_bulk_commit_group();
                cde::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();
        } else {
            for (int r = threadIdx.y; r < rows_this_chunk; r += blockDim.y) {
                const int gr = row_base + r;
                if (gr >= rows) {
                    continue;
                }
                for (int c = threadIdx.x; c < tile_cols; c += blockDim.x) {
                    const int gc = col0 + c;
                    if (gc >= cols) {
                        continue;
                    }
                    output[gr * ld_output + gc] = tile_ptr[r * TILE_N_VALUE + c];
                }
            }
            __syncthreads();
        }

        const int next = chunk + PIPELINE_STAGES_VALUE;
        if (next < num_chunks) {
            issue_chunk(next);
        }
    }
}

#endif  // CAPSTONE3_TMA_AVAILABLE

void run_optimized_kernel(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "Output must be float32");
    TORCH_CHECK(input.sizes() == output.sizes(), "Input/output shapes must match");
    TORCH_CHECK(input.dim() == 2, "Decode kernel expects 2D tensors");

    auto in_contig = input.contiguous();
    auto out_contig = output.contiguous();

    const int rows = static_cast<int>(in_contig.size(0));
    const int cols = static_cast<int>(in_contig.size(1));
    if (rows == 0 || cols == 0) {
        return;
    }

    const int ld_input = static_cast<int>(in_contig.stride(0));
    const int ld_output = static_cast<int>(out_contig.stride(0));
    const float* input_ptr = in_contig.data_ptr<float>();
    float* output_ptr = out_contig.data_ptr<float>();

    dim3 block(32, 4, 1);
    dim3 grid(
        (cols + TILE_N - 1) / TILE_N,
        (rows + TILE_M - 1) / TILE_M,
        1);

    auto stream = at::cuda::getCurrentCUDAStream();

#if CAPSTONE3_TMA_AVAILABLE
    TORCH_CHECK(
        device_supports_tma(),
        "Tensor Memory Accelerator not supported on this GPU. "
        "Run the optimized decode kernel on Hopper/Blackwell hardware.");

    PFN_cuTensorMapEncodeTiled_v12000 encode = load_cuTensorMapEncodeTiled();
    TORCH_CHECK(encode, "cuTensorMapEncodeTiled entry point unavailable on this runtime");

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    TORCH_CHECK(
        encode_tensor_map_silent(
            in_desc,
            encode,
            const_cast<float*>(input_ptr),
            cols,
            rows,
            ld_input,
            TILE_N,
            CHUNK_M,
            CU_TENSOR_MAP_SWIZZLE_128B),
        "Failed to encode input tensor map for TMA decode kernel");
    TORCH_CHECK(
        encode_tensor_map_silent(
            out_desc,
            encode,
            output_ptr,
            cols,
            rows,
            ld_output,
            TILE_N,
            CHUNK_M,
            CU_TENSOR_MAP_SWIZZLE_128B),
        "Failed to encode output tensor map for TMA decode kernel");

    tma_decode_kernel<TILE_N, CHUNK_M, PIPELINE_STAGES><<<grid, block, 0, stream>>>(
        in_desc,
        out_desc,
        output_ptr,
        rows,
        cols,
        ld_output);
    AT_CUDA_CHECK(cudaGetLastError());

#else
    TORCH_CHECK(
        false,
        "labs/moe_cuda optimized decode kernel requires CUDA 13.0+ with "
        "Tensor Memory Accelerator support.");
#endif

    if (!output.is_contiguous()) {
        output.copy_(out_contig);
    }
}

#if CAPSTONE3_TMA_AVAILABLE

bool supports_tma_for_shape(int rows, int cols, int ld) {
    if (!device_supports_tma()) {
        return false;
    }
    PFN_cuTensorMapEncodeTiled_v12000 encode = load_cuTensorMapEncodeTiled();
    if (!encode) {
        return false;
    }
    float* buffer = nullptr;
    std::size_t bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(float);
    if (cudaMalloc(&buffer, bytes) != cudaSuccess) {
        return false;
    }
    CUtensorMap desc{};
    bool ok = encode_tensor_map_silent(
        desc,
        encode,
        buffer,
        cols,
        rows,
        ld,
        TILE_N,
        CHUNK_M,
        CU_TENSOR_MAP_SWIZZLE_128B);
    cudaFree(buffer);
    return ok;
}
#else
bool supports_tma_for_shape(int rows, int cols, int ld) {
    (void)rows;
    (void)cols;
    (void)ld;
    return false;
}
#endif

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_optimized",
        &run_optimized_kernel,
        "Optimized decode kernel (TMA double-buffered pipeline)");
    m.def(
        "supports_tma",
        &supports_tma_for_shape,
        "Return True if the GPU/runtime can execute the TMA decode kernel");
}
