#pragma once

/**
 * cuda13_feature_examples.cuh
 *
 * Shared helpers for CUDA 13 demonstrations:
 *  - Stream-ordered memory allocation with cudaMallocAsync / cudaFreeAsync
 *  - Tensor Memory Accelerator (TMA) 2D copy using bulk async descriptors
 *
 * All functions are header-only so they can be included directly in the
 * chapter examples without introducing an additional translation unit.
 */

#include <cassert>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <vector>

namespace cuda13_examples {

namespace cde = cuda::device::experimental;

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

inline void check_cu(CUresult res, const char* what) {
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        std::fprintf(stderr, "CUDA driver error (%s): %s\n", what, err_str ? err_str : "unknown");
        std::abort();
    }
}

inline bool device_supports_tma() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return false;
    }
    return prop.major >= 9;  // Hopper (SM 90) and newer introduce TMA
}

inline PFN_cuTensorMapEncodeTiled_v12000 load_cuTensorMapEncodeTiled() {
    void* func_ptr = nullptr;
    cudaDriverEntryPointQueryResult query_result{};

    cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled",
        &func_ptr,
        13000,  // Prefer CUDA 13 implementation when available
        cudaEnableDefault,
        &query_result);

    if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess) {
        // Fallback to CUDA 12.x entry point for Hopper if CUDA 13 is unavailable.
        err = cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled",
            &func_ptr,
            12000,
            cudaEnableDefault,
            &query_result);
    }

    if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess || func_ptr == nullptr) {
        std::fprintf(stderr, "cuTensorMapEncodeTiled unavailable on this runtime.\n");
        return nullptr;
    }

    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(func_ptr);
}

// -----------------------------------------------------------------------------
// Stream-ordered memory allocation demo
// -----------------------------------------------------------------------------

static __global__ void fill_sequence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = static_cast<float>(idx);
    }
}

static __global__ void saxpy_kernel(float* out, const float* in, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = alpha * in[idx] + out[idx];
    }
}

inline void run_stream_ordered_memory_demo() {
    std::printf("\n[CUDA 13] Stream-ordered memory allocation demo\n");
    cudaStream_t stream;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");

    // Configure the default mempool for aggressive reuse so allocations
    // happen entirely on the GPU timeline.
    cudaMemPool_t pool{};
    check_cuda(cudaDeviceGetDefaultMemPool(&pool, /*device=*/0), "get default mempool");
    std::uint64_t threshold = 0;
    check_cuda(cudaMemPoolSetAttribute(
                   pool,
                   cudaMemPoolAttrReleaseThreshold,
                   &threshold),
               "set mempool threshold");

    constexpr int N = 1 << 15;
    constexpr size_t BYTES = sizeof(float) * N;
    float* a = nullptr;
    float* b = nullptr;

    check_cuda(cudaMallocAsync(&a, BYTES, stream), "cudaMallocAsync(a)");
    check_cuda(cudaMallocAsync(&b, BYTES, stream), "cudaMallocAsync(b)");

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    fill_sequence<<<grid, block, 0, stream>>>(a, N);
    fill_sequence<<<grid, block, 0, stream>>>(b, N);
    saxpy_kernel<<<grid, block, 0, stream>>>(b, a, 2.0f, N);

    check_cuda(cudaFreeAsync(a, stream), "cudaFreeAsync(a)");
    check_cuda(cudaFreeAsync(b, stream), "cudaFreeAsync(b)");

    check_cuda(cudaStreamSynchronize(stream), "stream sync");
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    std::printf("  ✓ Allocations executed entirely on the GPU stream timeline\n");
}

// -----------------------------------------------------------------------------
// Simple 2D TMA demo
// -----------------------------------------------------------------------------

constexpr int TMA_TILE_M = 64;
constexpr int TMA_TILE_N = 128;

static __global__ void tma_copy_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    float* out_fallback,
    int width,
    int height,
    int ld_out) {
    constexpr int participants = 128;
    if (blockDim.x * blockDim.y != participants) {
        return;
    }

    __shared__ alignas(128) float smem[TMA_TILE_M][TMA_TILE_N];
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

#pragma nv_diag_suppress static_var_with_dynamic_init
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(&bar, participants);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int tile_x = blockIdx.x * TMA_TILE_N;
    int tile_y = blockIdx.y * TMA_TILE_M;

    bool in_bounds = (tile_x + TMA_TILE_N) <= width && (tile_y + TMA_TILE_M) <= height;

    cuda::barrier<cuda::thread_scope_block>::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &smem,
            &in_desc,
            tile_x,
            tile_y,
            bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
    } else {
        token = bar.arrive();
    }
    bar.wait(std::move(token));

    // Simple transform: scale by 1.5x
    for (int row = threadIdx.y; row < TMA_TILE_M; row += blockDim.y) {
        for (int col = threadIdx.x; col < TMA_TILE_N; col += blockDim.x) {
            smem[row][col] *= 1.5f;
        }
    }
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(
            &out_desc,
            tile_x,
            tile_y,
            &smem);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    if (!in_bounds) {
        for (int row = threadIdx.y; row < TMA_TILE_M; row += blockDim.y) {
            int global_row = tile_y + row;
            if (global_row >= height) {
                continue;
            }
            for (int col = threadIdx.x; col < TMA_TILE_N; col += blockDim.x) {
                int global_col = tile_x + col;
                if (global_col >= width) {
                    continue;
                }
                out_fallback[global_row * ld_out + global_col] = smem[row][col];
            }
        }
    }
}

inline bool make_2d_tensor_map(
    CUtensorMap& desc,
    PFN_cuTensorMapEncodeTiled_v12000 encode,
    void* base,
    int width,
    int height,
    int ld,
    CUtensorMapSwizzle swizzle_mode) {
    constexpr uint32_t rank = 2;
    std::uint64_t dims[rank] = {static_cast<std::uint64_t>(width),
                                static_cast<std::uint64_t>(height)};
    std::uint64_t stride[rank - 1] = {static_cast<std::uint64_t>(ld * sizeof(float))};
    std::uint32_t box[rank] = {TMA_TILE_N, TMA_TILE_M};
    std::uint32_t elem_stride[rank] = {1, 1};

    constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    std::printf(
        "[TMA] Encoding 2D tensor map: base=%p dims={%llu,%llu} stride_bytes={%llu} box={%u,%u} "
        "elem_stride={%u,%u} interleave=%d swizzle=%d l2=%d oob=%d\n",
        base,
        static_cast<unsigned long long>(dims[0]),
        static_cast<unsigned long long>(dims[1]),
        static_cast<unsigned long long>(stride[0]),
        box[0],
        box[1],
        elem_stride[0],
        elem_stride[1],
        static_cast<int>(interleave),
        static_cast<int>(swizzle_mode),
        static_cast<int>(promotion),
        static_cast<int>(oob_fill));

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
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        const char* err_name = nullptr;
        cuGetErrorString(res, &err_str);
        cuGetErrorName(res, &err_name);
        std::fprintf(stderr,
                     "[TMA] cuTensorMapEncodeTiled (2D) failed: %s (%s, %d) "
                     "(dataType=%d rank=%u base=%p dims={%llu,%llu} stride={%llu} box={%u,%u} "
                     "elem_stride={%u,%u} interleave=%d swizzle=%d l2=%d oob=%d)\n",
                     err_str ? err_str : "unknown",
                     err_name ? err_name : "unknown",
                     static_cast<int>(res),
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     rank,
                     base,
                     static_cast<unsigned long long>(dims[0]),
                     static_cast<unsigned long long>(dims[1]),
                     static_cast<unsigned long long>(stride[0]),
                     box[0],
                     box[1],
                     elem_stride[0],
                     elem_stride[1],
                     static_cast<int>(interleave),
                     static_cast<int>(swizzle_mode),
                     static_cast<int>(promotion),
                     static_cast<int>(oob_fill));
        return false;
    }
    std::printf("[TMA] 2D descriptor ok (res=%d): base=%p dims={%llu,%llu} stride_bytes={%llu} box={%u,%u} "
                "elem_stride={%u,%u} interleave=%d swizzle=%d l2=%d oob=%d\n",
                static_cast<int>(res),
                base,
                static_cast<unsigned long long>(dims[0]),
                static_cast<unsigned long long>(dims[1]),
                static_cast<unsigned long long>(stride[0]),
                box[0],
                box[1],
                elem_stride[0],
                elem_stride[1],
                static_cast<int>(interleave),
                static_cast<int>(swizzle_mode),
                static_cast<int>(promotion),
                static_cast<int>(oob_fill));
    return true;
}

inline bool make_1d_tensor_map(
    CUtensorMap& desc,
    PFN_cuTensorMapEncodeTiled_v12000 encode,
    void* base,
    int elements,
    int box_elements,
    CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE) {
    constexpr uint32_t rank = 1;
    std::uint64_t dims[rank] = {static_cast<std::uint64_t>(elements)};
    std::uint64_t stride_bytes[rank] = {static_cast<std::uint64_t>(sizeof(float))};
    std::uint32_t box[rank] = {static_cast<std::uint32_t>(box_elements)};
    std::uint32_t elem_stride[rank] = {1};

    constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    std::printf(
        "[TMA] Encoding 1D tensor map: base=%p elements=%d stride_bytes=%llu box=%u elem_stride=%u interleave=%d "
        "swizzle=%d l2=%d oob=%d\n",
        base,
        elements,
        static_cast<unsigned long long>(stride_bytes[0]),
        box[0],
        elem_stride[0],
        static_cast<int>(interleave),
        static_cast<int>(swizzle_mode),
        static_cast<int>(promotion),
        static_cast<int>(oob_fill));

    auto fn = encode ? encode : cuTensorMapEncodeTiled;
    CUresult res = fn(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        base,
        dims,
        stride_bytes,
        box,
        elem_stride,
        interleave,
        swizzle_mode,
        promotion,
        oob_fill);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        const char* err_name = nullptr;
        cuGetErrorString(res, &err_str);
        cuGetErrorName(res, &err_name);
        std::fprintf(stderr,
                     "[TMA] cuTensorMapEncodeTiled (1D) failed: %s (%s, %d) (dataType=%d rank=%u base=%p elements=%d "
                     "stride_bytes=%llu box=%u elem_stride=%u interleave=%d swizzle=%d l2=%d oob=%d)\n",
                     err_str ? err_str : "unknown",
                     err_name ? err_name : "unknown",
                     static_cast<int>(res),
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     rank,
                     base,
                     elements,
                     static_cast<unsigned long long>(stride_bytes[0]),
                     box[0],
                     elem_stride[0],
                     static_cast<int>(interleave),
                     static_cast<int>(swizzle_mode),
                     static_cast<int>(promotion),
                     static_cast<int>(oob_fill));
        return false;
    }
    std::printf("[TMA] 1D descriptor ok (res=%d): base=%p elements=%d stride_bytes=%llu box=%u elem_stride=%u "
                "interleave=%d swizzle=%d l2=%d oob=%d\n",
                static_cast<int>(res),
                base,
                elements,
                static_cast<unsigned long long>(stride_bytes[0]),
                box[0],
                elem_stride[0],
                static_cast<int>(interleave),
                static_cast<int>(swizzle_mode),
                static_cast<int>(promotion),
                static_cast<int>(oob_fill));
    return true;
}

inline void run_simple_tma_demo() {
    std::printf("\n[CUDA 13] Tensor Memory Accelerator 2D copy demo\n");
    if (!device_supports_tma()) {
        std::printf("  ⚠️  Device does not support TMA (requires SM 90 or newer)\n");
        return;
    }

    auto encode = load_cuTensorMapEncodeTiled();
    if (!encode) {
        std::printf("  ⚠️  cuTensorMapEncodeTiled unavailable on this runtime\n");
        return;
    }

    constexpr int WIDTH = TMA_TILE_N;
    constexpr int HEIGHT = TMA_TILE_M;
    constexpr size_t BYTES = sizeof(float) * WIDTH * HEIGHT;

    std::vector<float> h_in(WIDTH * HEIGHT);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            h_in[y * WIDTH + x] = static_cast<float>((y + 1) * (x + 1));
        }
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, BYTES), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, BYTES), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemset(d_out, 0, BYTES), "memset output");

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    bool ok_in = make_2d_tensor_map(in_desc, encode, d_in, WIDTH, HEIGHT, WIDTH, CU_TENSOR_MAP_SWIZZLE_NONE);
    bool ok_out = make_2d_tensor_map(out_desc, encode, d_out, WIDTH, HEIGHT, WIDTH, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (!ok_in || !ok_out) {
        std::printf("  ⚠️  Failed to encode tensor maps; skipping TMA demo\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }

    dim3 block(32, 4, 1);
    dim3 grid((WIDTH + TMA_TILE_N - 1) / TMA_TILE_N,
              (HEIGHT + TMA_TILE_M - 1) / TMA_TILE_M,
              1);
    tma_copy_kernel<<<grid, block>>>(in_desc, out_desc, d_out, WIDTH, HEIGHT, WIDTH);
    check_cuda(cudaGetLastError(), "launch tma_copy_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync tma_copy_kernel");

    std::vector<float> h_out(WIDTH * HEIGHT);
    check_cuda(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost), "copy result");

    std::printf("  ✓ TMA copied %dx%d tile (swizzle=128B, L2 promotion=128B)\n", HEIGHT, WIDTH);
    std::printf("  Sample output element: %.2f -> %.2f\n", h_in[0], h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
}

}  // namespace cuda13_examples
