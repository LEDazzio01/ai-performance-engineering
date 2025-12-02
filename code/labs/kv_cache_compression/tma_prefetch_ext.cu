/**
 * Async Prefetch Extension for KV Cache
 * ======================================
 *
 * Demonstrates async bulk data movement patterns for KV cache prefetching.
 * Uses cp.async (SM80+) for non-blocking GMEM → SMEM transfers.
 *
 * Use case: Prefetch KV cache tiles from global memory into shared memory
 * ahead of attention computation. This hides memory latency by overlapping
 * data movement with computation.
 *
 * Key concepts:
 * - cp.async: Non-blocking copy that proceeds independently of compute
 * - Double-buffering: Overlap prefetch of next tile with compute of current
 *
 * Note: cp.async requires 4, 8, or 16 byte transfers. For efficiency,
 * we use 16-byte (128-bit) transfers when possible.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// Tile dimensions for KV cache prefetch
constexpr int TILE_ROWS = 32;      // Tokens per tile
constexpr int TILE_COLS = 128;     // Head dimension (must match model)
constexpr int THREADS_PER_BLOCK = 128;

// cp.async requires 4, 8, or 16 bytes. Use 16 for best performance.
constexpr int CP_ASYNC_BYTES = 16;

/**
 * Async prefetch kernel using cp.async PTX with 16-byte transfers
 */
template <typename T>
__global__ void async_prefetch_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int rows,
    int cols
) {
    // Each block handles one tile
    const int tile_row = blockIdx.x * TILE_ROWS;
    if (tile_row >= rows) return;

    const int rows_this_tile = min(TILE_ROWS, rows - tile_row);
    const int tid = threadIdx.x;

    // Shared memory for prefetched tile (aligned for 128-bit access)
    __shared__ alignas(128) T smem_tile[TILE_ROWS][TILE_COLS];

    // Number of elements per 16-byte transfer
    constexpr int elems_per_transfer = CP_ASYNC_BYTES / sizeof(T);
    constexpr int transfers_per_row = TILE_COLS / elems_per_transfer;
    constexpr int total_transfers = TILE_ROWS * transfers_per_row;
    constexpr int transfers_per_thread = total_transfers / THREADS_PER_BLOCK;

#if __CUDA_ARCH__ >= 800
    // SM80+: Use cp.async for true async 16-byte copy
    for (int i = 0; i < transfers_per_thread; ++i) {
        const int transfer_idx = tid * transfers_per_thread + i;
        const int r = transfer_idx / transfers_per_row;
        const int c_base = (transfer_idx % transfers_per_row) * elems_per_transfer;

        if (r < rows_this_tile) {
            const size_t src_offset = (tile_row + r) * cols + c_base;
            // cp.async with 16 bytes
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[r][c_base]))),
                  "l"(&src[src_offset])
                : "memory"
            );
        }
    }

    // Commit the async copies and wait for all to complete
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");

#else
    // Fallback for older architectures: synchronous copy
    for (int i = tid; i < TILE_ROWS * TILE_COLS; i += THREADS_PER_BLOCK) {
        const int r = i / TILE_COLS;
        const int c = i % TILE_COLS;
        if (r < rows_this_tile && c < cols) {
            smem_tile[r][c] = src[(tile_row + r) * cols + c];
        }
    }
#endif

    __syncthreads();

    // Now data is in SMEM - in real code, you'd compute attention here
    // For demo, we just write back to verify the prefetch worked

    for (int i = tid; i < TILE_ROWS * TILE_COLS; i += THREADS_PER_BLOCK) {
        const int r = i / TILE_COLS;
        const int c = i % TILE_COLS;
        if (r < rows_this_tile && c < cols) {
            dst[(tile_row + r) * cols + c] = smem_tile[r][c];
        }
    }
}

/**
 * Double-buffered prefetch kernel
 *
 * This demonstrates true overlap: while computing on tile N,
 * prefetch tile N+1. This is the pattern used in production kernels.
 */
template <typename T>
__global__ void double_buffer_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int rows,
    int cols,
    int num_tiles
) {
    const int tid = threadIdx.x;

    // Double buffer in shared memory
    __shared__ alignas(128) T smem_tile[2][TILE_ROWS][TILE_COLS];

    // Number of elements per 16-byte transfer
    constexpr int elems_per_transfer = CP_ASYNC_BYTES / sizeof(T);
    constexpr int transfers_per_row = TILE_COLS / elems_per_transfer;
    constexpr int total_transfers = TILE_ROWS * transfers_per_row;
    constexpr int transfers_per_thread = total_transfers / THREADS_PER_BLOCK;

    int current_buffer = 0;

#if __CUDA_ARCH__ >= 800
    // Prefetch first tile (async)
    if (num_tiles > 0) {
        for (int i = 0; i < transfers_per_thread; ++i) {
            const int transfer_idx = tid * transfers_per_thread + i;
            const int r = transfer_idx / transfers_per_row;
            const int c_base = (transfer_idx % transfers_per_row) * elems_per_transfer;

            if (r < TILE_ROWS && r < rows) {
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[0][r][c_base]))),
                      "l"(&src[r * cols + c_base])
                    : "memory"
                );
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    // Process tiles with double buffering
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_row = tile * TILE_ROWS;
        const int rows_this_tile = min(TILE_ROWS, rows - tile_row);
        const int next_tile = tile + 1;
        const int next_buffer = 1 - current_buffer;

        // Start prefetch of NEXT tile (async) 
        if (next_tile < num_tiles) {
            const int next_tile_row = next_tile * TILE_ROWS;
            for (int i = 0; i < transfers_per_thread; ++i) {
                const int transfer_idx = tid * transfers_per_thread + i;
                const int r = transfer_idx / transfers_per_row;
                const int c_base = (transfer_idx % transfers_per_row) * elems_per_transfer;

                if (r < TILE_ROWS && (next_tile_row + r) < rows) {
                    asm volatile(
                        "cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[next_buffer][r][c_base]))),
                          "l"(&src[(next_tile_row + r) * cols + c_base])
                        : "memory"
                    );
                }
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        // Wait for CURRENT tile's prefetch to complete (leave next one in flight)
        if (next_tile < num_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();

        // Process CURRENT tile (data is ready in smem_tile[current_buffer])
        // In real attention: compute QK^T, softmax, PV here
        // For demo, just write back
        for (int i = tid; i < TILE_ROWS * TILE_COLS; i += THREADS_PER_BLOCK) {
            const int r = i / TILE_COLS;
            const int c = i % TILE_COLS;
            if (r < rows_this_tile && c < cols) {
                dst[(tile_row + r) * cols + c] = smem_tile[current_buffer][r][c];
            }
        }
        __syncthreads();

        current_buffer = next_buffer;
    }

#else
    // Fallback: no async, just tile-by-tile
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_row = tile * TILE_ROWS;
        const int rows_this_tile = min(TILE_ROWS, rows - tile_row);

        // Load tile
        for (int i = tid; i < TILE_ROWS * TILE_COLS; i += THREADS_PER_BLOCK) {
            const int r = i / TILE_COLS;
            const int c = i % TILE_COLS;
            if (r < rows_this_tile && c < cols) {
                smem_tile[0][r][c] = src[(tile_row + r) * cols + c];
            }
        }
        __syncthreads();

        // Write tile
        for (int i = tid; i < TILE_ROWS * TILE_COLS; i += THREADS_PER_BLOCK) {
            const int r = i / TILE_COLS;
            const int c = i % TILE_COLS;
            if (r < rows_this_tile && c < cols) {
                dst[(tile_row + r) * cols + c] = smem_tile[0][r][c];
            }
        }
        __syncthreads();
    }
#endif
}

// Launch wrappers

template <typename T>
void launch_async_prefetch(const torch::Tensor& src, torch::Tensor& dst) {
    const int rows = static_cast<int>(src.size(0));
    const int cols = static_cast<int>(src.size(1));
    const int num_tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;

    dim3 grid(num_tiles);
    dim3 block(THREADS_PER_BLOCK);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    async_prefetch_kernel<T><<<grid, block, 0, stream>>>(
        src.data_ptr<T>(), dst.data_ptr<T>(), rows, cols);
    AT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_double_buffer(const torch::Tensor& src, torch::Tensor& dst) {
    const int rows = static_cast<int>(src.size(0));
    const int cols = static_cast<int>(src.size(1));
    const int num_tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;

    // Single block processes all tiles with double buffering
    dim3 grid(1);
    dim3 block(THREADS_PER_BLOCK);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    double_buffer_kernel<T><<<grid, block, 0, stream>>>(
        src.data_ptr<T>(), dst.data_ptr<T>(), rows, cols, num_tiles);
    AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

/**
 * Async prefetch for KV cache tensors.
 *
 * Demonstrates async GMEM → SMEM copy pattern using cp.async (SM80+).
 *
 * Args:
 *   src: Input tensor [tokens, head_dim]
 *   dst: Output tensor [tokens, head_dim]
 *   double_buffer: Use double-buffering for better overlap
 */
void tma_prefetch(torch::Tensor src, torch::Tensor dst, bool double_buffer) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "Async prefetch requires CUDA tensors");
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "Source/destination dtypes must match");
    TORCH_CHECK(src.dim() == 2 && dst.dim() == 2, "Async prefetch expects 2D tensors [tokens, head_dim]");
    TORCH_CHECK(src.sizes() == dst.sizes(), "Source/destination shapes must match");
    TORCH_CHECK(src.size(1) == TILE_COLS, "Async prefetch requires head_dim == 128");

    c10::cuda::CUDAGuard guard(src.get_device());

    auto src_contig = src.contiguous();
    auto dst_contig = dst.contiguous();

    if (double_buffer) {
        switch (src_contig.scalar_type()) {
            case torch::kFloat32:
                launch_double_buffer<float>(src_contig, dst_contig);
                break;
            case torch::kFloat16:
                launch_double_buffer<at::Half>(src_contig, dst_contig);
                break;
            case torch::kBFloat16:
                launch_double_buffer<at::BFloat16>(src_contig, dst_contig);
                break;
            default:
                TORCH_CHECK(false, "Async prefetch supports float32, float16, bfloat16");
        }
    } else {
        switch (src_contig.scalar_type()) {
            case torch::kFloat32:
                launch_async_prefetch<float>(src_contig, dst_contig);
                break;
            case torch::kFloat16:
                launch_async_prefetch<at::Half>(src_contig, dst_contig);
                break;
            case torch::kBFloat16:
                launch_async_prefetch<at::BFloat16>(src_contig, dst_contig);
                break;
            default:
                TORCH_CHECK(false, "Async prefetch supports float32, float16, bfloat16");
        }
    }

    if (!dst.is_contiguous()) {
        dst.copy_(dst_contig);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tma_prefetch", &tma_prefetch,
          "Async prefetch for KV cache tensors (SM80+ cp.async)",
          py::arg("src"),
          py::arg("dst"),
          py::arg("double_buffer") = false);
}
