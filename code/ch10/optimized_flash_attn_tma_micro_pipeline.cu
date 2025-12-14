// optimized_flash_attn_tma_micro_pipeline.cu
//
// FlashAttention-style micro-pipeline using TMA (Tensor Memory Accelerator).
// Double-buffered PREFETCH (K/V tiles) overlapped with COMPUTE using
// cp.async.bulk.tensor for global->shared transfers with mbarrier completion.
// Targets SM90+ (Hopper/Blackwell) for TMA bulk tensor operations.

#include <cuda/barrier>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/tma_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"

#if CUDART_VERSION < 13000
int main() {
    std::printf("SKIP: requires CUDA 13.0+ for TMA bulk tensor\nTIME_MS: 0.0\n");
    return 0;
}
#else

namespace cde = cuda::device::experimental;
using block_barrier = cuda::barrier<cuda::thread_scope_block>;
using cuda_tma::check_cuda;
using cuda_tma::load_cuTensorMapEncodeTiled;
using cuda_tma::make_2d_tensor_map;

constexpr int SEQ_LEN = 2048;
constexpr int D_HEAD  = 64;
constexpr int TILE_KV = 64;    // rows per tile (K/V)
constexpr int THREADS = 128;
constexpr int STAGES  = 2;     // double buffer
constexpr int ITERS   = 10;

inline bool make_2d_tensor_map_col_row(
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
        static_cast<std::uint64_t>(height),
    };
    std::uint64_t stride[rank - 1] = {static_cast<std::uint64_t>(ld * sizeof(float))};
    std::uint32_t box[rank] = {static_cast<std::uint32_t>(box_width),
                               static_cast<std::uint32_t>(box_height)};
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

// TMA kernel with double-buffered K/V prefetch
template <int TILE_M, int TILE_N>
__global__ void flash_attn_tma_kernel(
    const __grid_constant__ CUtensorMap k_desc,
    const __grid_constant__ CUtensorMap v_desc,
    const float* __restrict__ q,
    float* __restrict__ o,
    int seq_len,
    int d_head) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int q_idx = blockIdx.x;
    if (q_idx >= seq_len) return;

    constexpr size_t TILE_BYTES = size_t(TILE_M) * TILE_N * sizeof(float);
    
    // Aligned shared memory for TMA
    __shared__ alignas(128) float smem_k[STAGES][TILE_M][TILE_N];
    __shared__ alignas(128) float smem_v[STAGES][TILE_M][TILE_N];
    __shared__ alignas(block_barrier) unsigned char bar_storage[STAGES][sizeof(block_barrier)];
    
    block_barrier* bars[STAGES];
    for (int s = 0; s < STAGES; ++s) {
        bars[s] = reinterpret_cast<block_barrier*>(bar_storage[s]);
    }

    const int tid = threadIdx.x;
    
    // Initialize barriers
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init(bars[s], blockDim.x);
            cde::fence_proxy_async_shared_cta();
        }
    }
    __syncthreads();

    // Load Q row into registers
    float q_reg[D_HEAD];
    for (int d = tid; d < d_head; d += blockDim.x) {
        q_reg[d] = q[q_idx * d_head + d];
    }
    
    float o_reg[D_HEAD];
    for (int d = 0; d < D_HEAD; ++d) o_reg[d] = 0.f;

    const int num_tiles = (seq_len + TILE_M - 1) / TILE_M;
    block_barrier::arrival_token stage_tokens[STAGES];
    
    // Lambda to issue TMA load
    auto issue_tma_load = [&](int tile_idx) {
        if (tile_idx >= num_tiles) return;
        const int stage = tile_idx % STAGES;
        const int row_base = tile_idx * TILE_M;
        
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &smem_k[stage], &k_desc, 0, row_base, *bars[stage]);
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &smem_v[stage], &v_desc, 0, row_base, *bars[stage]);
            stage_tokens[stage] = cuda::device::barrier_arrive_tx(*bars[stage], 1, 2 * TILE_BYTES);
        }
        else {
            stage_tokens[stage] = bars[stage]->arrive();
        }
    };

    // Prime pipeline: issue loads for first STAGES tiles
    for (int t = 0; t < STAGES && t < num_tiles; ++t) {
        issue_tma_load(t);
    }

    // Main loop
    __shared__ float score_smem[128];
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int stage = tile_idx % STAGES;
        const int row_base = tile_idx * TILE_M;
        const int rows_this = min(TILE_M, seq_len - row_base);

        bars[stage]->wait(std::move(stage_tokens[stage]));
        __syncthreads();

        // Process all rows in this tile
        for (int r = 0; r < rows_this; ++r) {
            const float* k_row = &smem_k[stage][r][0];
            const float* v_row = &smem_v[stage][r][0];

            // Dot product q · k
            float score = 0.f;
            for (int d = tid; d < d_head; d += blockDim.x) {
                score += q_reg[d] * k_row[d];
            }

            // Warp-level reduction
            score_smem[tid] = score;
            __syncthreads();
            if (tid < 64) score_smem[tid] += score_smem[tid + 64];
            __syncthreads();
            if (tid < 32) score_smem[tid] += score_smem[tid + 32];
            __syncwarp();
            if (tid < 16) score_smem[tid] += score_smem[tid + 16];
            __syncwarp();
            if (tid < 8) score_smem[tid] += score_smem[tid + 8];
            __syncwarp();
            if (tid < 4) score_smem[tid] += score_smem[tid + 4];
            __syncwarp();
            if (tid < 2) score_smem[tid] += score_smem[tid + 2];
            __syncwarp();
            if (tid == 0) {
                float s = score_smem[0] + score_smem[1];
                s = fminf(fmaxf(s, -10.f), 10.f);
                score_smem[0] = __expf(s) * 1e-3f;
            }
            __syncthreads();

            float weight = score_smem[0];
            for (int d = tid; d < d_head; d += blockDim.x) {
                o_reg[d] += weight * v_row[d];
            }
            __syncthreads();
        }

        // Issue next tile load (pipelined)
        const int next_tile = tile_idx + STAGES;
        if (next_tile < num_tiles) {
            issue_tma_load(next_tile);
        }
    }

    // Write output
    for (int d = tid; d < d_head; d += blockDim.x) {
        o[q_idx * d_head + d] = o_reg[d];
    }
#else
    (void)k_desc; (void)v_desc; (void)q; (void)o; (void)seq_len; (void)d_head;
#endif
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: No CUDA device found.\nTIME_MS: 0.0\n");
        return 0;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const int sm_version = prop.major * 10 + prop.minor;
    
    if (sm_version < 90) {
        std::printf("SKIP: Requires SM90+ for TMA (found SM%d.%d)\nTIME_MS: 0.0\n",
                    prop.major, prop.minor);
        return 0;
    }
    
    if (!cuda_tma::device_supports_tma()) {
        std::printf("SKIP: TMA not supported on this device\nTIME_MS: 0.0\n");
        return 0;
    }

    const int seq_len = SEQ_LEN;
    const int d_head = D_HEAD;
    const size_t bytes = size_t(seq_len) * d_head * sizeof(float);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    check_cuda(cudaMalloc(&d_q, bytes), "malloc q");
    check_cuda(cudaMalloc(&d_k, bytes), "malloc k");
    check_cuda(cudaMalloc(&d_v, bytes), "malloc v");
    check_cuda(cudaMalloc(&d_o, bytes), "malloc o");

    // Deterministic non-zero initialization (outside timed region).
    std::vector<float> h_q(seq_len * d_head);
    std::vector<float> h_k(seq_len * d_head);
    std::vector<float> h_v(seq_len * d_head);
    for (int i = 0; i < seq_len * d_head; ++i) {
        h_q[i] = (static_cast<float>((i % 13) - 6)) * 0.01f;
        h_k[i] = (static_cast<float>((i % 17) - 8)) * 0.01f;
        h_v[i] = (static_cast<float>((i % 19) - 9)) * 0.01f;
    }
    check_cuda(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice), "copy k");
    check_cuda(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice), "copy v");
    check_cuda(cudaMemset(d_o, 0, bytes), "zero o");

    // Create TMA descriptors
    auto encode = load_cuTensorMapEncodeTiled();
    if (!encode) {
        std::printf("SKIP: Failed to load cuTensorMapEncodeTiled\nTIME_MS: 0.0\n");
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
        return 0;
    }

    CUtensorMap k_desc{}, v_desc{};
    const int box_h = TILE_KV;
    const int box_w = D_HEAD;
    
    if (!make_2d_tensor_map_col_row(k_desc, encode, d_k, d_head, seq_len, d_head,
                                    box_w, box_h, CU_TENSOR_MAP_SWIZZLE_NONE) ||
        !make_2d_tensor_map_col_row(v_desc, encode, d_v, d_head, seq_len, d_head,
                                    box_w, box_h, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        std::printf("SKIP: Failed to encode TMA descriptors\nTIME_MS: 0.0\n");
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
        return 0;
    }

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "stream create");

    const dim3 block(THREADS);
    const dim3 grid(seq_len);

    // Warmup
    flash_attn_tma_kernel<TILE_KV, D_HEAD><<<grid, block, 0, stream>>>(
        k_desc, v_desc, d_q, d_o, seq_len, d_head);
    check_cuda(cudaStreamSynchronize(stream), "warmup sync");
    check_cuda(cudaGetLastError(), "warmup error check");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "event start");
    check_cuda(cudaEventCreate(&stop), "event stop");

    check_cuda(cudaEventRecord(start, stream), "record start");
    for (int i = 0; i < ITERS; ++i) {
        flash_attn_tma_kernel<TILE_KV, D_HEAD><<<grid, block, 0, stream>>>(
            k_desc, v_desc, d_q, d_o, seq_len, d_head);
    }
    check_cuda(cudaEventRecord(stop, stream), "record stop");
    check_cuda(cudaEventSynchronize(stop), "event sync");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");
    float avg_ms = total_ms / ITERS;

    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");

#ifdef VERIFY
    std::vector<float> h_o(seq_len * d_head);
    check_cuda(cudaMemcpy(h_o.data(), d_o, bytes, cudaMemcpyDeviceToHost), "copy o");
    double checksum = 0.0;
    for (float v : h_o) {
        checksum += static_cast<double>(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    check_cuda(cudaFree(d_q), "free q");
    check_cuda(cudaFree(d_k), "free k");
    check_cuda(cudaFree(d_v), "free v");
    check_cuda(cudaFree(d_o), "free o");

    std::printf("FlashAttention TMA pipelined: %.3f ms\n", avg_ms);
    std::printf("TIME_MS: %.6f\n", avg_ms);
    return 0;
}

#endif  // CUDART_VERSION >= 13000
