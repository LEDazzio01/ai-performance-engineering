// baseline_flash_attn_tma_micro_pipeline.cu
//
// FlashAttention-shaped micro-pipeline without async copies.
// Uses blocking global->shared loads per tile for K/V, then compute.
// Serves as the baseline against the async TMA-enabled variant.

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t _status = (call);                                        \
        if (_status != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(_status));   \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)
#endif

// Toy problem sizes (kept identical to the optimized variant for A/B comparisons)
constexpr int SEQ_LEN = 2048;
constexpr int D_HEAD  = 64;
constexpr int TILE_KV = 40;    // rows per tile (kept small to fit shared memory comfortably)
constexpr int THREADS = 128;

__global__ void flash_attn_baseline_kernel(
    const float* __restrict__ q,   // [SEQ_LEN, D_HEAD]
    const float* __restrict__ k,   // [SEQ_LEN, D_HEAD]
    const float* __restrict__ v,   // [SEQ_LEN, D_HEAD]
    float* __restrict__ o,         // [SEQ_LEN, D_HEAD]
    int seq_len,
    int d_head) {
    const int q_idx = blockIdx.x;  // one query row per block
    if (q_idx >= seq_len) return;

    extern __shared__ float smem[];
    float* smem_k = smem;
    float* smem_v = smem_k + TILE_KV * d_head;

    const int tid = threadIdx.x;
    __shared__ float score_smem[THREADS];

    // Load Q row into registers.
    float q_reg[D_HEAD];
    for (int d = tid; d < d_head; d += blockDim.x) {
        q_reg[d] = q[q_idx * d_head + d];
    }
    __syncthreads();

    float o_reg[D_HEAD];
    for (int d = 0; d < d_head; ++d) {
        o_reg[d] = 0.0f;
    }

    const int num_tiles = (seq_len + TILE_KV - 1) / TILE_KV;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int row_base = tile_idx * TILE_KV;
        const int rows_this_tile = min(TILE_KV, seq_len - row_base);

        // Blocking load of K/V tile into shared memory.
        for (int r = tid; r < rows_this_tile; r += blockDim.x) {
            const float* k_row = k + (row_base + r) * d_head;
            const float* v_row = v + (row_base + r) * d_head;
            float* k_s = smem_k + r * d_head;
            float* v_s = smem_v + r * d_head;
            for (int d = 0; d < d_head; ++d) {
                k_s[d] = k_row[d];
                v_s[d] = v_row[d];
            }
        }
        __syncthreads();

        for (int r = 0; r < rows_this_tile; ++r) {
            const float* k_row = smem_k + r * d_head;
            const float* v_row = smem_v + r * d_head;

            // Dot product q · k_r
            float score = 0.f;
            for (int d = tid; d < d_head; d += blockDim.x) {
                score += q_reg[d] * k_row[d];
            }

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
    }

    for (int d = tid; d < d_head; d += blockDim.x) {
        o[q_idx * d_head + d] = o_reg[d];
    }
}

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

int main() {
    if (!device_available()) {
        std::printf("SKIP: No CUDA device found.\nTIME_MS=0.0\n");
        return 0;
    }

    // Baseline runs on any GPU - no TMA required

    const int seq_len = SEQ_LEN;
    const int d_head = D_HEAD;
    const size_t bytes = size_t(seq_len) * d_head * sizeof(float);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q, bytes));
    CHECK_CUDA(cudaMalloc(&d_k, bytes));
    CHECK_CUDA(cudaMalloc(&d_v, bytes));
    CHECK_CUDA(cudaMalloc(&d_o, bytes));

    // Deterministic non-zero initialization (outside timed region).
    std::vector<float> h_q(seq_len * d_head);
    std::vector<float> h_k(seq_len * d_head);
    std::vector<float> h_v(seq_len * d_head);
    for (int i = 0; i < seq_len * d_head; ++i) {
        h_q[i] = (static_cast<float>((i % 13) - 6)) * 0.01f;
        h_k[i] = (static_cast<float>((i % 17) - 8)) * 0.01f;
        h_v[i] = (static_cast<float>((i % 19) - 9)) * 0.01f;
    }
    CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_o, 0, bytes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const dim3 block(THREADS);
    const dim3 grid(seq_len);
    const size_t shmem_bytes = 2 * TILE_KV * d_head * sizeof(float); // K + V

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    flash_attn_baseline_kernel<<<grid, block, shmem_bytes, stream>>>(
        d_q, d_k, d_v, d_o, seq_len, d_head);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Benchmark with multiple iterations
    constexpr int ITERS = 10;
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERS; ++i) {
        flash_attn_baseline_kernel<<<grid, block, shmem_bytes, stream>>>(
            d_q, d_k, d_v, d_o, seq_len, d_head);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

#ifdef VERIFY
    std::vector<float> h_o(seq_len * d_head);
    CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, bytes, cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_o) {
        checksum += static_cast<double>(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_o));

    std::printf("FlashAttention baseline: %.3f ms\n", avg_ms);
    std::printf("TIME_MS: %.6f\n", avg_ms);
    return 0;
}
