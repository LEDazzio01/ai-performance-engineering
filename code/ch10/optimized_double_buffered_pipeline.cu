// optimized_double_buffered_pipeline.cu -- Shared-memory tiled GEMM with double buffering.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,\
                        cudaGetErrorString(status));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

namespace cg = cooperative_groups;

template<int TILE_M, int TILE_N, int CHUNK_K, int THREAD_TILE_M, int THREAD_TILE_N>
__global__ void gemm_double_buffered_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;
    cg::thread_block block = cg::this_thread_block();

    extern __shared__ float shared[];
    const int tileA_elems = TILE_M * CHUNK_K;
    const int tileB_elems = CHUNK_K * TILE_N;
    float* A_tiles[2];
    float* B_tiles[2];
    A_tiles[0] = shared;
    B_tiles[0] = A_tiles[0] + tileA_elems;
    A_tiles[1] = B_tiles[0] + tileB_elems;
    B_tiles[1] = A_tiles[1] + tileA_elems;

    const int total_chunks = (K + CHUNK_K - 1) / CHUNK_K;
    const int valid_cols = min(TILE_N, max(0, N - block_col));

    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;
    __shared__ alignas(pipeline_state_t) unsigned char pipe_state_bytes[sizeof(pipeline_state_t)];
    auto* pipe_state = reinterpret_cast<pipeline_state_t*>(pipe_state_bytes);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        new (pipe_state) pipeline_state_t();
    }
    block.sync();
    auto pipe = cuda::make_pipeline(block, pipe_state);

    auto zero_stage = [&](int stage) {
        const int threads = blockDim.x * blockDim.y;
        const int linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = linear_idx; idx < tileA_elems; idx += threads) {
            A_tiles[stage][idx] = 0.0f;
        }
        for (int idx = linear_idx; idx < tileB_elems; idx += threads) {
            B_tiles[stage][idx] = 0.0f;
        }
    };

    auto stage_async = [&](int stage, int chunk_index) {
        const int chunk_base = chunk_index * CHUNK_K;
        const int valid_k = max(0, min(CHUNK_K, K - chunk_base));
        pipe.producer_acquire();
        zero_stage(stage);
        block.sync();
        auto warp = cg::tiled_partition<32>(block);
        const int warp_id = warp.meta_group_rank();
        const int warp_count = warp.meta_group_size();

        if (valid_k > 0) {
            for (int row = warp_id; row < TILE_M; row += warp_count) {
                const int global_row = block_row + row;
                if (global_row >= M) {
                    continue;
                }
                float* dst = A_tiles[stage] + row * CHUNK_K;
                const float* src = A + global_row * K + chunk_base;
                if (valid_k == CHUNK_K) {
                    cuda::memcpy_async(
                        warp,
                        dst,
                        src,
                        cuda::aligned_size_t<16>(static_cast<size_t>(CHUNK_K) * sizeof(float)),
                        pipe);
                } else {
                    cuda::memcpy_async(
                        warp,
                        dst,
                        src,
                        static_cast<size_t>(valid_k) * sizeof(float),
                        pipe);
                }
            }
        }

        if (valid_cols > 0) {
            for (int row = warp_id; row < valid_k; row += warp_count) {
                const int global_row = chunk_base + row;
                if (global_row >= K) {
                    continue;
                }
                float* dst = B_tiles[stage] + row * TILE_N;
                const float* src = B + global_row * N + block_col;
                if (valid_cols == TILE_N) {
                    cuda::memcpy_async(
                        warp,
                        dst,
                        src,
                        cuda::aligned_size_t<16>(static_cast<size_t>(TILE_N) * sizeof(float)),
                        pipe);
                } else {
                    cuda::memcpy_async(
                        warp,
                        dst,
                        src,
                        static_cast<size_t>(valid_cols) * sizeof(float),
                        pipe);
                }
            }
        }

        pipe.producer_commit();
    };

    int next_chunk = 0;
    for (int stage = 0; stage < 2 && next_chunk < total_chunks; ++stage) {
        stage_async(stage, next_chunk++);
    }

    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    for (int chunk = 0; chunk < total_chunks; ++chunk) {
        const int stage = chunk % 2;
        pipe.consumer_wait();
        block.sync();
        const int chunk_base = chunk * CHUNK_K;
        for (int kk = 0; kk < CHUNK_K; ++kk) {
            if (chunk_base + kk >= K) {
                continue;
            }
            float a_frag[THREAD_TILE_M];
            float b_frag[THREAD_TILE_N];
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                int local_row = threadIdx.y * THREAD_TILE_M + i;
                a_frag[i] = A_tiles[stage][local_row * CHUNK_K + kk];
            }
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                int local_col = threadIdx.x * THREAD_TILE_N + j;
                b_frag[j] = B_tiles[stage][kk * TILE_N + local_col];
            }
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    accum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        pipe.consumer_release();
        if (next_chunk < total_chunks) {
            stage_async(stage, next_chunk++);
        }
    }

    block.sync();
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        const int row = block_row + threadIdx.y * THREAD_TILE_M + i;
        if (row >= M) continue;
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            const int col = block_col + threadIdx.x * THREAD_TILE_N + j;
            if (col >= N) continue;
            C[row * N + col] = accum[i][j];
        }
    }
}

int main() {
    // Larger matrices to show double-buffering benefit
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    constexpr int THREAD_TILE_M = 4;
    constexpr int THREAD_TILE_N = 4;
    constexpr int TILE_M = THREAD_TILE_M * 16;
    constexpr int TILE_N = THREAD_TILE_N * 16;
    constexpr int CHUNK_K = 32;
    dim3 block(16, 16);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    size_t shared_bytes = 2 * (TILE_M * CHUNK_K + CHUNK_K * TILE_N) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemm_double_buffered_kernel<TILE_M, TILE_N, CHUNK_K, THREAD_TILE_M, THREAD_TILE_N>
            <<<grid, block, shared_bytes>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(iterations);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float v : h_C) checksum += v;
    checksum /= static_cast<double>(M * N);

    std::printf("Optimized GEMM (double buffered tiles): %.3f ms (avg over %d iters)\n",
                avg_ms, iterations);
    std::printf("TIME_MS: %.6f\n", avg_ms);
    std::printf("Checksum: %.6f\n", checksum);
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
