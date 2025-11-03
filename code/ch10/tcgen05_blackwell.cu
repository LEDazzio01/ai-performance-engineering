/**
 * Blackwell Tensor Core GEMM (adaptive)
 * -------------------------------------
 * Demonstrates selecting Tensor Core tile shapes at runtime based on the
 * active GPU. The simple kernel remains educational; production code should
 * use CUTLASS 4.2+ or cuBLAS 13.x which already map to tcgen05.mma on
 * Blackwell and wgmma on Hopper.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../common/headers/arch_detection.cuh"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _status = (call);                                                    \
        if (_status != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                         cudaGetErrorString(_status));                                   \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void simple_fp16_gemm(const __half* __restrict__ A,
                                 const __half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty * THREAD_TILE_M;
    int col = bx * TILE_N + tx * THREAD_TILE_N;

    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    __shared__ __half As[TILE_M][TILE_K];
    __shared__ __half Bs[TILE_K][TILE_N];

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        int linear_tid = ty * blockDim.x + tx;
        int threads_per_block = blockDim.x * blockDim.y;

        for (int idx = linear_tid; idx < TILE_M * TILE_K; idx += threads_per_block) {
            int local_row = idx / TILE_K;
            int local_col = idx % TILE_K;
            int a_row = by * TILE_M + local_row;
            int a_col = k_tile + local_col;
            As[local_row][local_col] = (a_row < M && a_col < K)
                                           ? A[a_row * K + a_col]
                                           : __float2half(0.0f);
        }

        for (int idx = linear_tid; idx < TILE_K * TILE_N; idx += threads_per_block) {
            int local_row = idx / TILE_N;
            int local_col = idx % TILE_N;
            int b_row = k_tile + local_row;
            int b_col = bx * TILE_N + local_col;
            Bs[local_row][local_col] = (b_row < K && b_col < N)
                                           ? B[b_row * N + b_col]
                                           : __float2half(0.0f);
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    int global_row = row + i;
                    int global_col = col + j;
                    if (global_row < M && global_col < N) {
                        accum[i][j] += __half2float(As[ty * THREAD_TILE_M + i][k]) *
                                       __half2float(Bs[k][tx * THREAD_TILE_N + j]);
                    }
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE_M; ++i) {
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int global_row = row + i;
            int global_col = col + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[i][j];
            }
        }
    }
}

template <int TM, int TN, int TK>
void launch_simple_kernel(int M, int N, int K,
                          const std::vector<__half>& h_A,
                          const std::vector<__half>& h_B,
                          std::vector<float>& h_C) {
    dim3 block(16, 16);
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);

    size_t size_A = static_cast<size_t>(M) * K * sizeof(__half);
    size_t size_B = static_cast<size_t>(K) * N * sizeof(__half);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(float);

    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    simple_fp16_gemm<TM, TN, TK><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    auto cfg = cuda_arch::select_tensor_core_tile();
    printf("=== Adaptive Tensor Core GEMM ===\n");
    printf("Matrix size: %dx%d @ %dx%d\n", M, K, K, N);
    printf("Selected tile (M,N,K) = (%d,%d,%d)\n", cfg.m, cfg.n, cfg.k);

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    std::vector<__half> h_A(size_A);
    std::vector<__half> h_B(size_B);
    std::vector<float> h_C(size_C);
    std::vector<float> h_ref(size_C);

    for (auto& v : h_A) { v = __float2half(static_cast<float>(rand()) / RAND_MAX); }
    for (auto& v : h_B) { v = __float2half(static_cast<float>(rand()) / RAND_MAX); }

    if (cfg.m == 128 && cfg.n == 128 && cfg.k == 64) {
        launch_simple_kernel<128, 128, 64>(M, N, K, h_A, h_B, h_C);
    } else if (cfg.m == 64 && cfg.n == 64 && cfg.k == 32) {
        launch_simple_kernel<64, 64, 32>(M, N, K, h_A, h_B, h_C);
    } else {
        launch_simple_kernel<32, 32, 16>(M, N, K, h_A, h_B, h_C);
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(h_A[m * K + k]) * __half2float(h_B[k * N + n]);
            }
            h_ref[m * N + n] = acc;
        }
    }

    double max_err = 0.0;
    for (size_t i = 0; i < size_C; ++i) {
        max_err = std::max(max_err, static_cast<double>(std::abs(h_C[i] - h_ref[i])));
    }
    printf("Max absolute error: %.5e\n", max_err);
    printf("Note: For production workloads use CUTLASS 4.2+ or cuBLAS 13.x.\n");

    return 0;
}
