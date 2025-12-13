// Chapter 11: Book-aligned baseline that processes warp-specialized batches sequentially on the default stream.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <numeric>
#include <chrono>
#include "../core/common/headers/cuda_verify.cuh"

namespace {
constexpr int TILE = 32;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int THREADS = 96; // three warps

__device__ void compute_tile(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int lane) {
    for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
        int row = idx / TILE;
        int col = idx % TILE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += A[row * TILE + k] * B[k * TILE + col];
        }
        C[idx] = acc;
    }
}

__global__ void simple_warp_specialized_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C) {
    extern __shared__ float shared[];
    float* A_tile = shared;
    float* B_tile = shared + TILE_ELEMS;
    float* C_tile = shared + 2 * TILE_ELEMS;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (warp_id == 0) {
        for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
            A_tile[idx] = A[idx];
            B_tile[idx] = B[idx];
        }
    }

    __syncthreads();

    if (warp_id == 1) {
        compute_tile(A_tile, B_tile, C_tile, lane_id);
    }

    __syncthreads();

    if (warp_id == 2) {
        for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
            C[idx] = C_tile[idx];
        }
    }
}

void run_baseline() {
    constexpr int batches = 4096;
    const size_t bytes = TILE_ELEMS * sizeof(float);

    std::vector<float> h_A(batches * TILE_ELEMS);
    std::vector<float> h_B(batches * TILE_ELEMS);
    std::vector<float> h_C(batches * TILE_ELEMS);
    std::iota(h_A.begin(), h_A.end(), 0.0f);
    std::iota(h_B.begin(), h_B.end(), 1.0f);

    cudaDeviceSynchronize();
    const auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < batches; ++b) {
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMalloc(&dA, bytes);
        cudaMalloc(&dB, bytes);
        cudaMalloc(&dC, bytes);

        cudaMemcpy(dA, h_A.data() + b * TILE_ELEMS, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, h_B.data() + b * TILE_ELEMS, bytes, cudaMemcpyHostToDevice);

        simple_warp_specialized_kernel<<<1, THREADS, 3 * bytes>>>(dA, dB, dC);
        cudaMemcpy(h_C.data() + b * TILE_ELEMS, dC, bytes, cudaMemcpyDeviceToHost);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    cudaDeviceSynchronize();
    const auto stop = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    double checksum = 0.0;
    for (float v : h_C) checksum += v;

    const float verify_checksum = static_cast<float>(checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
    printf("TIME_MS: %.3f\n", ms);
}
}  // namespace

int main() {
    run_baseline();
    return 0;
}
