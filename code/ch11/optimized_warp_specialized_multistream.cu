// Chapter 11: Book-aligned optimized version overlapping batches across CUDA streams.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <numeric>
#include <chrono>
#include "../core/common/headers/cuda_verify.cuh"

namespace {
constexpr int TILE = 32;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int THREADS = 96;

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

void run_optimized() {
    constexpr int batches = 4096;
    constexpr int num_streams = 3;
    const size_t bytes = TILE_ELEMS * sizeof(float);

    // Use pinned host memory so H2D/D2H can overlap with compute.
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, static_cast<size_t>(batches) * TILE_ELEMS * sizeof(float));
    cudaMallocHost(&h_B, static_cast<size_t>(batches) * TILE_ELEMS * sizeof(float));
    cudaMallocHost(&h_C, static_cast<size_t>(batches) * TILE_ELEMS * sizeof(float));
    for (int i = 0; i < batches * TILE_ELEMS; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i + 1);
        h_C[i] = 0.0f;
    }

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    cudaDeviceSynchronize();
    const auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < batches; ++b) {
        cudaStream_t st = streams[b % num_streams];
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMallocAsync(&dA, bytes, st);
        cudaMallocAsync(&dB, bytes, st);
        cudaMallocAsync(&dC, bytes, st);

        cudaMemcpyAsync(dA, h_A + static_cast<size_t>(b) * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB, h_B + static_cast<size_t>(b) * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, st);

        simple_warp_specialized_kernel<<<1, THREADS, 3 * bytes, st>>>(dA, dB, dC);

        cudaMemcpyAsync(h_C + static_cast<size_t>(b) * TILE_ELEMS, dC, bytes, cudaMemcpyDeviceToHost, st);

        cudaFreeAsync(dA, st);
        cudaFreeAsync(dB, st);
        cudaFreeAsync(dC, st);
    }

    cudaDeviceSynchronize();
    const auto stop = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    double checksum = 0.0;
    for (int i = 0; i < batches * TILE_ELEMS; ++i) checksum += h_C[i];

    const float verify_checksum = static_cast<float>(checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
    printf("TIME_MS: %.3f\n", ms);
    for (int i = 0; i < num_streams; ++i) cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}
}  // namespace

int main() {
    run_optimized();
    return 0;
}
