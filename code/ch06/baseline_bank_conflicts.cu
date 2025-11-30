// baseline_bank_conflicts.cu
// Demonstrates how tightly coupled shared-memory accesses bottleneck when all
// threads in a warp hammer the same bank.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "../core/common/headers/profiling_helpers.cuh"

#define NUM_BANKS 32
#define INNER_SWEEPS 256
#define TILE_COLS (INNER_SWEEPS + NUM_BANKS)
#define TILE_VALUES (TILE_COLS * NUM_BANKS)

__global__ void bank_conflicts_kernel(float* output, const float* input, int N) {
    __shared__ float shared_data[TILE_VALUES];

    int tile_start = blockIdx.x * TILE_VALUES;
    if (tile_start >= N) {
        return;
    }

    int tile_len = std::min(TILE_VALUES, N - tile_start);
    int cols_in_tile = (tile_len + NUM_BANKS - 1) / NUM_BANKS;
    int active_sweeps = std::min(cols_in_tile, INNER_SWEEPS);

    for (int offset = threadIdx.x; offset < tile_len; offset += blockDim.x) {
        shared_data[offset] = input[tile_start + offset];
    }
    __syncthreads();

    for (int elem = threadIdx.x; elem < tile_len; elem += blockDim.x) {
        const int lane = elem & (NUM_BANKS - 1);
        const int conflict_stride = NUM_BANKS;
        const int logical_base = lane * conflict_stride;
        float acc = 0.0f;

        for (int sweep = 0; sweep < active_sweeps; ++sweep) {
            int logical_offset = logical_base + sweep * conflict_stride;
            if (logical_offset >= tile_len) {
                break;
            }
            acc += shared_data[logical_offset];
        }

        output[tile_start + elem] = acc;
    }
}

float measure_kernel(
    void (*kernel)(float*, const float*, int),
    float* d_output,
    const float* d_input,
    int N,
    int threads_per_block,
    cudaStream_t stream,
    const char* name,
    int launches
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + TILE_VALUES - 1) / TILE_VALUES;

    for (int launch = 0; launch < launches; ++launch) {
        kernel<<<blocks, threads_per_block, 0, stream>>>(d_output, d_input, N);
    }
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    {
        PROFILE_KERNEL_LAUNCH(name);
        for (int launch = 0; launch < launches; ++launch) {
            kernel<<<blocks, threads_per_block, 0, stream>>>(d_output, d_input, N);
        }
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int N = 1 << 20;
    const int threads_per_block = 256;
    const int baseline_launches = 32;

    printf("========================================\n");
    printf("Shared Memory Bank Conflicts (Baseline)\n");
    printf("========================================\n");
    printf("Problem size: %d elements\n", N);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Tile size: %d elements (%.2f KB)\n",
           TILE_VALUES,
           TILE_VALUES * sizeof(float) / 1024.0f);
    printf("Inner sweeps per launch: %d\n\n", INNER_SWEEPS);

    float* h_input = nullptr;
    float* h_output = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 1024);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMallocAsync(&d_input, N * sizeof(float), stream);
    cudaMallocAsync(&d_output, N * sizeof(float), stream);

    {
        PROFILE_MEMORY_COPY("H2D copy");
        cudaMemcpyAsync(
            d_input,
            h_input,
            N * sizeof(float),
            cudaMemcpyHostToDevice,
            stream);
    }
    cudaStreamSynchronize(stream);

    float conflicts_time = measure_kernel(
        bank_conflicts_kernel,
        d_output,
        d_input,
        N,
        threads_per_block,
        stream,
        "bank_conflicts_baseline",
        baseline_launches);
    printf("Baseline (conflicted) time: %.3f ms over %d launches\n",
           conflicts_time,
           baseline_launches);

    cudaMemcpyAsync(
        h_output,
        d_output,
        N * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    cudaStreamSynchronize(stream);

    printf("\n========================================\n");
    printf("Conflicted accesses complete. Output checksum: %.3f\n",
           h_output[0] + h_output[N / 2] + h_output[N - 1]);
    printf("========================================\n");

    cudaFreeAsync(d_output, stream);
    cudaFreeAsync(d_input, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_output);
    cudaFreeHost(h_input);

    return 0;
}
