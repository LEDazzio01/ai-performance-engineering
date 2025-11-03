#include <assert.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

namespace cg = cooperative_groups;

constexpr int TILE_ELEMS = 128 * 32;  // matches float4 alignment requirement
constexpr int THREADS = 256;

extern "C" __global__
void stage_ab_tiles(const float* __restrict__ globalA,
                    const float* __restrict__ globalB,
                    float* __restrict__ outC,
                    int tile_elems,
                    int num_tiles) {
    assert((tile_elems % (32 * 4)) == 0 && "tile_elems must be multiple of 128 floats");

    extern __shared__ float smem[];
    float* A0 = smem + 0 * tile_elems;
    float* A1 = smem + 1 * tile_elems;
    float* B0 = smem + 2 * tile_elems;
    float* B1 = smem + 3 * tile_elems;

    cg::thread_block block = cg::this_thread_block();

    constexpr auto scope = cuda::thread_scope_block;
    constexpr int stages = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages> pstate;
    auto pipe = cuda::make_pipeline(block, &pstate);

    // Prime stage 0.
    pipe.producer_acquire();
    cuda::memcpy_async(block,
                       A0,
                       globalA,
                       cuda::aligned_size_t<16>(tile_elems * sizeof(float)),
                       pipe);
    cuda::memcpy_async(block,
                       B0,
                       globalB,
                       cuda::aligned_size_t<16>(tile_elems * sizeof(float)),
                       pipe);
    pipe.producer_commit();

    for (int tile = 0; tile < num_tiles; ++tile) {
        float* a_stage = (tile % 2 == 0) ? A0 : A1;
        float* b_stage = (tile % 2 == 0) ? B0 : B1;
        float* a_next = (tile % 2 == 0) ? A1 : A0;
        float* b_next = (tile % 2 == 0) ? B1 : B0;

        pipe.consumer_wait();
        block.sync();  // ensure shared memory is ready

        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            outC[tile * tile_elems + i] = a_stage[i] + b_stage[i];
        }

        pipe.consumer_release();

        if (tile + 1 < num_tiles) {
            pipe.producer_acquire();
            cuda::memcpy_async(block,
                               a_next,
                               globalA + (tile + 1) * tile_elems,
                               cuda::aligned_size_t<16>(tile_elems * sizeof(float)),
                               pipe);
            cuda::memcpy_async(block,
                               b_next,
                               globalB + (tile + 1) * tile_elems,
                               cuda::aligned_size_t<16>(tile_elems * sizeof(float)),
                               pipe);
            pipe.producer_commit();
        }
    }
}

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

int main() {
    printf("Running two_stage_pipeline example\n");

    const int tiles = 8;
    const int elems = tiles * TILE_ELEMS;

    std::vector<float> hA(elems, 1.0f);
    std::vector<float> hB(elems, 2.0f);
    std::vector<float> hC(elems, 0.0f);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    check(cudaMalloc(&dA, elems * sizeof(float)));
    check(cudaMalloc(&dB, elems * sizeof(float)));
    check(cudaMalloc(&dC, elems * sizeof(float)));

    check(cudaMemcpy(dA, hA.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dB, hB.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(1);
    dim3 block(THREADS);
    size_t shared_bytes = 4 * TILE_ELEMS * sizeof(float);
    stage_ab_tiles<<<grid, block, shared_bytes>>>(dA, dB, dC, TILE_ELEMS, tiles);
    check(cudaDeviceSynchronize());

    check(cudaMemcpy(hC.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (float v : hC) {
        max_error = fmaxf(max_error, fabsf(v - 3.0f));
    }
    printf("Max error: %.6f\n", max_error);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
