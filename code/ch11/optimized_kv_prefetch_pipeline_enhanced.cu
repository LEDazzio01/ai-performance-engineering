#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void kv_prefetch_overlap(const float* __restrict__ keys,
                                    const float* __restrict__ values,
                                    float* __restrict__ out,
                                    int seq_len,
                                    int head_dim) {
    extern __shared__ float smem[];
    float* k_tile = smem;
    float* v_tile = smem + head_dim;

    // Use uninitialized storage to avoid dynamic initialization warning
    using pipe_state = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;
    __shared__ alignas(pipe_state) unsigned char state_storage[sizeof(pipe_state)];
    auto* state = reinterpret_cast<pipe_state*>(state_storage);
    auto pipe = cuda::make_pipeline(cg::this_thread_block(), state);

    for (int t = 0; t < seq_len; ++t) {
        int offset = t * head_dim;

        pipe.producer_acquire();
        for (int idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
            k_tile[idx] = keys[offset + idx];
            v_tile[idx] = values[offset + idx];
        }
        pipe.producer_commit();

        pipe.consumer_wait();
        __syncthreads();

        for (int idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
            out[offset + idx] = k_tile[idx] + v_tile[idx];
        }

        pipe.consumer_release();
        __syncthreads();
    }
}

int main() {
    const int seq = 64;
    const int head_dim = 128;
    const size_t elems = static_cast<size_t>(seq) * head_dim;

    float *keys = nullptr, *values = nullptr, *out = nullptr;
    cudaMalloc(&keys, elems * sizeof(float));
    cudaMalloc(&values, elems * sizeof(float));
    cudaMalloc(&out, elems * sizeof(float));

    kv_prefetch_overlap<<<1, 128, 2 * head_dim * sizeof(float)>>>(keys, values, out, seq, head_dim);
    cudaDeviceSynchronize();

    cudaFree(keys);
    cudaFree(values);
    cudaFree(out);
    printf("kv_prefetch_overlap demo completed.\n");
    return 0;
}
