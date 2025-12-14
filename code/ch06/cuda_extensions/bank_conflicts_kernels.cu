// bank_conflicts_kernels.cu - CUDA kernels for shared memory bank conflicts benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "profiling_helpers.cuh"

#define NUM_BANKS 32
#define INNER_SWEEPS 256
#define TILE_COLS (INNER_SWEEPS + NUM_BANKS)
#define TILE_VALUES (TILE_COLS * NUM_BANKS)
#define PADDED_STRIDE (NUM_BANKS + 1)
#define PADDED_VALUES (TILE_COLS * PADDED_STRIDE)

__global__ void bank_conflicts_kernel(float* output, const float* input, int N) {
    __shared__ float shared_data[TILE_VALUES];

    int tile_start = blockIdx.x * TILE_VALUES;
    if (tile_start >= N) {
        return;
    }

    int tile_len = min(TILE_VALUES, N - tile_start);
    int cols_in_tile = (tile_len + NUM_BANKS - 1) / NUM_BANKS;
    int active_sweeps = min(cols_in_tile, INNER_SWEEPS);

    for (int offset = threadIdx.x; offset < tile_len; offset += blockDim.x) {
        shared_data[offset] = input[tile_start + offset];
    }
    __syncthreads();

    for (int elem = threadIdx.x; elem < tile_len; elem += blockDim.x) {
        const int lane = elem & (NUM_BANKS - 1);
        const int conflict_stride = NUM_BANKS;
        const int logical_base = lane * conflict_stride;
        float acc = 0.0f;

        #pragma unroll 8
        for (int sweep = 0; sweep < INNER_SWEEPS; ++sweep) {
            if (sweep >= active_sweeps) {
                break;
            }
            int logical_offset = logical_base + sweep * conflict_stride;
            if (logical_offset >= tile_len) {
                break;
            }
            acc += shared_data[logical_offset];
        }

        output[tile_start + elem] = acc;
    }
}

__global__ void bank_conflicts_padded_kernel(float* output, const float* input, int N) {
    __shared__ float shared_data[PADDED_VALUES];

    int tile_start = blockIdx.x * TILE_VALUES;
    if (tile_start >= N) {
        return;
    }

    int tile_len = min(TILE_VALUES, N - tile_start);
    int cols_in_tile = (tile_len + NUM_BANKS - 1) / NUM_BANKS;
    int active_sweeps = min(cols_in_tile, INNER_SWEEPS);

    for (int offset = threadIdx.x; offset < tile_len; offset += blockDim.x) {
        int col = offset / NUM_BANKS;
        int row = offset % NUM_BANKS;
        int padded_idx = col * PADDED_STRIDE + row;
        shared_data[padded_idx] = input[tile_start + offset];
    }
    __syncthreads();

    for (int elem = threadIdx.x; elem < tile_len; elem += blockDim.x) {
        const int lane = elem & (NUM_BANKS - 1);
        float acc = 0.0f;

        #pragma unroll 8
        for (int sweep = 0; sweep < INNER_SWEEPS; ++sweep) {
            if (sweep >= active_sweeps) {
                break;
            }
            int logical_offset = (lane + sweep) * NUM_BANKS;
            if (logical_offset >= tile_len) {
                break;
            }
            int padded_idx = (lane + sweep) * PADDED_STRIDE;
            acc += shared_data[padded_idx];
        }

        output[tile_start + elem] = acc;
    }
}

void bank_conflicts(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (N + TILE_VALUES - 1) / TILE_VALUES;
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("bank_conflicts");
        bank_conflicts_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
    }
}

void bank_conflicts_padded(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (N + TILE_VALUES - 1) / TILE_VALUES;
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("bank_conflicts_padded");
        bank_conflicts_padded_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bank_conflicts", &bank_conflicts, "Bank conflicts kernel (baseline)");
    m.def("bank_conflicts_padded", &bank_conflicts_padded, "Bank conflicts kernel with padding (optimized)");
}
