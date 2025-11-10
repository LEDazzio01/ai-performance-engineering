// coalescing_kernels.cu - CUDA kernels for memory coalescing benchmarks
// Can be loaded as PyTorch CUDA extension

#include <algorithm>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "profiling_helpers.cuh"

namespace {
constexpr int kTileDim = 32;
constexpr int kTileRows = 8;
}

// Naive transpose where every thread reads row-major but writes column-major,
// producing fully uncoalesced global memory traffic.
__global__ void uncoalesced_transpose_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int rows,
    int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }
    const int input_idx = row * cols + col;
    const int output_idx = col * rows + row;
    output[output_idx] = input[input_idx];
}

template <int TileDim, int BlockRows>
__global__ void coalesced_transpose_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int rows,
    int cols) {
    __shared__ float tile[TileDim][TileDim + 1];

    int x = blockIdx.x * TileDim + threadIdx.x;
    int y = blockIdx.y * TileDim + threadIdx.y;

    for (int j = 0; j < TileDim; j += BlockRows) {
        const int load_y = y + j;
        if (load_y < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[load_y * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TileDim + threadIdx.x;
    y = blockIdx.x * TileDim + threadIdx.y;

    for (int j = 0; j < TileDim; j += BlockRows) {
        const int store_y = y + j;
        if (store_y < cols && x < rows) {
            output[store_y * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void check_inputs(torch::Tensor output, torch::Tensor input, int rows, int cols) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(input.numel() == rows * cols, "input numel mismatch");
    TORCH_CHECK(output.numel() == rows * cols, "output numel mismatch");
}

// Python-callable wrapper for naive transpose
void uncoalesced_copy(torch::Tensor output, torch::Tensor input, int rows, int cols) {
    check_inputs(output, input, rows, cols);

    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    cudaStream_t stream = nullptr;

    {
        PROFILE_KERNEL_LAUNCH("uncoalesced_transpose");
        uncoalesced_transpose_kernel<<<grid, block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            rows,
            cols
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
        }
    }
}

// Python-callable wrapper for tiled, coalesced transpose
void coalesced_copy(torch::Tensor output, torch::Tensor input, int rows, int cols) {
    check_inputs(output, input, rows, cols);

    dim3 block(kTileDim, kTileRows);
    dim3 grid((cols + kTileDim - 1) / kTileDim, (rows + kTileDim - 1) / kTileDim);
    cudaStream_t stream = nullptr;

    {
        PROFILE_KERNEL_LAUNCH("coalesced_transpose");
        coalesced_transpose_kernel<kTileDim, kTileRows><<<grid, block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            rows,
            cols
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uncoalesced_copy", &uncoalesced_copy, "Naive transpose with uncoalesced access");
    m.def("coalesced_copy", &coalesced_copy, "Tiled transpose with coalesced access");
}
