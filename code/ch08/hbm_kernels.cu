#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "hbm_common.cuh"

namespace ch08 {

void hbm_baseline(torch::Tensor col_major, torch::Tensor output) {
    TORCH_CHECK(col_major.is_cuda(), "col_major tensor must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output tensor must be CUDA");
    TORCH_CHECK(col_major.dtype() == torch::kFloat32, "col_major must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(col_major.is_contiguous(), "col_major must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(col_major.dim() == 2, "col_major must be 2D matrix");

    const int cols = static_cast<int>(col_major.size(0));
    const int rows = static_cast<int>(col_major.size(1));
    TORCH_CHECK(output.numel() == rows, "output must have 'rows' elements");

    at::cuda::CUDAGuard guard(col_major.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_hbm_naive(
        col_major.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_optimized(torch::Tensor row_major, torch::Tensor output) {
    TORCH_CHECK(row_major.is_cuda(), "row_major tensor must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output tensor must be CUDA");
    TORCH_CHECK(row_major.dtype() == torch::kFloat32, "row_major must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(row_major.is_contiguous(), "row_major must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(row_major.dim() == 2, "row_major must be 2D matrix");

    const int rows = static_cast<int>(row_major.size(0));
    const int cols = static_cast<int>(row_major.size(1));
    TORCH_CHECK(cols % kVectorWidth == 0, "column count must be divisible by 4");
    TORCH_CHECK(output.numel() == rows, "output must have 'rows' elements");

    at::cuda::CUDAGuard guard(row_major.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_hbm_vectorized(
        row_major.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ch08

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hbm_baseline", &ch08::hbm_baseline, "HBM baseline kernel");
    m.def("hbm_optimized", &ch08::hbm_optimized, "HBM optimized kernel");
}
