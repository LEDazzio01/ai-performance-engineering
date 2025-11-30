#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "loop_unrolling_common.cuh"

namespace ch08 {

void loop_unrolling_baseline(torch::Tensor inputs, torch::Tensor weights, torch::Tensor output) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.dim() == 2, "inputs must be [rows, elements]");
    TORCH_CHECK(weights.numel() == kWeightPeriod, "weights must have length ", kWeightPeriod);
    TORCH_CHECK(output.numel() == inputs.size(0), "output must match row count");
    TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(inputs.size(1) == kElementsPerRow, "inputs second dimension must be ", kElementsPerRow);

    const int rows = static_cast<int>(inputs.size(0));
    at::cuda::CUDAGuard guard(inputs.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_loop_unrolling_baseline(
        inputs.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void loop_unrolling_optimized(torch::Tensor inputs, torch::Tensor weights, torch::Tensor output) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.dim() == 2, "inputs must be [rows, elements]");
    TORCH_CHECK(weights.numel() == kWeightPeriod, "weights must have length ", kWeightPeriod);
    TORCH_CHECK(output.numel() == inputs.size(0), "output must match row count");
    TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(inputs.size(1) == kElementsPerRow, "inputs second dimension must be ", kElementsPerRow);

    const int rows = static_cast<int>(inputs.size(0));
    at::cuda::CUDAGuard guard(inputs.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_loop_unrolling_optimized(
        inputs.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ch08

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "loop_unrolling_baseline",
        &ch08::loop_unrolling_baseline,
        "Baseline loop-unrolling kernel with redundant accumulation");
    m.def(
        "loop_unrolling_optimized",
        &ch08::loop_unrolling_optimized,
        "Optimized loop-unrolling kernel with ILP and vectorized loads");
}
