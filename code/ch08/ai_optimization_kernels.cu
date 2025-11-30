#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "ai_optimization_common.cuh"

namespace ch08 {

void ai_baseline(torch::Tensor inputs, torch::Tensor weights, torch::Tensor output) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D vector");
    TORCH_CHECK(inputs.size(1) == weights.size(0), "weight length must match feature columns");
    TORCH_CHECK(output.numel() == inputs.size(0), "output must have one element per row");

    at::cuda::CUDAGuard guard(inputs.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_ai_baseline(
        inputs.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(inputs.size(0)),
        static_cast<int>(inputs.size(1)),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ai_optimized(torch::Tensor inputs, torch::Tensor weights, torch::Tensor output) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D vector");
    TORCH_CHECK(inputs.size(1) == weights.size(0), "weight length must match feature columns");
    TORCH_CHECK(output.numel() == inputs.size(0), "output must have one element per row");
    TORCH_CHECK((inputs.size(1) % 4) == 0, "columns must be divisible by 4 for vectorized loads");

    at::cuda::CUDAGuard guard(inputs.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_ai_optimized(
        inputs.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(inputs.size(0)),
        static_cast<int>(inputs.size(1)),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ch08

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ai_baseline", &ch08::ai_baseline, "AI optimization baseline kernel");
    m.def("ai_optimized", &ch08::ai_optimized, "AI optimization kernel with ILP/shared weights");
}
