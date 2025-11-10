#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "double_buffering_common.cuh"

namespace ch8 {

void double_buffer_baseline(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.numel() == input.numel(), "output length mismatch");

    at::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_double_buffer_baseline(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void double_buffer_optimized(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.numel() == input.numel(), "output length mismatch");

    at::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_double_buffer_optimized(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ch8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("double_buffer_baseline", &ch8::double_buffer_baseline, "Double buffering baseline");
    m.def("double_buffer_optimized", &ch8::double_buffer_optimized, "Double buffering optimized");
}
