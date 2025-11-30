#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "threshold_common.cuh"
#include "threshold_async_kernel.cuh"
#include "threshold_tma_kernel.cuh"
#include "blackwell_guard.cuh"

namespace ch08 {

namespace {

void ensure_blackwell_tma_device(const char* label) {
    cudaDeviceProp props{};
    cudaError_t err = cudaSuccess;
    if (is_blackwell_device(&props, &err)) {
        return;
    }

    if (err == cudaSuccess && props.major > 0) {
        TORCH_CHECK(
            false,
            label,
            " requires Blackwell/GB-series GPUs (found SM ",
            props.major,
            ".",
            props.minor,
            ")");
    }

    TORCH_CHECK(
        false,
        label,
        " requires Blackwell/GB-series GPUs (",
        cudaGetErrorString(err),
        ")");
}

}  // namespace

void threshold_baseline(torch::Tensor inputs, torch::Tensor output, double threshold) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(inputs.numel() == output.numel(), "input/output size mismatch");

    const int count = static_cast<int>(inputs.numel());
    at::cuda::CUDAGuard guard(inputs.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_threshold_naive(
        inputs.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(threshold),
        count,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void threshold_optimized(torch::Tensor inputs, torch::Tensor output, double threshold) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(inputs.numel() == output.numel(), "input/output size mismatch");

    const int count = static_cast<int>(inputs.numel());
    at::cuda::CUDAGuard guard(inputs.device());
    const auto stream = at::cuda::getCurrentCUDAStream();

#if CUDA_VERSION >= 12000
    cudaError_t async_error = cudaSuccess;
    const auto async_status = launch_threshold_predicated_async(
        inputs.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(threshold),
        count,
        stream,
        &async_error);
    if (async_status == ThresholdAsyncLaunchResult::kSuccess) {
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    if (async_status == ThresholdAsyncLaunchResult::kFailed) {
        TORCH_CHECK(
            false,
            "threshold_predicated_async launch failed: ",
            cudaGetErrorString(async_error));
    }
    TORCH_WARN_ONCE(
        "threshold_predicated_async unavailable (",
        cudaGetErrorString(async_error),
        "); falling back to predicated kernel.");
#endif

    launch_threshold_predicated(
        inputs.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(threshold),
        count,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void threshold_tma_baseline(torch::Tensor inputs, torch::Tensor output, double threshold) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(inputs.numel() == output.numel(), "input/output size mismatch");

    ensure_blackwell_tma_device("threshold_tma_baseline");

    const int count = static_cast<int>(inputs.numel());
    at::cuda::CUDAGuard guard(inputs.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    for (int pass = 0; pass < 3; ++pass) {
        launch_threshold_naive(
            inputs.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<float>(threshold),
            count,
            stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void threshold_tma_optimized(torch::Tensor inputs, torch::Tensor output, double threshold) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(inputs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(inputs.numel() == output.numel(), "input/output size mismatch");

    ensure_blackwell_tma_device("threshold_tma_optimized");

    const int count = static_cast<int>(inputs.numel());
    at::cuda::CUDAGuard guard(inputs.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = launch_threshold_tma_pipeline(
        inputs.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(threshold),
        count,
        stream);
    TORCH_CHECK(
        err == cudaSuccess,
        "threshold_tma pipeline launch failed: ",
        cudaGetErrorString(err));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ch08

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "threshold_baseline",
        &ch08::threshold_baseline,
        "Baseline threshold kernel with diverging branches",
        pybind11::arg("inputs"),
        pybind11::arg("output"),
        pybind11::arg("threshold"));
    m.def(
        "threshold_optimized",
        &ch08::threshold_optimized,
        "Optimized threshold kernel with predication",
        pybind11::arg("inputs"),
        pybind11::arg("output"),
        pybind11::arg("threshold"));
    m.def(
        "threshold_tma_baseline",
        &ch08::threshold_tma_baseline,
        "Baseline threshold kernel (gated for Blackwell TMA examples)",
        pybind11::arg("inputs"),
        pybind11::arg("output"),
        pybind11::arg("threshold"));
    m.def(
        "threshold_tma_optimized",
        &ch08::threshold_tma_optimized,
        "Optimized threshold kernel using CUDA pipeline/TMA staging",
        pybind11::arg("inputs"),
        pybind11::arg("output"),
        pybind11::arg("threshold"));
}
