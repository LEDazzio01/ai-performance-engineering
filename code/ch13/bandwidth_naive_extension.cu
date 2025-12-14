#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace ch13 {

static inline bool is_power_of_two(uint64_t x) {
    return x && ((x & (x - 1)) == 0);
}

__global__ void bandwidth_add_mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint64_t n,
    uint32_t stride,
    uint32_t mask) {
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total_threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t logical = tid; logical < n; logical += total_threads) {
        const uint32_t idx = static_cast<uint32_t>((logical * static_cast<uint64_t>(stride)) & mask);
        const float out = (a[idx] + b[idx]) * 0.5f;
        c[idx] = out;
    }
}

void bandwidth_add_mul(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    int64_t stride,
    int64_t passes) {
    TORCH_CHECK(a.is_cuda(), "A must be CUDA");
    TORCH_CHECK(b.is_cuda(), "B must be CUDA");
    TORCH_CHECK(c.is_cuda(), "C must be CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(c.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "C must be contiguous");
    TORCH_CHECK(a.numel() == b.numel(), "A and B must have identical numel()");
    TORCH_CHECK(a.numel() == c.numel(), "A and C must have identical numel()");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(passes > 0, "passes must be positive");

    const uint64_t n = static_cast<uint64_t>(a.numel());
    TORCH_CHECK(
        is_power_of_two(n),
        "bandwidth_add_mul requires numel() to be a power-of-two so we can use a fast bitmask."
    );
    TORCH_CHECK((static_cast<uint64_t>(stride) & 1ULL) == 1ULL, "stride must be odd for power-of-two permutations");
    TORCH_CHECK(n <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()), "numel() must fit in uint32");

    at::cuda::CUDAGuard guard(a.device());
    const auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(std::min<uint64_t>((n + kThreads - 1) / kThreads, 65535ULL));
    const uint32_t mask = static_cast<uint32_t>(n - 1);
    const uint32_t stride_u32 = static_cast<uint32_t>(stride);

    for (int64_t pass = 0; pass < passes; ++pass) {
        bandwidth_add_mul_kernel<<<blocks, kThreads, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            n,
            stride_u32,
            mask
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ch13

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bandwidth_add_mul", &ch13::bandwidth_add_mul, "Bandwidth add/mul kernel with index permutation");
}
