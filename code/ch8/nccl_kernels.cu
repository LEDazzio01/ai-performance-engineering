#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

constexpr int kNcclThreads = 256;

__global__ void nccl_ring_kernel(
    const float* __restrict__ chunks,
    float* __restrict__ output,
    int world_size,
    int chunk_elems) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_elems) {
        return;
    }

    float sum = 0.0f;
    for (int rank = 0; rank < world_size; ++rank) {
        sum += chunks[rank * chunk_elems + idx];
    }
    output[idx] = sum;
}

}  // namespace

void nccl_ring_reduce(torch::Tensor chunks, torch::Tensor output) {
    TORCH_CHECK(chunks.is_cuda(), "chunks must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(chunks.dtype() == torch::kFloat32, "chunks must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(chunks.dim() == 2, "chunks must be [world, elems]");
    TORCH_CHECK(output.numel() == chunks.size(1), "output must match chunk size");

    at::cuda::CUDAGuard guard(chunks.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int elems = static_cast<int>(chunks.size(1));
    const dim3 grid((elems + kNcclThreads - 1) / kNcclThreads);
    nccl_ring_kernel<<<grid, kNcclThreads, 0, stream>>>(
        chunks.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(chunks.size(0)),
        elems);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nccl_ring_reduce", &nccl_ring_reduce, "NCCL-style ring reduction");
}
