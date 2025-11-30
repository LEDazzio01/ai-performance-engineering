// launch_bounds_kernels.cu - CUDA kernels for launch bounds benchmarks
// Can be loaded as PyTorch CUDA extension

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        TORCH_CHECK(_status == cudaSuccess,                                    \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(_status));                              \
    } while (0)

constexpr int kLaunchBoundsWorkIters = 96;
constexpr int kLaunchBoundsTransformRepeats = 3;
constexpr float kLaunchBoundsEps = 1e-6f;
constexpr int kBaselineThreadsPerBlock = 64;
constexpr int kOptimizedThreadsPerBlock = 256;

__device__ __forceinline__ float launch_bounds_workload(float value) {
    float acc0 = value * 1.0001f + 0.1f;
    float acc1 = value * 0.9997f - 0.05f;

    #pragma unroll
    for (int repeat = 0; repeat < kLaunchBoundsTransformRepeats; ++repeat) {
        #pragma unroll 4
        for (int iter = 0; iter < kLaunchBoundsWorkIters; ++iter) {
            const float coupled = (acc0 * acc1) * 0.00025f + (iter + 1 + repeat) * kLaunchBoundsEps;
            const float inv = rsqrtf(fabsf(acc0) + fabsf(acc1) + coupled + kLaunchBoundsEps);
            acc0 = fmaf(acc0, 1.00003f, inv * 0.0002f + coupled);
            acc1 = fmaf(acc1, 0.99991f, -inv * 0.00015f - coupled * 0.5f);
        }
        // Re-mix the registers to keep the dependency chain long.
        float mix = acc0 * 0.125f + acc1 * 0.875f;
        acc0 = mix * 1.00001f + acc1 * 0.0001f;
        acc1 = mix * 0.75f - acc0 * 0.00005f;
    }
    return acc0 + acc1;
}

} // anonymous namespace

// Kernel without launch bounds (baseline)
__global__ void kernel_no_launch_bounds(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = launch_bounds_workload(input[idx]);
    }
}

// Kernel with launch bounds annotation (optimized)
__global__ __launch_bounds__(kOptimizedThreadsPerBlock, 4)
void kernel_with_launch_bounds(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = launch_bounds_workload(input[idx]);
    }
}

void launch_bounds_baseline(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    int n = input.size(0);
    int threads_per_block = kBaselineThreadsPerBlock;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use PyTorch's current CUDA stream for consistency
    c10::cuda::CUDAGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    cudaStream_t stream_handle = stream.stream();
    
    for (int i = 0; i < iterations; ++i) {
        kernel_no_launch_bounds<<<num_blocks, threads_per_block, 0, stream_handle>>>(
            const_cast<float*>(input.data_ptr<float>()),
            output.data_ptr<float>(),
            n
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_handle));
}

void launch_bounds_optimized(torch::Tensor input, torch::Tensor output, int iterations) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "input and output must have same size");
    
    int n = input.size(0);
    int threads_per_block = kOptimizedThreadsPerBlock;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use PyTorch's current CUDA stream for consistency
    c10::cuda::CUDAGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    cudaStream_t stream_handle = stream.stream();
    
    for (int i = 0; i < iterations; ++i) {
        kernel_with_launch_bounds<<<num_blocks, threads_per_block, 0, stream_handle>>>(
            const_cast<float*>(input.data_ptr<float>()),
            output.data_ptr<float>(),
            n
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_handle));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_bounds_baseline", &launch_bounds_baseline, "Kernel without launch bounds (baseline)");
    m.def("launch_bounds_optimized", &launch_bounds_optimized, "Kernel with launch bounds (optimized)");
}
