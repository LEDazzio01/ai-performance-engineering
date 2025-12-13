// ilp_kernels.cu - CUDA kernels for Instruction-Level Parallelism benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "profiling_helpers.cuh"

// Baseline: Sequential operations (low ILP)
__global__ void sequential_ops_kernel(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Dependent chain (intentionally non-linear so the compiler can't
        // trivially collapse it into a single fused op).
        float val = input[idx];
        val = val * val + 0.10f;
        val = val * val + 0.20f;
        val = val * val + 0.30f;
        val = val * val + 0.40f;
        output[idx] = val;
    }
}

// Optimized: unroll + multiple independent registers per thread (higher ILP).
// Computes the SAME function as sequential_ops_kernel for every element.
__global__ void independent_ops_kernel(float* output, const float* input, int N) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base_idx >= N) return;

    float v0 = input[base_idx];
    float v1 = (base_idx + 1 < N) ? input[base_idx + 1] : 0.0f;
    float v2 = (base_idx + 2 < N) ? input[base_idx + 2] : 0.0f;
    float v3 = (base_idx + 3 < N) ? input[base_idx + 3] : 0.0f;

    // Interleave dependent chains across 4 independent values.
    v0 = v0 * v0 + 0.10f; v1 = v1 * v1 + 0.10f; v2 = v2 * v2 + 0.10f; v3 = v3 * v3 + 0.10f;
    v0 = v0 * v0 + 0.20f; v1 = v1 * v1 + 0.20f; v2 = v2 * v2 + 0.20f; v3 = v3 * v3 + 0.20f;
    v0 = v0 * v0 + 0.30f; v1 = v1 * v1 + 0.30f; v2 = v2 * v2 + 0.30f; v3 = v3 * v3 + 0.30f;
    v0 = v0 * v0 + 0.40f; v1 = v1 * v1 + 0.40f; v2 = v2 * v2 + 0.40f; v3 = v3 * v3 + 0.40f;

    output[base_idx] = v0;
    if (base_idx + 1 < N) output[base_idx + 1] = v1;
    if (base_idx + 2 < N) output[base_idx + 2] = v2;
    if (base_idx + 3 < N) output[base_idx + 3] = v3;
}

// Further optimized: Loop unrolling to expose more ILP
__global__ void unrolled_ilp_kernel(float* output, const float* input, int N) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process 4 elements per thread with independent operations
    if (base_idx + 3 < N) {
        float val0 = input[base_idx];
        float val1 = input[base_idx + 1];
        float val2 = input[base_idx + 2];
        float val3 = input[base_idx + 3];
        
        val0 = val0 * val0 + 0.10f; val1 = val1 * val1 + 0.10f; val2 = val2 * val2 + 0.10f; val3 = val3 * val3 + 0.10f;
        val0 = val0 * val0 + 0.20f; val1 = val1 * val1 + 0.20f; val2 = val2 * val2 + 0.20f; val3 = val3 * val3 + 0.20f;
        val0 = val0 * val0 + 0.30f; val1 = val1 * val1 + 0.30f; val2 = val2 * val2 + 0.30f; val3 = val3 * val3 + 0.30f;
        val0 = val0 * val0 + 0.40f; val1 = val1 * val1 + 0.40f; val2 = val2 * val2 + 0.40f; val3 = val3 * val3 + 0.40f;

        output[base_idx] = val0;
        output[base_idx + 1] = val1;
        output[base_idx + 2] = val2;
        output[base_idx + 3] = val3;
    } else {
        // Handle remainder elements
        for (int i = 0; i < 4 && base_idx + i < N; ++i) {
            float val = input[base_idx + i];
            val = val * val + 0.10f;
            val = val * val + 0.20f;
            val = val * val + 0.30f;
            val = val * val + 0.40f;
            output[base_idx + i] = val;
        }
    }
}

void sequential_ops(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("sequential_ops");
        sequential_ops_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
        // Synchronize to catch kernel execution errors
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
        }
    }
}

void independent_ops(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    
    int N = input.size(0);
    int threads_per_block = 256;
    // Each thread processes 4 elements.
    int num_blocks = ((N + 3) / 4 + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("independent_ops");
        independent_ops_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
        // Synchronize to catch kernel execution errors
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
        }
    }
}

void unrolled_ilp(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = ((N + 3) / 4 + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("unrolled_ilp");
        unrolled_ilp_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        }
        // Synchronize to catch kernel execution errors
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sequential_ops", &sequential_ops, "Sequential operations kernel (baseline, low ILP)");
    m.def("independent_ops", &independent_ops, "Independent operations kernel (optimized, high ILP)");
    m.def("unrolled_ilp", &unrolled_ilp, "Unrolled ILP kernel (optimized, high ILP)");
}
