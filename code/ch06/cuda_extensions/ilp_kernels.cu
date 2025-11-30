// ilp_kernels.cu - CUDA kernels for Instruction-Level Parallelism benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "profiling_helpers.cuh"

// Baseline: Sequential operations (low ILP)
__global__ void sequential_ops_kernel(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Sequential operations - each depends on previous
        float val = input[idx];
        val = val * 2.0f;      // Op 1
        val = val + 1.0f;      // Op 2 (depends on Op 1)
        val = val * 3.0f;      // Op 3 (depends on Op 2)
        val = val - 5.0f;      // Op 4 (depends on Op 3)
        output[idx] = val;
    }
}

// Optimized: Independent operations (high ILP)
// Computes SAME function as sequential: output = ((input * 2 + 1) * 3) - 5 = input * 6 - 2
// But uses independent partial computations that expose ILP to the compiler/hardware
__global__ void independent_ops_kernel(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Independent operations that combine to same result as sequential
        // Sequential computes: ((val * 2 + 1) * 3) - 5 = val * 6 + 3 - 5 = val * 6 - 2
        // We compute: (val * 3) + (val * 3) - 2 = val * 6 - 2 (same result!)
        float val = input[idx];
        float part1 = val * 3.0f;   // Independent: 3x
        float part2 = val * 3.0f;   // Independent: 3x (same op, different register)
        float part3 = -2.0f;        // Independent: constant
        
        // Combine independent results: 3x + 3x - 2 = 6x - 2
        output[idx] = part1 + part2 + part3;
    }
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
        
        // Process all 4 independently (exposes ILP)
        float res0 = val0 * 2.0f + 1.0f;
        float res1 = val1 * 3.0f - 5.0f;
        float res2 = val2 * 4.0f + 2.0f;
        float res3 = val3 * 5.0f - 3.0f;
        
        output[base_idx] = res0;
        output[base_idx + 1] = res1;
        output[base_idx + 2] = res2;
        output[base_idx + 3] = res3;
    } else {
        // Handle remainder elements
        for (int i = 0; i < 4 && base_idx + i < N; ++i) {
            float val = input[base_idx + i];
            output[base_idx + i] = val * 2.0f + 1.0f;
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
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
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
    // Each thread processes 4 elements, so fewer blocks needed
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

