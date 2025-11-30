// bank_conflicts_kernels.cu - CUDA kernels for shared memory bank conflicts benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "profiling_helpers.cuh"

#define SHARED_SIZE 1024
#define CONFLICT_STRIDE 32
#define CONFLICT_ITERS 64

// Baseline: Bank conflicts occur when threads access same bank
// All threads in a warp read from same bank (stride 32 = bank 0 always)
__global__ void bank_conflicts_kernel(float* output, const float* input, int N) {
    // Shared memory sized to block dimension
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory - each thread loads one element
    if (idx < N) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Access pattern causing bank conflicts: all threads in warp access same bank
    // We simulate this by having each thread accumulate from strided positions
    if (idx < N) {
        float acc = 0.0f;
        int block_size = blockDim.x;
        // Access pattern that causes conflicts: stride by 32 (bank size)
        // All accesses hit bank 0 causing 32-way conflicts
        #pragma unroll
        for (int iter = 0; iter < CONFLICT_ITERS; ++iter) {
            // Access element at (tid + iter * 32) % block_size
            // This causes all threads to hit the same bank in shared memory
            int conflict_idx = (iter * CONFLICT_STRIDE) % block_size;
            acc += shared_data[conflict_idx];
        }
        output[idx] = acc + shared_data[tid];  // Include own element for unique output
    }
}

// Optimized: Eliminate bank conflicts using padding
// Computes SAME result as baseline but avoids bank conflicts via stride adjustment
__global__ void bank_conflicts_padded_kernel(float* output, const float* input, int N) {
    // Add padding: +1 element per 32 to offset bank access patterns
    extern __shared__ float shared_data[];  // Size will be blockDim.x + blockDim.x/32
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = blockDim.x;
    
    // Load data into shared memory with padding offset
    // Element i goes to position i + (i / 32) to spread across banks
    if (idx < N) {
        int padded_tid = tid + (tid / 32);
        shared_data[padded_tid] = input[idx];
    }
    __syncthreads();
    
    // Same computation as baseline, but with padded indexing to avoid conflicts
    if (idx < N) {
        float acc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < CONFLICT_ITERS; ++iter) {
            // Same logical access as baseline
            int conflict_idx = (iter * CONFLICT_STRIDE) % block_size;
            // Padded index avoids bank conflicts
            int padded_conflict_idx = conflict_idx + (conflict_idx / 32);
            acc += shared_data[padded_conflict_idx];
        }
        // Include own element with padded access
        int padded_tid = tid + (tid / 32);
        output[idx] = acc + shared_data[padded_tid];  // Same result as baseline
    }
}

void bank_conflicts(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // Dynamic shared memory: one float per thread
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("bank_conflicts");
        bank_conflicts_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
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

void bank_conflicts_padded(torch::Tensor output, torch::Tensor input) {
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // Dynamic shared memory: one float per thread + padding (1 extra per 32 elements)
    size_t shared_mem_size = (threads_per_block + threads_per_block / 32) * sizeof(float);
    
    // Use default stream (nullptr)
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("bank_conflicts_padded");
        bank_conflicts_padded_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
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
    m.def("bank_conflicts", &bank_conflicts, "Bank conflicts kernel (baseline)");
    m.def("bank_conflicts_padded", &bank_conflicts_padded, "Bank conflicts kernel with padding (optimized)");
}
