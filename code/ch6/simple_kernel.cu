// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// simple_kernel.cu
// Improved version with dynamic block/grid calculation + stream-ordered allocation

#include <cuda_runtime.h>
#include <stdio.h>

//-------------------------------------------------------
// Kernel: myKernel running on the device (GPU)
// - input : device pointer to float array of length N
// - N : total number of elements in the input
//-------------------------------------------------------
__global__ void myKernel(float* input, int N) {
    // Compute a unique global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process valid elements
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

// This code runs on the host (CPU)
int main() {
    // 1) Problem size: one million floats
    const int N = 1'000'000;
    float* h_input = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));
    
    // Allocate input array of size N on the host (h_)
    // Initialize host data (for example, all ones)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }
    
    // Stream-ordered memory pool demo (CUDA 13 best practice)
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    float* d_input = nullptr;
    cudaMallocAsync(&d_input, N * sizeof(float), stream);
    
    cudaMemcpyAsync(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // 2) Tune launch parameters
    const int threadsPerBlock = 256; // multiple of 32
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 3,907, in this case
    
    // Launch myKernel across blocksPerGrid number of blocks
    // Each block has threadsPerBlock number of threads
    // Pass a reference to the d_input device array
    myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, N);
    
    // When finished, copy the results (stored in d_input) from the device back to the host (stored in h_input)
    cudaMemcpyAsync(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Verify results
    printf("First 5 values after doubling: %.1f %.1f %.1f %.1f %.1f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    
    // Cleanup: Free memory on the device and host
    cudaFreeAsync(d_input, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_input);
    
    return 0; // return 0 for success!
}
