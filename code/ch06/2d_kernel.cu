// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// 2d_kernel.cu
// 2D kernel example for processing images/matrices

#include <cuda_runtime.h>
#include <iostream>

//-------------------------------------------------------
// Kernel: my2DKernel running on the device (GPU)
// - input : device pointer to float array of size width×height
// - width : number of columns
// - height : number of rows
//-------------------------------------------------------
__global__ void my2DKernel(float* input, int width, int height) {
    // Compute 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process valid pixels
    if (x < width && y < height) {
        int idx = y * width + x;
        input[idx] *= 2.0f;
    }
}

int main() {
    // Image dimensions
    const int width = 1024;
    const int height = 1024;
    const int N = width * height;
    
    // 1) Allocate and initialize host image
    float* h_image = nullptr;
    cudaMallocHost(&h_image, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_image[i] = 1.0f; // e.g., initialize all pixels to 1.0f
    }
    
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // 2) Allocate device image and copy data to device
    float* d_image = nullptr;
    cudaMallocAsync(&d_image, N * sizeof(float), stream);
    cudaMemcpyAsync(d_image, h_image, N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    // 3) Configure and launch the 2D kernel
    dim3 threadsPerBlock2D(16, 16); // 256 threads per block
    dim3 blocksPerGrid2D(
        (width + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
        (height + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y
    );
    
    // 4) launch the kernel
    my2DKernel<<<blocksPerGrid2D, threadsPerBlock2D, 0, stream>>>(d_image, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaStreamDestroy(stream);
        cudaFreeHost(h_image);
        return 1;
    }
    
    // 5) wait for kernel to finish
    cudaMemcpyAsync(h_image, d_image, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 6) Verify a sample element
    std::cout << "h_image[0] = " << h_image[0] << std::endl;
    
    // 7) Cleanup
    cudaFreeAsync(d_image, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_image);
    
    return 0;
}
