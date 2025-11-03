#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void child_scale_kernel(float* data, float scale, int elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems) {
        data[idx] *= scale;
    }
}

__global__ void parent_kernel(float* data, int elems) {
    if (threadIdx.x == 0) {
        int threads = 128;
        int blocks = (elems + threads - 1) / threads;
        child_scale_kernel<<<blocks, threads>>>(data, 3.0f, elems);
    }
}

int main() {
    constexpr int elems = 1 << 15;
    std::vector<float> host(elems, 1.0f);

    float* device = nullptr;
    cudaMalloc(&device, elems * sizeof(float));
    cudaMemcpy(device, host.data(), elems * sizeof(float), cudaMemcpyHostToDevice);

    parent_kernel<<<1, 32>>>(device, elems);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(device);
        return -1;
    }

    cudaMemcpy(host.data(), device, elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device);

    std::printf("Device-launched child kernel executed. Sample value=%.1f\n", host.front());
    return 0;
}
