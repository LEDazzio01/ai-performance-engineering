#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void scale_kernel(float* data, float scale, int elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems) {
        data[idx] *= scale;
    }
}

int main() {
    constexpr int elems = 1 << 16;
    constexpr int threads = 256;
    constexpr int blocks = (elems + threads - 1) / threads;

    std::vector<float> host(elems, 1.0f);
    float* device = nullptr;
    cudaMalloc(&device, elems * sizeof(float));
    cudaMemcpy(device, host.data(), elems * sizeof(float), cudaMemcpyHostToDevice);

    scale_kernel<<<blocks, threads>>>(device, 2.0f, elems);
    scale_kernel<<<blocks, threads>>>(device, 0.5f, elems);
    scale_kernel<<<blocks, threads>>>(device, 4.0f, elems);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(device);
        return -1;
    }

    cudaMemcpy(host.data(), device, elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device);

    double max_err = 0.0;
    for (float v : host) {
        max_err = std::max(max_err, std::abs(static_cast<double>(v) - 2.0));
    }
    std::printf("Host-launched dynamic parallelism baseline complete. max_err=%.3e\n", max_err);
    return 0;
}

