#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void child_tail_kernel(const float* in, float* out, int start, int elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems) {
        int global_idx = start + idx;
        out[global_idx] = in[global_idx] + 1.0f;
    }
}

__global__ void parent_dynamic_kernel(const float* in, float* out, int elems) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tile = gridDim.x * blockDim.x;
    int tail = elems % tile;
    int limit = elems - tail;

    if (global_idx < limit) {
        out[global_idx] = in[global_idx] + 1.0f;
    }

    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0 && tail > 0) {
        int start = limit;
        int threads = 64;
        int blocks = (tail + threads - 1) / threads;
        child_tail_kernel<<<blocks, threads>>>(in, out, start, tail);
    }
}

int main() {
    constexpr int elems = 10'000 + 123;
    std::vector<float> input(elems, 5.0f);
    std::vector<float> output(elems, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, elems * sizeof(float));
    cudaMalloc(&d_out, elems * sizeof(float));
    cudaMemcpy(d_in, input.data(), elems * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(40);
    dim3 block(256);
    parent_dynamic_kernel<<<grid, block>>>(d_in, d_out, elems);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_out, elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    double max_err = 0.0;
    for (int i = 0; i < elems; ++i) {
        max_err = std::max(max_err, std::abs(output[i] - 6.0));
    }
    std::printf("Dynamic uneven partition completed. max_err=%.3e\n", max_err);
    return 0;
}
