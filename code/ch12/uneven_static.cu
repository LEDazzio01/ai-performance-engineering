#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void static_partition_kernel(const float* in, float* out, int elems, int start, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int global_idx = start + idx;
        if (global_idx < elems) {
            out[global_idx] = in[global_idx] * in[global_idx];
        }
    }
}

int main() {
    constexpr int elems = 1024 + 300;  // uneven size
    std::vector<float> input(elems);
    for (int i = 0; i < elems; ++i) {
        input[i] = static_cast<float>(i);
    }
    std::vector<float> output(elems, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, elems * sizeof(float));
    cudaMalloc(&d_out, elems * sizeof(float));

    cudaMemcpy(d_in, input.data(), elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, elems * sizeof(float));

    int full_tiles = elems / 512;
    int tail = elems % 512;

    for (int tile = 0; tile < full_tiles; ++tile) {
        int start = tile * 512;
        static_partition_kernel<<<2, 256>>>(d_in, d_out, elems, start, 512);
    }
    if (tail > 0) {
        int start = full_tiles * 512;
        int blocks = (tail + 255) / 256;
        static_partition_kernel<<<blocks, 256>>>(d_in, d_out, elems, start, tail);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_out, elems * sizeof(float), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (int i = 0; i < elems; ++i) {
        double err = std::fabs(output[i] - input[i] * input[i]);
        max_err = std::max(max_err, err);
    }
    std::printf("Static uneven partition completed. max_err=%.3e\n", max_err);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
