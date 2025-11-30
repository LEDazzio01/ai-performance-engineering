// independent_ops.cu -- ILP demo with two independent operations per thread.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1 << 20;

__global__ void independent_ops(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = a[idx];
    float y = b[idx];
    float u = x * x;
    float v = y * y;
    out[idx] = u + v;
  }
}

int main() {
  float *h_a, *h_b, *h_out;
  cudaMallocHost(&h_a, N * sizeof(float));
  cudaMallocHost(&h_b, N * sizeof(float));
  cudaMallocHost(&h_out, N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(N - i);
  }

  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  independent_ops<<<grid, block>>>(d_a, d_b, d_out, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("out[0]=%.1f\n", h_out[0]);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_out);
  return 0;
}
