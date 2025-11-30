// unified_memory.cu
// Minimal example using CUDA managed memory with prefetching.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * data[idx] + 1.0f;
  }
}

int main() {
  constexpr int N = 1 << 20;
  size_t bytes = N * sizeof(float);

  float* data = nullptr;
  cudaMallocManaged(&data, bytes);

  for (int i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i);
  }

  int device = 0;
  cudaGetDevice(&device);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  cudaMemLocation gpu_loc{};
  gpu_loc.type = cudaMemLocationTypeDevice;
  gpu_loc.id = device;
  cudaMemPrefetchAsync(data, bytes, gpu_loc, /*flags=*/0, stream);

  int block = 256;
  int grid = (N + block - 1) / block;
  kernel<<<grid, block, 0, stream>>>(data, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaStreamDestroy(stream);
    cudaFree(data);
    return 1;
  }

  cudaMemLocation cpu_loc{};
  cpu_loc.type = cudaMemLocationTypeHost;
  cpu_loc.id = 0;
  cudaMemPrefetchAsync(data, bytes, cpu_loc, /*flags=*/0, stream);
  cudaStreamSynchronize(stream);

  printf("First value: %.1f\n", data[0]);

  cudaFree(data);
  cudaStreamDestroy(stream);
  return 0;
}
