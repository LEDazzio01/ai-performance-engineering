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
  
  // CUDA 13.0 API: cudaMemPrefetchAsync requires cudaMemLocation struct
  struct cudaMemLocation gpuLoc;
  gpuLoc.type = cudaMemLocationTypeDevice;
  gpuLoc.id = device;
  cudaMemPrefetchAsync(data, bytes, gpuLoc, 0, 0);

  int block = 256;
  int grid = (N + block - 1) / block;
  kernel<<<grid, block>>>(data, N);
  cudaDeviceSynchronize();

  struct cudaMemLocation cpuLoc;
  cpuLoc.type = cudaMemLocationTypeHost;
  cpuLoc.id = 0;
  cudaMemPrefetchAsync(data, bytes, cpuLoc, 0, 0);
  cudaDeviceSynchronize();

  printf("First value: %.1f\n", data[0]);

  cudaFree(data);
  return 0;
}
