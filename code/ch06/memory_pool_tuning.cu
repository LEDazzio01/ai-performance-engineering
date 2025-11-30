// memory_pool_tuning.cu
// Demonstrates tuning the default stream-ordered memory pool on CUDA 13.

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

int main() {
  cudaMemPool_t pool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, /*device=*/0));

  // Set a 1 GiB release threshold so the pool retains memory for reuse.
  unsigned long long release_threshold = 1ull << 30;
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &release_threshold));

  // Make reuse more aggressive when stream dependencies are satisfied.
  int enable = 1;
  CUDA_CHECK(cudaMemPoolSetAttribute(
      pool, cudaMemPoolReuseFollowEventDependencies, &enable));
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowOpportunistic,
                                     &enable));
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies,
                                     &enable));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  void* buffer = nullptr;
  size_t bytes = 64 * 1024 * 1024;  // 64 MiB sample allocation
  CUDA_CHECK(cudaMallocAsync(&buffer, bytes, stream));
  CUDA_CHECK(cudaFreeAsync(buffer, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("Default memory pool tuned and sample allocation completed.\n");
  return 0;
}
