// cuda_helpers.cuh - Common CUDA utility functions and macros
// Used across all chapter examples for consistent error handling and timing

#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// CUDA kernel launch checking
#define CUDA_CHECK_LAST_ERROR()                                              \
  do {                                                                       \
    cudaError_t status = cudaGetLastError();                                 \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA kernel error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(status));        \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Simple CUDA timer class
class CudaTimer {
private:
  cudaEvent_t start_event, stop_event;
  bool started;

public:
  CudaTimer() : started(false) {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~CudaTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  void start() {
    CUDA_CHECK(cudaEventRecord(start_event));
    started = true;
  }

  float stop() {
    if (!started) {
      std::fprintf(stderr, "Error: Timer not started\n");
      return 0.0f;
    }
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    started = false;
    return milliseconds;
  }
};

// Get GPU properties helper
inline void printGpuInfo(int device = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  
  std::printf("GPU %d: %s\n", device, prop.name);
  std::printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
  std::printf("  SMs: %d\n", prop.multiProcessorCount);
  std::printf("  Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  std::printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
  std::printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
  std::printf("  Warp size: %d\n", prop.warpSize);
}

// Calculate grid dimensions helper
inline dim3 calculateGridDim(int n, int blockSize) {
  return dim3((n + blockSize - 1) / blockSize);
}

inline dim3 calculateGridDim2D(int width, int height, int blockSizeX, int blockSizeY) {
  return dim3((width + blockSizeX - 1) / blockSizeX, 
              (height + blockSizeY - 1) / blockSizeY);
}

// Bandwidth calculation helper (GB/s)
inline float calculateBandwidthGBs(size_t bytes, float milliseconds) {
  return (bytes / (1024.0f * 1024.0f * 1024.0f)) / (milliseconds / 1000.0f);
}

// FLOPS calculation helper (GFLOPS)
inline float calculateGFLOPS(size_t flops, float milliseconds) {
  return (flops / 1e9f) / (milliseconds / 1000.0f);
}

#endif // CUDA_HELPERS_CUH

