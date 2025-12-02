// nvtx_profiling.cu -- NVTX ranges for inference pipelines (CUDA 13 / NVTX v3 header-only).
// Build (Linux):
//   nvcc nvtx_profiling.cu -std=c++17 -O3 -lineinfo -o nvtx_profiling
// NOTE: NVTX v3 is header-only on CUDA 12.9+/13.0; we ship a stub -lnvToolsExt
// archive so existing build systems can continue linking with the familiar flag.

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

struct Token { int id; };

// CUDA kernel must be defined outside class (global functions cannot be member functions)
__global__ void simple_model_dummy_kernel(const float* weights,
                                          float* cache,
                                          int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    cache[idx] = weights[idx] + cache[idx] * 0.99f;
  }
}

class SimpleModel {
 public:
  explicit SimpleModel(int dim = 2048) : dim_(dim) {
    CUDA_CHECK(cudaMalloc(&weights_, dim_ * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cache_, dim_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(weights_, 0, dim_ * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(cache_, 0, dim_ * sizeof(float)));
  }
  ~SimpleModel() {
    CUDA_CHECK(cudaFree(weights_));
    CUDA_CHECK(cudaFree(cache_));
  }

  void encode(const std::vector<Token>& prompt) {
    nvtxRangePush("encode");
    for (size_t i = 0; i < prompt.size(); ++i) {
      char label[32];
      std::snprintf(label, sizeof(label), "token_%zu", i);
      nvtxRangePush(label);
      run_kernel("attention", 0xFF00FF00);
      run_kernel("feedforward", 0xFF0000FF);
      nvtxRangePop();
    }
    nvtxRangePop();
  }

  Token decode() {
    nvtxRangePush("decode_step");
    run_kernel("attention", 0xFF00FF00);
    run_kernel("feedforward", 0xFF0000FF);
    nvtxRangePop();
    return Token{42};
  }

 private:
  void run_kernel(const char* name, uint32_t color) {
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
    dim3 block(256);
    dim3 grid((dim_ + block.x - 1) / block.x);
    simple_model_dummy_kernel<<<grid, block>>>(weights_, cache_, dim_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();
  }

  int dim_;
  float* weights_;
  float* cache_;
};

int main() {
  std::printf("Build example: nvcc nvtx_profiling.cu -std=c++17 -O3 -lineinfo -o nvtx_profiling\n");

  SimpleModel model;
  std::vector<Token> prompt(32);

  nvtxRangePush("Inference");
  model.encode(prompt);
  for (int i = 0; i < 5; ++i) {
    char label[32];
    std::snprintf(label, sizeof(label), "decode_%d", i);
    nvtxRangePush(label);
    model.decode();
    nvtxRangePop();
  }
  nvtxRangePop();

  return 0;
}
