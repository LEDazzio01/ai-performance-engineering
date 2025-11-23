// optimized_cuda_graphs_conditional_enhanced.cu -- CUDA graph static path (optimized).

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int N = 1 << 20;

__global__ void expensive_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            val = sqrtf(val * val + scale) * 0.99f;
        }
        data[idx] = val;
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    bool is_sm12x = (prop.major >= 12);
    bool is_sm10x = (prop.major == 10);
    bool supports_graphs = (prop.major >= 7 && prop.minor >= 5) || prop.major >= 8;
    
    std::printf("=== Optimized Conditional Graphs Enhanced (CUDA Graph Static Path) ===\n");
    std::printf("Architecture: %s (SM %d.%d)\n", 
                is_sm12x ? "Grace-Blackwell GB10" : is_sm10x ? "Blackwell B200/B300" : "Other",
                prop.major, prop.minor);
    
    if (!supports_graphs) {
        std::printf("CUDA Graphs require compute capability 7.5 or newer.\n");
        return 0;
    }
    
    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f + (i % 100) * 0.01f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaStream_t graph_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking));
    
    cudaGraph_t graph_static;
    cudaGraphExec_t graph_exec_static;
    
    CUDA_CHECK(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal));
    expensive_kernel<<<grid, block, 0, graph_stream>>>(d_data, N, 1.01f);
    CUDA_CHECK(cudaStreamEndCapture(graph_stream, &graph_static));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_static, graph_static, nullptr, nullptr, 0));
    
    constexpr int ITERS = 1000;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, graph_stream));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec_static, graph_stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, graph_stream));
    CUDA_CHECK(cudaStreamSynchronize(graph_stream));
    
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    std::printf("Optimized (CUDA graph static path): %.2f ms (%.3f ms/iter)\n", ms, ms / ITERS);
    std::printf("Note: Fixed path, no dynamic branching\n");
    
    if (is_sm10x || is_sm12x) {
        std::printf("\n✅ Blackwell Optimizations:\n");
        std::printf("  • Hardware support for conditional execution\n");
        std::printf("  • Reduced kernel launch overhead in graphs\n");
        std::printf("  • Better overlap with memory operations\n");
    }
    
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec_static));
    CUDA_CHECK(cudaGraphDestroy(graph_static));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(graph_stream));
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}
