// graph_bandwidth_kernels.cu - CUDA kernels for graph bandwidth benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "profiling_helpers.cuh"

// Simple memory copy kernel
__global__ void copy_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        TORCH_CHECK(_status == cudaSuccess,                                    \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(_status));                              \
    } while (0)

} // anonymous namespace

void separate_kernel_launches(torch::Tensor dst, torch::Tensor src, int iterations) {
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA tensor");
    TORCH_CHECK(src.is_cuda(), "src must be CUDA tensor");
    TORCH_CHECK(dst.dtype() == torch::kFloat32, "dst must be float32");
    TORCH_CHECK(src.dtype() == torch::kFloat32, "src must be float32");
    
    // Ensure we're on the correct device
    int device_id = dst.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = src.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr) - this is the legacy default stream
    // PyTorch operations on default stream will be properly synchronized
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("separate_kernel_launches");
        for (int i = 0; i < iterations; ++i) {
            copy_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
                dst.data_ptr<float>(),
                src.data_ptr<float>(),
                n
            );
        }
        CHECK_CUDA(cudaGetLastError());
    }
    // Note: No explicit synchronization here - PyTorch benchmark harness handles synchronization
}

struct GraphCache {
    bool initialized = false;
    int n = 0;
    int device_id = -1;
    float* dst_ptr = nullptr;
    const float* src_ptr = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t exec = nullptr;
};

static GraphCache g_graph_cache;

static void destroy_graph_cache() {
    if (g_graph_cache.initialized) {
        cudaGraphExecDestroy(g_graph_cache.exec);
        cudaStreamDestroy(g_graph_cache.stream);
        g_graph_cache = {};
    }
}

void graph_kernel(torch::Tensor dst, torch::Tensor src, int iterations) {
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA tensor");
    TORCH_CHECK(src.is_cuda(), "src must be CUDA tensor");
    TORCH_CHECK(dst.dtype() == torch::kFloat32, "dst must be float32");
    TORCH_CHECK(src.dtype() == torch::kFloat32, "src must be float32");
    
    int n = src.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    int device_id = dst.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    float* dst_ptr = dst.data_ptr<float>();
    const float* src_ptr = src.data_ptr<float>();
    
    bool rebuild = !g_graph_cache.initialized ||
                   g_graph_cache.n != n ||
                   g_graph_cache.device_id != device_id ||
                   g_graph_cache.dst_ptr != dst_ptr ||
                   g_graph_cache.src_ptr != src_ptr;
    
    if (rebuild) {
        destroy_graph_cache();
        
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        try {
            CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            copy_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
                dst_ptr,
                src_ptr,
                n
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
            CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        } catch (...) {
            if (exec != nullptr) {
                cudaGraphExecDestroy(exec);
            }
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
            throw;
        }
        
        cudaGraphDestroy(graph);
        g_graph_cache.initialized = true;
        g_graph_cache.n = n;
        g_graph_cache.device_id = device_id;
        g_graph_cache.dst_ptr = dst_ptr;
        g_graph_cache.src_ptr = src_ptr;
        g_graph_cache.stream = stream;
        g_graph_cache.exec = exec;
    }
    
    {
        PROFILE_KERNEL_LAUNCH("graph_kernel");
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUDA(cudaGraphLaunch(g_graph_cache.exec, g_graph_cache.stream));
        }
    }
    
    CHECK_CUDA(cudaStreamSynchronize(g_graph_cache.stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("separate_kernel_launches", &separate_kernel_launches, "Separate kernel launches (baseline)");
    m.def("graph_kernel", &graph_kernel, "CUDA graph kernel (optimized)");
}
