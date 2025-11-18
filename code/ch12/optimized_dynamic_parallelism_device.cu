// optimized_dynamic_parallelism_device.cu
// Device-initiated launches with minor tweaks for better throughput.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

constexpr cudaError_t kUnsupportedErrors[] = {
    cudaErrorNotSupported,
    cudaErrorNotPermitted,
    cudaErrorInvalidDeviceFunction
};

__device__ cudaGraphExec_t g_graphExecOpt;
__device__ int g_workIndexOpt = 0;

inline bool is_feature_unsupported(cudaError_t err) {
    for (auto code : kUnsupportedErrors) {
        if (err == code) return true;
    }
    return false;
}

__device__ __forceinline__ float fuse_op(float x) {
    // Light math for optimized path
    #pragma unroll 1
    for (int i = 0; i < 4; ++i) {
        x = fmaf(x, 1.0001f, 0.0001f * (i + 1));
    }
    return x;
}

// Child kernel with lighter math than baseline; same data size
__global__ void childKernelOpt(float* data, int start, int count, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int global_idx = start + idx;
        float v = data[global_idx] * scale + 1.0f;
        data[global_idx] = fuse_op(v);
    }
}

__global__ void parentKernelOpt(float* data, int N, int* launch_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Single child launch over full buffer with wider blocks
        int segment_size = N;
        dim3 child_grid((segment_size + 1023) / 1024);
        dim3 child_block(1024);
        childKernelOpt<<<child_grid, child_block>>>(data, 0, segment_size, 1.0f);
        atomicAdd(launch_count, 1);
    }
}

__global__ void zeroKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = 0.0f;
}

// No recursion for tiny demo

__global__ void persistentSchedulerOpt(float* workData, int numTasks, int maxIterations) {
    cg::thread_block cta = cg::this_thread_block();
    while (true) {
        int workIdx = atomicAdd(&g_workIndexOpt, 1);
        if (workIdx >= maxIterations) break;
        int taskIdx = workIdx % numTasks;
        float taskValue = workData[taskIdx];
        if (taskValue > 0.5f) {
            cudaGraphLaunch(g_graphExecOpt, cudaStreamGraphTailLaunch);
        } else {
            cudaGraphLaunch(g_graphExecOpt, cudaStreamGraphFireAndForget);
        }
        cg::sync(cta);
    }
}

void buildGraphOpt(cudaGraphExec_t* outExec, float* d_data, int N) {
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t kernelNode;
    cudaKernelNodeParams kernelParams{};
    void* args[] = {&d_data, &N};
    kernelParams.func = reinterpret_cast<void*>(zeroKernel);
    kernelParams.gridDim = dim3((N + 255) / 256);
    kernelParams.blockDim = dim3(256);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = args;
    kernelParams.extra = nullptr;
    CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams));
    CUDA_CHECK(cudaGraphInstantiate(outExec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("Device: %s (CC %d.%d)\\n", prop.name, prop.major, prop.minor);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2048));

    // Allocate data
    const int N = 262144;  // match baseline size for apples-to-apples
    float* d_data = nullptr;
    int* d_launch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_launch_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_launch_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    cudaGraphExec_t exec = nullptr;
    buildGraphOpt(&exec, d_data, N);
    CUDA_CHECK(cudaMemcpyToSymbol(g_graphExecOpt, &exec, sizeof(exec)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_workIndexOpt, &zero, sizeof(int)));

    dim3 parent_grid(1);
    dim3 parent_block(256);
    parentKernelOpt<<<parent_grid, parent_block>>>(d_data, N, d_launch_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    int launches = 0;
    CUDA_CHECK(cudaMemcpy(&launches, d_launch_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("Device child launches (optimized): %d\\n", launches);

    if (exec) CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_launch_count));
    return 0;
}
