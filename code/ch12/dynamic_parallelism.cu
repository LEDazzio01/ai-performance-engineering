// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// dynamic_parallelism.cu
// Device-initiated kernel launches and CUDA graph orchestration

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "../common/cuda13_teasers.cuh"
#endif

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

inline bool is_feature_unsupported(cudaError_t err) {
    for (auto code : kUnsupportedErrors) {
        if (err == code) {
            return true;
        }
    }
    return false;
}

// Child kernel launched by parent
__global__ void childKernel(float* data, int start, int count, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int global_idx = start + idx;
        data[global_idx] = data[global_idx] * scale + 1.0f;
    }
}

// Parent kernel that launches child kernels based on data conditions
__global__ void parentKernel(float* data, int N, int* launch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only some threads will launch child kernels
    if (idx < N && (idx % 1000) == 0) {
        // Check condition based on data value
        float value = data[idx];
        
        if (value > 0.5f) {
            // Launch child kernel for this segment
            int segment_size = min(100, N - idx);
            dim3 child_grid((segment_size + 63) / 64);
            dim3 child_block(64);
            
            // Dynamic kernel launch from device
            childKernel<<<child_grid, child_block>>>(data, idx, segment_size, 2.0f);
            
            // Increment launch counter atomically
            atomicAdd(launch_count, 1);
        }
    }
}

// Recursive kernel example
__global__ void recursiveKernel(float* data, int N, int depth, int max_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N && depth < max_depth) {
        // Process current level
        data[idx] = data[idx] * 0.9f + 0.1f;
        
        // Launch next level if conditions are met (limit recursion)
        if (depth < max_depth - 1 && (idx % (1 << (depth + 2))) == 0 && N > 256) {
            int next_N = N / 2;
            if (next_N > 0) {
                dim3 next_grid((next_N + 255) / 256);
                dim3 next_block(256);
                
                // Recursive launch (limit to avoid too many launches)
                if (next_grid.x <= 4) { // Limit grid size
                    recursiveKernel<<<next_grid, next_block>>>(data, next_N, depth + 1, max_depth);
                }
            }
        }
    }
}

// Persistent scheduler kernel with device-initiated graphs
__device__ cudaGraphExec_t g_graphExec; // Global graph executor
__device__ int g_workIndex = 0;         // Global work counter

__global__ void persistentScheduler(float* workData, int numTasks, int maxIterations) {
    cg::thread_block cta = cg::this_thread_block();
    while (true) {
        // Atomically get next work item
        int workIdx = atomicAdd(&g_workIndex, 1);
        
        if (workIdx >= maxIterations) break;
        
        // Decide which work to do based on data
        int taskIdx = workIdx % numTasks;
        float taskValue = workData[taskIdx];
        
        // Launch appropriate graph based on condition
        if (taskValue > 0.5f) {
            // Launch high-intensity graph
            cudaGraphLaunch(g_graphExec, cudaStreamGraphTailLaunch);
        } else {
            // Launch low-intensity graph  
            cudaGraphLaunch(g_graphExec, cudaStreamGraphFireAndForget);
        }
        
        // Small delay to simulate work
        for (int i = 0; i < 1000; ++i) {
            cta.sync();
        }
    }
}

// Work kernels for graph nodes
__global__ void workKernelHigh(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // High-intensity computation
        float x = data[idx];
        for (int i = 0; i < 10; ++i) {
            x = sinf(x) * cosf(x) + 0.1f;
        }
        data[idx] = x;
    }
}

__global__ void workKernelLow(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Low-intensity computation
        data[idx] = data[idx] * 1.1f + 0.05f;
    }
}

bool demonstrateBasicDynamicParallelism() {
    printf("=== Basic Dynamic Parallelism ===\n");
    
    const int N = 10000;
    const int bytes = N * sizeof(float);
    
    // Allocate and initialize data
    float *h_data = new float[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    
    float *d_data = nullptr;
    int *d_launch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_launch_count, sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_launch_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Launching parent kernel with %d elements...\n", N);
    
    // Launch parent kernel
    dim3 parent_grid((N + 255) / 256);
    dim3 parent_block(256);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    parentKernel<<<parent_grid, parent_block>>>(d_data, N, d_launch_count);
    CUDA_CHECK(cudaEventRecord(stop));

    cudaError_t launch_err = cudaGetLastError();
    if (is_feature_unsupported(launch_err)) {
        printf("⚠️  Skipping basic dynamic parallelism demo: %s\n", cudaGetErrorString(launch_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaFree(d_launch_count);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (launch_err != cudaSuccess) {
        printf("CUDA error launching parent kernel: %s\n", cudaGetErrorString(launch_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaFree(d_launch_count);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (is_feature_unsupported(sync_err)) {
        printf("⚠️  Device cannot execute dynamic parallelism workload (%s); skipping demo\n",
               cudaGetErrorString(sync_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaFree(d_launch_count);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (sync_err != cudaSuccess) {
        printf("CUDA error syncing parent kernel: %s\n", cudaGetErrorString(sync_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaFree(d_launch_count);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    // Check results
    int launch_count;
    CUDA_CHECK(cudaMemcpy(&launch_count, d_launch_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    printf("Execution time: %.2f ms\n", elapsed_ms);
    printf("Child kernels launched: %d\n", launch_count);
    printf("Sample results: %.3f, %.3f, %.3f\n", h_data[0], h_data[1000], h_data[5000]);
    
    // Cleanup
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_launch_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}

bool demonstrateRecursiveLaunches() {
    printf("\n=== Recursive Dynamic Parallelism ===\n");
    
    const int N = 8192;
    const int max_depth = 4;
    const int bytes = N * sizeof(float);
    
    float *h_data = new float[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;
    }
    
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    printf("Starting recursive kernel with depth %d...\n", max_depth);
    
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    recursiveKernel<<<grid, block>>>(d_data, N, 0, max_depth);
    CUDA_CHECK(cudaEventRecord(stop));

    cudaError_t launch_err = cudaGetLastError();
    if (is_feature_unsupported(launch_err)) {
        printf("⚠️  Skipping recursive dynamic parallelism demo: %s\n", cudaGetErrorString(launch_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (launch_err != cudaSuccess) {
        printf("CUDA error launching recursive kernel: %s\n", cudaGetErrorString(launch_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (is_feature_unsupported(sync_err)) {
        printf("⚠️  Device cannot execute recursive DP workload (%s); skipping demo\n",
               cudaGetErrorString(sync_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (sync_err != cudaSuccess) {
        printf("CUDA error syncing recursive kernel: %s\n", cudaGetErrorString(sync_err));
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    printf("Execution time: %.2f ms\n", elapsed_ms);
    printf("Final values: %.6f, %.6f, %.6f\n", h_data[0], h_data[N/2], h_data[N-1]);
    
    // Show how values change with recursive processing
    printf("Recursive processing creates hierarchical patterns in data\n");
    
    delete[] h_data;
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}

bool demonstrateDeviceGraphLaunch() {
    printf("\n=== Device-Initiated Graph Launch ===\n");
    
    const int N = 1024;
    const int bytes = N * sizeof(float);
    
    float *h_data = new float[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)i / N;
    }
    
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    // Create a graph to be launched from device
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    printf("Creating graph for device-initiated launch...\n");
    
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Add kernels to the graph
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    
    workKernelHigh<<<grid, block, 0, stream>>>(d_data, N);
    workKernelLow<<<grid, block, 0, stream>>>(d_data, N);
    
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    
    // Instantiate for device launch
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    // Upload graph to device memory for device-initiated launch
    CUDA_CHECK(cudaGraphUpload(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy graph executor to device global memory
    CUDA_CHECK(cudaMemcpyToSymbol(g_graphExec, &graphExec, sizeof(cudaGraphExec_t)));
    
    printf("Launching persistent scheduler kernel...\n");
    
    const int numTasks = 100;
    const int maxIterations = 50;
    
    // Reset work counter
    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_workIndex, &zero, sizeof(int)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch persistent scheduler (single block for simplicity)
    persistentScheduler<<<1, 32>>>(d_data, numTasks, maxIterations);
    
    CUDA_CHECK(cudaEventRecord(stop));
    cudaError_t launch_err = cudaGetLastError();
    if (is_feature_unsupported(launch_err)) {
        printf("⚠️  Skipping device-initiated graph launch demo: %s\n", cudaGetErrorString(launch_err));
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (launch_err != cudaSuccess) {
        printf("CUDA error launching persistent scheduler: %s\n", cudaGetErrorString(launch_err));
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (is_feature_unsupported(sync_err)) {
        printf("⚠️  Device cannot execute graph tail launches (%s); skipping demo\n",
               cudaGetErrorString(sync_err));
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (sync_err != cudaSuccess) {
        printf("CUDA error syncing graph launch: %s\n", cudaGetErrorString(sync_err));
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        delete[] h_data;
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    printf("Execution time: %.2f ms\n", elapsed_ms);
    printf("Device-initiated %d iterations with embedded graph launches\n", maxIterations);
    printf("Final data sample: %.6f, %.6f, %.6f\n", h_data[0], h_data[N/2], h_data[N-1]);
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    delete[] h_data;
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}

// Adaptive scheduling kernel that chooses between different algorithms
__global__ void adaptiveScheduler(float* input, float* output, int N, int* algorithm_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float value = input[idx];
        
        // Choose algorithm based on input characteristics
        if (value < 0.3f) {
            // Algorithm 1: Simple scaling
            output[idx] = value * 2.0f;
            atomicAdd(&algorithm_counts[0], 1);
            
        } else if (value < 0.7f) {
            // Algorithm 2: Trigonometric
            output[idx] = sinf(value * 3.14159f);
            atomicAdd(&algorithm_counts[1], 1);
            
            // Launch additional processing if needed
            if ((idx % 100) == 0) {
                dim3 child_grid(1);
                dim3 child_block(32);
                
                // Launch child kernel for specialized processing
                childKernel<<<child_grid, child_block>>>(output, idx, min(50, N-idx), 1.5f);
            }
            
        } else {
            // Algorithm 3: Complex computation
            float result = value;
            for (int i = 0; i < 5; ++i) {
                result = sqrtf(result * result + 0.1f);
            }
            output[idx] = result;
            atomicAdd(&algorithm_counts[2], 1);
        }
    }
}

bool demonstrateAdaptiveScheduling() {
    printf("\n=== Adaptive Device-Side Scheduling ===\n");
    
    const int N = 10000;
    const int bytes = N * sizeof(float);
    
    float *h_input = new float[N];
    float *h_output = new float[N];
    
    // Create data with different characteristics
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    float *d_input, *d_output;
    int *d_algorithm_counts;
    
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_algorithm_counts, 3 * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_algorithm_counts, 0, 3 * sizeof(int)));
    
    printf("Running adaptive scheduler on %d elements...\n", N);
    
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    adaptiveScheduler<<<grid, block>>>(d_input, d_output, N, d_algorithm_counts);
    CUDA_CHECK(cudaEventRecord(stop));
    
    cudaError_t launch_err = cudaGetLastError();
    if (is_feature_unsupported(launch_err)) {
        printf("⚠️  Skipping adaptive scheduling demo: %s\n", cudaGetErrorString(launch_err));
        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_algorithm_counts);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (launch_err != cudaSuccess) {
        printf("CUDA error launching adaptive scheduler: %s\n", cudaGetErrorString(launch_err));
        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_algorithm_counts);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (is_feature_unsupported(sync_err)) {
        printf("⚠️  Device cannot execute adaptive DP workload (%s); skipping demo\n",
               cudaGetErrorString(sync_err));
        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_algorithm_counts);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    if (sync_err != cudaSuccess) {
        printf("CUDA error syncing adaptive scheduler: %s\n", cudaGetErrorString(sync_err));
        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_algorithm_counts);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    // Get results
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    int algorithm_counts[3];
    CUDA_CHECK(cudaMemcpy(algorithm_counts, d_algorithm_counts, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("Execution time: %.2f ms\n", elapsed_ms);
    printf("Algorithm usage:\n");
    printf("  Simple scaling:  %d elements (%.1f%%)\n", 
           algorithm_counts[0], 100.0f * algorithm_counts[0] / N);
    printf("  Trigonometric:   %d elements (%.1f%%)\n", 
           algorithm_counts[1], 100.0f * algorithm_counts[1] / N);
    printf("  Complex compute: %d elements (%.1f%%)\n", 
           algorithm_counts[2], 100.0f * algorithm_counts[2] / N);
    
    printf("Adaptive scheduling enables optimal algorithm selection per element\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_algorithm_counts);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}

int main() {
    printf("Dynamic Parallelism and Device-Initiated Orchestration - Chapter 12\n");
    printf("===================================================================\n");
    
    // Check device capabilities
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // Check for dynamic parallelism support (CC 3.5+)
    if ((prop.major > 3) || (prop.major == 3 && prop.minor >= 5)) {
        printf("Dynamic Parallelism: Supported\n");
    } else {
        printf("Dynamic Parallelism: Not supported (requires CC 3.5+)\n");
        printf("Current CC: %d.%d\n", prop.major, prop.minor);
        return 1;
    }
    
    // Check for graph support
    if (prop.major >= 8 || (prop.major == 7 && prop.minor >= 5)) {
        printf("Device Graph Launch: Supported\n");
    } else {
        printf("Device Graph Launch: Not supported (requires CC 7.5+)\n");
    }
    
    printf("\n");
    
    (void)demonstrateBasicDynamicParallelism();
    (void)demonstrateRecursiveLaunches();
    
    // Only run graph examples on supported devices
    if (prop.major >= 8 || (prop.major == 7 && prop.minor >= 5)) {
        demonstrateDeviceGraphLaunch();
    }
    
    demonstrateAdaptiveScheduling();
    
    printf("\n=== Dynamic Parallelism Summary ===\n");
    printf("Benefits:\n");
    printf("- Device-side decision making without CPU involvement\n");
    printf("- Adaptive scheduling based on data characteristics\n");
    printf("- Recursive algorithms with natural GPU implementation\n");
    printf("- Reduced host-device synchronization overhead\n");
    printf("\nConsiderations:\n");
    printf("- Additional overhead from device-side launches\n");
    printf("- Potential for load imbalance and divergence\n");
    printf("- Debugging complexity with nested kernel calls\n");
    printf("- Resource management across kernel generations\n");
    printf("\nBest practices:\n");
    printf("- Use for workloads with significant data-dependent branching\n");
    printf("- Profile carefully to ensure performance benefits\n");
    printf("- Consider alternatives like conditional graphs first\n");
    printf("- Monitor resource usage and kernel nesting depth\n");
    
    printf("\n=== Profiling Commands ===\n");
    printf("nsys profile --force-overwrite=true -o dynamic_parallelism ./dynamic_parallelism\n");
    printf("ncu --section LaunchStats --section WarpStateStats ./dynamic_parallelism\n");
    
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    // CUDA 13 teasers for Blackwell readers.
    cuda13_teasers::stream_ordered_teaser();
    cuda13_teasers::tma_teaser();
#else
    printf("CUDA 13 teasers require CUDA 13 headers; see later chapters for full demos.\n");
#endif

    return 0;
}

// CUDA 13.0 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    // See ch11/stream_ordered_allocator.cu for a full cudaMallocAsync demo.
    // Your kernel code here
}

// CUDA 13.0 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    // See ch7/async_prefetch_tma.cu or ch10/tma_2d_pipeline_blackwell.cu for real TMA usage.
    // Your TMA code here
}
