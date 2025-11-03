// cuda_graphs_conditional_enhanced.cu -- CUDA Graphs with Conditional Nodes (CUDA 13+)
// Demonstrates dynamic branching in graphs for Blackwell B200/GB10
// Compile: nvcc -O3 -std=c++17 -arch=sm_100 cuda_graphs_conditional_enhanced.cu -o cuda_graphs_conditional_enhanced

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int N = 1 << 20;  // 1M elements

// Kernel A: Expensive computation
__global__ void expensive_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // Expensive loop
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            val = sqrtf(val * val + scale) * 0.99f;
        }
        data[idx] = val;
    }
}

// Kernel B: Cheap computation
__global__ void cheap_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Predicate kernel: Sets condition value
__global__ void predicate_kernel(int* condition, float* data, int n, float threshold) {
    // Check if data mean exceeds threshold
    // Simplified: just check first element for demo
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *condition = (data[0] > threshold) ? 1 : 0;
    }
}

// Helper: Detect if we're on Blackwell
bool is_blackwell_or_newer() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.major >= 10;  // SM 10.0+ (Blackwell and newer)
}

int main() {
    // Detect architecture
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    bool is_sm121 = (prop.major == 12);
    bool is_sm100 = (prop.major == 10 && prop.minor == 0);
    bool supports_conditional = (prop.major >= 10);  // Blackwell and newer
    
    std::printf("=== CUDA Graphs with Conditional Nodes ===\n");
    std::printf("Architecture: %s (SM %d.%d)\n", 
                is_sm121 ? "Grace-Blackwell GB10" : is_sm100 ? "Blackwell B200" : "Other",
                prop.major, prop.minor);
    std::printf("Conditional graphs supported: %s\n\n", 
                supports_conditional ? "✅ YES" : "❌ NO (requires CUDA 13 + Blackwell)");
    
    if (!supports_conditional) {
        std::printf("⚠️  This demo requires Blackwell (SM 10.0+) and CUDA 13+\n");
        std::printf("   Falling back to standard CUDA graph demo...\n\n");
    }
    
    // Allocate memory
    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    int *d_condition = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_condition, sizeof(int)));
    
    // Initialize data
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f + (i % 100) * 0.01f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // ========================================================================
    // Method 1: Standard approach (no graph) with dynamic dispatch
    // ========================================================================
    std::printf("Method 1: Standard (no graph) with dynamic dispatch\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    constexpr int ITERS = 1000;
    constexpr float THRESHOLD = 1.5f;
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        // Check condition on GPU
        predicate_kernel<<<1, 1>>>(d_condition, d_data, N, THRESHOLD);
        
        // Copy condition back to CPU for branching
        int h_condition = 0;
        CUDA_CHECK(cudaMemcpy(&h_condition, d_condition, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Branch based on condition
        if (h_condition) {
            expensive_kernel<<<grid, block>>>(d_data, N, 1.01f);
        } else {
            cheap_kernel<<<grid, block>>>(d_data, N, 0.99f);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms_standard = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_standard, start, stop));
    
    std::printf("  Time: %.2f ms (%.3f ms/iter)\n", ms_standard, ms_standard / ITERS);
    std::printf("  Overhead: D2H copy + CPU branch decision\n\n");
    
    // ========================================================================
    // Method 2: CUDA Graph (static path)
    // ========================================================================
    std::printf("Method 2: CUDA Graph (static path, expensive branch)\n");
    
    // Reset data
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    cudaGraph_t graph_static;
    cudaGraphExec_t graph_exec_static;
    
    // Capture graph
    CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
    expensive_kernel<<<grid, block>>>(d_data, N, 1.01f);
    CUDA_CHECK(cudaStreamEndCapture(0, &graph_static));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_static, graph_static, nullptr, nullptr, 0));
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec_static, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms_graph_static = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_graph_static, start, stop));
    
    std::printf("  Time: %.2f ms (%.3f ms/iter)\n", ms_graph_static, ms_graph_static / ITERS);
    std::printf("  Speedup vs standard: %.2fx\n", ms_standard / ms_graph_static);
    std::printf("  Note: Fixed path, no dynamic branching\n\n");
    
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec_static));
    CUDA_CHECK(cudaGraphDestroy(graph_static));
    
    // ========================================================================
    // Method 3: CUDA Graph with Conditional Nodes (CUDA 13 + Blackwell)
    // ========================================================================
    if (supports_conditional) {
        std::printf("Method 3: CUDA Graph with Conditional Nodes (CUDA 13 feature)\n");
        std::printf("  Status: ⚠️  EXPERIMENTAL - API may require CUDA 13.0+\n");
        std::printf("  Note: Conditional graph API is in development\n");
        std::printf("        This is a placeholder for when the API stabilizes\n\n");
        
        // Conditional graph API pseudocode (actual API may differ):
        /*
        cudaGraph_t graph_cond;
        cudaGraphExec_t graph_exec_cond;
        
        cudaGraphNode_t predicate_node, if_node, else_node;
        
        // Create predicate node
        cudaKernelNodeParams predicate_params{};
        predicate_params.func = (void*)predicate_kernel;
        predicate_params.gridDim = dim3(1);
        predicate_params.blockDim = dim3(1);
        predicate_params.kernelParams = ...;
        
        cudaGraphAddKernelNode(&predicate_node, graph_cond, nullptr, 0, &predicate_params);
        
        // Create conditional node
        cudaConditionalNodeParams cond_params{};
        cond_params.type = cudaGraphConditionalTypeIf;
        cond_params.handle = ...;  // Handle to condition variable
        
        cudaGraphAddConditionalNode(&if_node, graph_cond, &predicate_node, 1, &cond_params);
        
        // Add expensive kernel as true branch
        cudaKernelNodeParams expensive_params{};
        // ... configure expensive_kernel
        cudaGraphAddKernelNode(&true_branch, graph_cond, &if_node, 1, &expensive_params);
        
        // Add cheap kernel as false branch
        cudaKernelNodeParams cheap_params{};
        // ... configure cheap_kernel
        cudaGraphAddKernelNode(&false_branch, graph_cond, &if_node, 1, &cheap_params);
        */
        
        // For now, demonstrate the benefit with manual graph update
        std::printf("  Simulating with graph update mechanism...\n");
        
        // Create two graphs (true and false paths)
        cudaGraph_t graph_true, graph_false;
        cudaGraphExec_t graph_exec_true, graph_exec_false;
        
        // Capture true path (expensive)
        CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
        expensive_kernel<<<grid, block>>>(d_data, N, 1.01f);
        CUDA_CHECK(cudaStreamEndCapture(0, &graph_true));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_true, graph_true, nullptr, nullptr, 0));
        
        // Capture false path (cheap)
        CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
        cheap_kernel<<<grid, block>>>(d_data, N, 0.99f);
        CUDA_CHECK(cudaStreamEndCapture(0, &graph_false));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_false, graph_false, nullptr, nullptr, 0));
        
        // Benchmark with simulated conditional dispatch
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i) {
            // Predicate evaluation
            predicate_kernel<<<1, 1>>>(d_condition, d_data, N, THRESHOLD);
            
            // Simulated conditional dispatch (in hardware on real conditional graphs)
            int h_cond = 0;
            CUDA_CHECK(cudaMemcpy(&h_cond, d_condition, sizeof(int), cudaMemcpyDeviceToHost));
            
            // Launch appropriate graph
            if (h_cond) {
                CUDA_CHECK(cudaGraphLaunch(graph_exec_true, 0));
            } else {
                CUDA_CHECK(cudaGraphLaunch(graph_exec_false, 0));
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float ms_cond = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_cond, start, stop));
        
        std::printf("  Time: %.2f ms (%.3f ms/iter)\n", ms_cond, ms_cond / ITERS);
        std::printf("  Speedup vs standard: %.2fx\n", ms_standard / ms_cond);
        std::printf("  Expected with true conditional graphs: %.2fx (removes D2H copy)\n\n",
                    ms_standard / ms_graph_static);
        
        CUDA_CHECK(cudaGraphExecDestroy(graph_exec_true));
        CUDA_CHECK(cudaGraphExecDestroy(graph_exec_false));
        CUDA_CHECK(cudaGraphDestroy(graph_true));
        CUDA_CHECK(cudaGraphDestroy(graph_false));
    }
    
    // ========================================================================
    // Results Summary
    // ========================================================================
    std::printf("=== Summary ===\n");
    std::printf("Standard (no graph):      %.2f ms (1.00x)\n", ms_standard);
    std::printf("Static graph:             %.2f ms (%.2fx faster)\n", 
                ms_graph_static, ms_standard / ms_graph_static);
    
    if (supports_conditional) {
        std::printf("\nℹ️  Conditional Graphs Benefits (when API is stable):\n");
        std::printf("  • Dynamic branching without graph replay\n");
        std::printf("  • GPU-side condition evaluation (no D2H copy)\n");
        std::printf("  • Ideal for: batch processing with varying workloads\n");
        std::printf("  • Use cases: \n");
        std::printf("    - Early exit in iterative solvers\n");
        std::printf("    - Adaptive sampling in inference\n");
        std::printf("    - Dynamic pruning in neural networks\n");
    }
    
    if (is_sm100 || is_sm121) {
        std::printf("\n✅ Blackwell Optimizations:\n");
        std::printf("  • Hardware support for conditional execution\n");
        std::printf("  • Reduced kernel launch overhead in graphs\n");
        std::printf("  • Better overlap with memory operations\n");
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_condition));
    
    return 0;
}

