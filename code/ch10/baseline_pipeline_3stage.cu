// baseline_pipeline_3stage.cu - Sequential Kernel Launches (Ch10)
//
// WHAT: Launch kernels one after another on default stream.
// Each kernel waits for the previous to complete.
//
// WHY THIS IS SLOWER:
//   - No overlap between independent work
//   - GPU idle between kernel launches
//   - Serial execution wastes parallelism
//
// COMPARE WITH: optimized_pipeline_3stage.cu
//   - Uses multiple streams to overlap independent work
//   - Different streams can execute in parallel

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;

//============================================================================
// Simple compute kernel - processes a segment of data
//============================================================================

__global__ void compute_segment(
    const float* __restrict__ input,
    float* __restrict__ output,
    int offset,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = input[offset + idx];
        // Moderate compute work
        #pragma unroll 4
        for (int i = 0; i < 20; ++i) {
            val = sinf(val) * cosf(val) + tanhf(val * 0.5f);
        }
        output[offset + idx] = val;
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Baseline Sequential Kernel Launches\n");
    printf("===================================\n");
    printf("Device: %s\n\n", prop.name);
    
    // Problem size - split into segments for stream overlap
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int NUM_SEGMENTS = 8;
    const int SEGMENT_SIZE = N / NUM_SEGMENTS;
    
    printf("Elements: %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Segments: %d x %d elements\n", NUM_SEGMENTS, SEGMENT_SIZE);
    printf("Approach: Sequential kernels on default stream\n\n");
    
    // Allocate
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    // Initialize
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 0.5f;
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((SEGMENT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 3;
    const int iterations = 20;
    
    // Warmup
    for (int iter = 0; iter < warmup; ++iter) {
        for (int seg = 0; seg < NUM_SEGMENTS; ++seg) {
            int offset = seg * SEGMENT_SIZE;
            compute_segment<<<grid, block>>>(d_input, d_output, offset, SEGMENT_SIZE);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark - sequential execution on default stream
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (int seg = 0; seg < NUM_SEGMENTS; ++seg) {
            int offset = seg * SEGMENT_SIZE;
            compute_segment<<<grid, block>>>(d_input, d_output, offset, SEGMENT_SIZE);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    printf("Results:\n");
    printf("  Time: %.3f ms\n", avg_ms);
    printf("\nNote: Kernels execute sequentially, no overlap.\n");
    printf("Compare with optimized_pipeline_3stage for stream parallelism.\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
