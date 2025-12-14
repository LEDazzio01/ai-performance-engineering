// optimized_pipeline_3stage.cu - Multi-Stream Parallel Execution (Ch10)
//
// WHAT: Use multiple CUDA streams to overlap independent work.
// Different segments can execute in parallel on the GPU.
//
// WHY THIS IS FASTER:
//   - Independent segments execute concurrently
//   - Better GPU utilization
//   - Multiple kernel waves in flight
//
// COMPARE WITH: baseline_pipeline_3stage.cu
//   - Baseline executes sequentially on single stream
//   - No parallelism between segments

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"

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
constexpr int NUM_STREAMS = 4;  // Multiple streams for parallelism

//============================================================================
// Same compute kernel as baseline
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
        // Same compute work as baseline
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
    
    printf("Optimized Multi-Stream Execution\n");
    printf("================================\n");
    printf("Device: %s\n\n", prop.name);
    
    // Problem size - same as baseline
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int NUM_SEGMENTS = 8;
    const int SEGMENT_SIZE = N / NUM_SEGMENTS;
    
    printf("Elements: %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Segments: %d x %d elements\n", NUM_SEGMENTS, SEGMENT_SIZE);
    printf("Streams: %d (parallel execution)\n", NUM_STREAMS);
    printf("Approach: Round-robin segment dispatch to streams\n\n");
    
    // Allocate
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    // Initialize
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 0.5f;
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
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
            int stream_idx = seg % NUM_STREAMS;
            compute_segment<<<grid, block, 0, streams[stream_idx]>>>(
                d_input, d_output, offset, SEGMENT_SIZE);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark - parallel execution on multiple streams
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (int seg = 0; seg < NUM_SEGMENTS; ++seg) {
            int offset = seg * SEGMENT_SIZE;
            int stream_idx = seg % NUM_STREAMS;
            compute_segment<<<grid, block, 0, streams[stream_idx]>>>(
                d_input, d_output, offset, SEGMENT_SIZE);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    printf("Results:\n");
    printf("  Time: %.3f ms\n", avg_ms);
    printf("\nNote: Independent segments execute in parallel across streams.\n");
    printf("Better GPU utilization = faster execution.\n");

#ifdef VERIFY
    std::vector<float> h_output(N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_output) {
        checksum += static_cast<double>(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
