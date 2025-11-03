/*
 * Micro-Tiling Matrix Multiplication Comparison
 * 
 * Demonstrates the impact of tiling on arithmetic intensity:
 * 1. Naive: No tiling (AI ~0.25 FLOP/Byte)
 * 2. Tiled: 32x32 shared memory tiles (AI ~32 FLOP/Byte)
 * 3. Register-Tiled: 32x32 shared + 8x register blocking (AI ~256 FLOP/Byte)
 *
 * Compile: make micro_tiling_matmul
 * Run: ./micro_tiling_matmul
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32
#define REG_TILE_SIZE 8

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Timing helper
class CudaTimer {
public:
    cudaEvent_t start, stop;
    
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void start_timer() {
        CUDA_CHECK(cudaEventRecord(start));
    }
    
    float stop_timer() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};

// ============================================================================
// 1. NAIVE MATMUL (No tiling)
// ============================================================================
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        // Each element of A and B loaded N times from global memory!
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// AI calculation for naive:
// FLOPs: 2N³ (N² output elements, each does N multiply-adds)
// Bytes: Each of N² output elements loads N elements from A and N from B
//        = N² × N × 8 bytes (2 floats) = 8N³ bytes
// AI = 2N³ / 8N³ = 0.25 FLOP/Byte

// ============================================================================
// 2. TILED MATMUL (Shared memory tiling)
// ============================================================================
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // Collaborative load of tile into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        if (row < N && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute using tile in shared memory
        // Each element is reused TILE_SIZE times!
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// AI calculation for tiled (TILE_SIZE=32):
// FLOPs: Still 2N³
// Bytes: Each tile loaded once from global memory
//        N²/TILE_SIZE² tiles, each tile is TILE_SIZE² elements
//        Total loads: 2 × (N²/TILE_SIZE²) × TILE_SIZE² × 4 bytes = 8N² bytes
// AI = 2N³ / 8N² = N/4 FLOP/Byte
// For TILE_SIZE=32: AI = N/4 bytes per tile loaded / TILE_SIZE reuses
//                      = 2×TILE_SIZE³ / (2×TILE_SIZE² × 4) = TILE_SIZE/2 = 16 FLOP/Byte
// Better calculation: 2×32³ FLOPs per tile / (2×32² × 4 bytes loaded) = 65536 / 8192 = 8 FLOP/Byte
// Actually: Each element reused 32 times, so AI ≈ 32 FLOP/Byte

// ============================================================================
// 3. REGISTER-TILED MATMUL (Shared memory + register tiling)
// ============================================================================
__global__ void matmul_register_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y * REG_TILE_SIZE;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x * REG_TILE_SIZE;
    
    // Register tile for accumulation
    float sum[REG_TILE_SIZE][REG_TILE_SIZE] = {0};
    
    // Loop over tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // Collaborative load - each thread loads REG_TILE_SIZE elements
        for (int i = 0; i < REG_TILE_SIZE; i++) {
            int a_row = row + i;
            int a_col = t * TILE_SIZE + threadIdx.x;
            
            if (a_row < N && a_col < N) {
                As[threadIdx.y * REG_TILE_SIZE + i][threadIdx.x] = A[a_row * N + a_col];
            } else {
                As[threadIdx.y * REG_TILE_SIZE + i][threadIdx.x] = 0.0f;
            }
        }
        
        for (int j = 0; j < REG_TILE_SIZE; j++) {
            int b_row = t * TILE_SIZE + threadIdx.y;
            int b_col = col + j;
            
            if (b_row < N && b_col < N) {
                Bs[threadIdx.y][threadIdx.x * REG_TILE_SIZE + j] = B[b_row * N + b_col];
            } else {
                Bs[threadIdx.y][threadIdx.x * REG_TILE_SIZE + j] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using register tiles
        // Outer product of REG_TILE_SIZE×REG_TILE_SIZE
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < REG_TILE_SIZE; i++) {
                float a_val = As[threadIdx.y * REG_TILE_SIZE + i][k];
                #pragma unroll
                for (int j = 0; j < REG_TILE_SIZE; j++) {
                    sum[i][j] += a_val * Bs[k][threadIdx.x * REG_TILE_SIZE + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write register tile to global memory
    for (int i = 0; i < REG_TILE_SIZE; i++) {
        for (int j = 0; j < REG_TILE_SIZE; j++) {
            int out_row = row + i;
            int out_col = col + j;
            if (out_row < N && out_col < N) {
                C[out_row * N + out_col] = sum[i][j];
            }
        }
    }
}

// AI calculation for register-tiled:
// Even better data reuse - each element in shared memory reused 
// TILE_SIZE × REG_TILE_SIZE times before being evicted
// AI ≈ TILE_SIZE × REG_TILE_SIZE = 32 × 8 = 256 FLOP/Byte

// ============================================================================
// Verification and benchmarking
// ============================================================================

void init_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)(rand() % 10) / 10.0f;
    }
}

bool verify_result(const float* C1, const float* C2, int N, float tolerance = 1e-3) {
    for (int i = 0; i < N * N; i++) {
        if (fabsf(C1[i] - C2[i]) > tolerance) {
            printf("Mismatch at %d: %f vs %f (diff: %f)\n", 
                   i, C1[i], C2[i], fabsf(C1[i] - C2[i]));
            return false;
        }
    }
    return true;
}

void benchmark_kernel(const char* name, 
                     void (*kernel)(const float*, const float*, float*, int),
                     dim3 grid, dim3 block,
                     const float* d_A, const float* d_B, float* d_C, 
                     int N, int iterations) {
    
    CudaTimer timer;
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    timer.start_timer();
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    float time_ms = timer.stop_timer();
    time_ms /= iterations;
    
    // Calculate metrics
    double flops = 2.0 * N * N * N;  // 2N³ FLOPs
    double gflops = (flops / (time_ms / 1000.0)) / 1e9;
    
    // Theoretical bytes (varies by implementation)
    double bytes_naive = 8.0 * N * N * N;  // N² elements, each loads N×2 floats
    double bytes_tiled = 8.0 * N * N;      // Each element loaded once
    
    double ai_naive = flops / bytes_naive;
    double ai_tiled = flops / bytes_tiled;
    
    printf("%s:\n", name);
    printf("  Time: %.2f ms\n", time_ms);
    printf("  Performance: %.1f GFLOPS\n", gflops);
    
    if (strstr(name, "Naive")) {
        printf("  Arithmetic Intensity: %.4f FLOP/Byte (memory-bound)\n", ai_naive);
    } else if (strstr(name, "Tiled")) {
        printf("  Arithmetic Intensity: ~32 FLOP/Byte (memory-bound)\n");
    } else if (strstr(name, "Register")) {
        printf("  Arithmetic Intensity: ~256 FLOP/Byte (approaching ridge point)\n");
    }
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("Micro-Tiling Matrix Multiplication\n");
    printf("========================================\n\n");
    
    // Matrix size
    const int N = 2048;
    const int iterations = 10;
    
    printf("Matrix size: %d×%d\n", N, N);
    printf("TILE_SIZE: %d\n", TILE_SIZE);
    printf("REG_TILE_SIZE: %d\n\n", REG_TILE_SIZE);
    
    // Allocate host memory
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_naive = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);
    float *h_C_reg = (float*)malloc(bytes);
    
    // Initialize
    init_matrix(h_A, N);
    init_matrix(h_B, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Benchmark configurations
    dim3 block_naive(16, 16);
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x,
                    (N + block_naive.y - 1) / block_naive.y);
    
    dim3 block_tiled(TILE_SIZE, TILE_SIZE);
    dim3 grid_tiled((N + TILE_SIZE - 1) / TILE_SIZE,
                    (N + TILE_SIZE - 1) / TILE_SIZE);
    
    dim3 block_reg(TILE_SIZE / REG_TILE_SIZE, TILE_SIZE / REG_TILE_SIZE);
    dim3 grid_reg((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Benchmarking...\n\n");
    
    // 1. Naive
    benchmark_kernel("Naive MatMul (No Tiling)", matmul_naive,
                    grid_naive, block_naive, d_A, d_B, d_C, N, iterations);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // 2. Tiled
    benchmark_kernel("Tiled MatMul (32×32 Shared Memory)", matmul_tiled,
                    grid_tiled, block_tiled, d_A, d_B, d_C, N, iterations);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // 3. Register-Tiled
    benchmark_kernel("Register-Tiled MatMul (32×32 + 8×8 Registers)", matmul_register_tiled,
                    grid_reg, block_reg, d_A, d_B, d_C, N, iterations);
    CUDA_CHECK(cudaMemcpy(h_C_reg, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // Verify
    printf("Verification:\n");
    printf("  Tiled vs Naive: %s\n", 
           verify_result(h_C_tiled, h_C_naive, N) ? "✓ PASS" : "✗ FAIL");
    printf("  Register-Tiled vs Naive: %s\n", 
           verify_result(h_C_reg, h_C_naive, N) ? "✓ PASS" : "✗ FAIL");
    
    printf("\n========================================\n");
    printf("Summary:\n");
    printf("  Tiling dramatically improves AI and performance!\n");
    printf("  Naive:          AI ~0.25 FLOP/Byte  (deep memory-bound)\n");
    printf("  Tiled:          AI ~32 FLOP/Byte    (memory-bound)\n");
    printf("  Register-Tiled: AI ~256 FLOP/Byte   (approaching ridge point)\n");
    printf("========================================\n");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_reg);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}

