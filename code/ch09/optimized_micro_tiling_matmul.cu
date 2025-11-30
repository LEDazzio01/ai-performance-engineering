// optimized_micro_tiling_matmul.cu -- Register-tiled matmul (optimized).

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32
#define REG_TILE_SIZE 8

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// REGISTER-TILED MATMUL (Shared memory + register tiling)
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

int main() {
    const int N = 2048;
    const size_t bytes = N * N * sizeof(float);
    
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
        h_B[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    dim3 block_reg(TILE_SIZE / REG_TILE_SIZE, TILE_SIZE / REG_TILE_SIZE);
    dim3 grid_reg((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        matmul_register_tiled<<<grid_reg, block_reg>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    printf("Register-tiled matmul (optimized): %.2f ms\n", avg_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}

