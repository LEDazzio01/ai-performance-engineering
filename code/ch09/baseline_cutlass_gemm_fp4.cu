// baseline_cutlass_gemm_fp4.cu -- Naive FP4 GEMM baseline for tensor core comparison
//
// This baseline uses block-scaled FP4 (NVFP4) format to provide an apples-to-apples
// comparison with the optimized FP4 cuBLASLt version that uses tensor cores.
//
// BOOK REFERENCE (Ch9/Ch19): FP4 on Blackwell provides 3-5x throughput over FP16
// due to 4-bit precision with per-block scaling for numerical stability.
// This baseline shows naive FP4 GEMM without tensor core acceleration.
//
// NOTE: Block-scaled FP4 uses groups of 32 or 64 elements sharing a single scale factor.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Block scaling parameters for FP4
constexpr int BLOCK_SIZE_SCALE = 32;  // Elements per scaling block

// FP4 representation: packed 2 values per byte (nibble format)
// Value range: [-6, 6] with 4-bit representation (signed)
struct PackedFP4 {
    uint8_t data;  // Contains 2 x 4-bit values
    
    __host__ __device__ void pack(float v0, float v1, float scale) {
        // Quantize to 4-bit signed integer (-8 to 7)
        int q0 = static_cast<int>(roundf(v0 / scale));
        int q1 = static_cast<int>(roundf(v1 / scale));
        q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
        q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
        // Pack into single byte (low nibble = v0, high nibble = v1)
        data = ((q1 & 0xF) << 4) | (q0 & 0xF);
    }
    
    __host__ __device__ void unpack(float& v0, float& v1, float scale) const {
        // Unpack 4-bit signed integers
        int q0 = (data & 0xF);
        int q1 = ((data >> 4) & 0xF);
        // Sign extend
        if (q0 & 0x8) q0 |= 0xFFFFFFF0;
        if (q1 & 0x8) q1 |= 0xFFFFFFF0;
        v0 = static_cast<float>(q0) * scale;
        v1 = static_cast<float>(q1) * scale;
    }
};

// Naive tiled FP4 GEMM kernel (no tensor cores)
// Uses block-scaled FP4 with FP32 accumulation
template<int TILE_SIZE = 32>
__global__ void tiled_fp4_gemm_kernel(const PackedFP4* __restrict__ A,
                                       const float* __restrict__ A_scales,
                                       const PackedFP4* __restrict__ B,
                                       const float* __restrict__ B_scales,
                                       __half* __restrict__ C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        const int k_idx = t * TILE_SIZE + threadIdx.x;
        const int k_row = t * TILE_SIZE + threadIdx.y;
        
        // Load tile of A into shared memory (unpack FP4 to FP32)
        if (row < M && k_idx < K) {
            // Determine which packed byte and position within byte
            const int a_idx = row * (K / 2) + k_idx / 2;
            const int scale_idx = row * (K / BLOCK_SIZE_SCALE) + k_idx / BLOCK_SIZE_SCALE;
            const float scale_a = A_scales[scale_idx];
            
            float v0, v1;
            A[a_idx].unpack(v0, v1, scale_a);
            As[threadIdx.y][threadIdx.x] = (k_idx % 2 == 0) ? v0 : v1;
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory (unpack FP4 to FP32)
        if (k_row < K && col < N) {
            const int b_idx = k_row * (N / 2) + col / 2;
            const int scale_idx = k_row * (N / BLOCK_SIZE_SCALE) + col / BLOCK_SIZE_SCALE;
            const float scale_b = B_scales[scale_idx];
            
            float v0, v1;
            B[b_idx].unpack(v0, v1, scale_b);
            Bs[threadIdx.y][threadIdx.x] = (col % 2 == 0) ? v0 : v1;
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        float c_val = (beta != 0.0f) ? __half2float(C[row * N + col]) : 0.0f;
        float result = alpha * sum + beta * c_val;
        C[row * N + col] = __float2half(result);
    }
}

// Helper function to quantize FP32 to block-scaled FP4
void quantize_to_fp4(const float* input, PackedFP4* output, float* scales,
                     int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int block_start = 0; block_start < cols; block_start += BLOCK_SIZE_SCALE) {
            // Find max absolute value in block
            float max_abs = 0.0f;
            for (int i = 0; i < BLOCK_SIZE_SCALE && block_start + i < cols; ++i) {
                max_abs = std::max(max_abs, std::abs(input[r * cols + block_start + i]));
            }
            
            // Scale factor to map max_abs to 7 (max positive FP4 value)
            float scale = (max_abs > 0.0f) ? max_abs / 7.0f : 1.0f;
            scales[r * (cols / BLOCK_SIZE_SCALE) + block_start / BLOCK_SIZE_SCALE] = scale;
            
            // Quantize pairs of values
            for (int i = 0; i < BLOCK_SIZE_SCALE; i += 2) {
                if (block_start + i + 1 < cols) {
                    float v0 = input[r * cols + block_start + i];
                    float v1 = input[r * cols + block_start + i + 1];
                    output[r * (cols / 2) + (block_start + i) / 2].pack(v0, v1, scale);
                }
            }
        }
    }
}

int main() {
    // Matrix dimensions - must be divisible by BLOCK_SIZE_SCALE
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    constexpr int kIterations = 10;
    constexpr int kBatchCount = 8;

    // Sizes for packed FP4 (2 values per byte)
    const size_t packed_A_size = (M * K / 2) * kBatchCount;
    const size_t packed_B_size = (K * N / 2) * kBatchCount;
    const size_t elements_C = static_cast<size_t>(M) * N;
    
    // Scale factors per block
    const size_t scales_A_size = (M * K / BLOCK_SIZE_SCALE) * kBatchCount;
    const size_t scales_B_size = (K * N / BLOCK_SIZE_SCALE) * kBatchCount;

    // Host allocation
    float* h_A_fp32 = new float[M * K * kBatchCount];
    float* h_B_fp32 = new float[K * N * kBatchCount];
    PackedFP4* h_A = new PackedFP4[packed_A_size];
    PackedFP4* h_B = new PackedFP4[packed_B_size];
    float* h_A_scales = new float[scales_A_size];
    float* h_B_scales = new float[scales_B_size];
    __half* h_C = new __half[elements_C * kBatchCount];

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < M * K * kBatchCount; ++i) {
        h_A_fp32[i] = dis(gen);
    }
    for (size_t i = 0; i < K * N * kBatchCount; ++i) {
        h_B_fp32[i] = dis(gen);
    }
    
    // Quantize to block-scaled FP4
    for (int batch = 0; batch < kBatchCount; ++batch) {
        quantize_to_fp4(h_A_fp32 + batch * M * K, 
                        h_A + batch * (M * K / 2),
                        h_A_scales + batch * (M * K / BLOCK_SIZE_SCALE),
                        M, K);
        quantize_to_fp4(h_B_fp32 + batch * K * N,
                        h_B + batch * (K * N / 2),
                        h_B_scales + batch * (K * N / BLOCK_SIZE_SCALE),
                        K, N);
    }
    
    for (size_t i = 0; i < elements_C * kBatchCount; ++i) {
        h_C[i] = __float2half(0.0f);
    }

    // Device allocation
    PackedFP4 *d_A = nullptr, *d_B = nullptr;
    float *d_A_scales = nullptr, *d_B_scales = nullptr;
    __half *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, packed_A_size * sizeof(PackedFP4)));
    CUDA_CHECK(cudaMalloc(&d_B, packed_B_size * sizeof(PackedFP4)));
    CUDA_CHECK(cudaMalloc(&d_A_scales, scales_A_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_scales, scales_B_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * kBatchCount * sizeof(__half)));

    // Pre-load all data before timing
    CUDA_CHECK(cudaMemcpy(d_A, h_A, packed_A_size * sizeof(PackedFP4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, packed_B_size * sizeof(PackedFP4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales, scales_A_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales, scales_B_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, elements_C * kBatchCount * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    constexpr int TILE_SIZE = 32;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup
    for (int batch = 0; batch < kBatchCount; ++batch) {
        const size_t offset_A = batch * (M * K / 2);
        const size_t offset_B = batch * (K * N / 2);
        const size_t offset_C = batch * elements_C;
        const size_t offset_A_scales = batch * (M * K / BLOCK_SIZE_SCALE);
        const size_t offset_B_scales = batch * (K * N / BLOCK_SIZE_SCALE);
        
        tiled_fp4_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_A + offset_A, d_A_scales + offset_A_scales,
            d_B + offset_B, d_B_scales + offset_B_scales,
            d_C + offset_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed section: Kernel execution only
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * (M * K / 2);
            const size_t offset_B = batch * (K * N / 2);
            const size_t offset_C = batch * elements_C;
            const size_t offset_A_scales = batch * (M * K / BLOCK_SIZE_SCALE);
            const size_t offset_B_scales = batch * (K * N / BLOCK_SIZE_SCALE);
            
            tiled_fp4_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
                d_A + offset_A, d_A_scales + offset_A_scales,
                d_B + offset_B, d_B_scales + offset_B_scales,
                d_C + offset_C, M, N, K, alpha, beta);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kBatchCount);
    
    // Calculate TFLOPS (effective computation)
    const double flops = 2.0 * M * N * K * kBatchCount * kIterations;
    const double tflops = flops / (total_ms * 1e9);
    
    std::cout << "Naive Tiled FP4 GEMM (baseline): " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_scales;
    delete[] h_B_scales;
    delete[] h_C;

    return 0;
}

