// baseline_cutlass_gemm.cu -- Host-staged GEMM baseline.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)
#define CUDA_CHECK_LAST_ERROR()                                                  \
  do {                                                                           \
    cudaError_t status = cudaGetLastError();                                     \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Simple GEMM kernel with no tiling (intentionally bandwidth bound).
__global__ void simple_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    const volatile float* A_volatile = reinterpret_cast<const volatile float*>(A);
    const volatile float* B_volatile = reinterpret_cast<const volatile float*>(B);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A_volatile[row * K + k];
        float b = B_volatile[k * N + col];
        sum = fmaf(a, b, sum);
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int kIterations = 5;
    constexpr int kMicroBatches = 32;

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t size_A = elements_A * sizeof(float);
    const size_t size_B = elements_B * sizeof(float);
    const size_t size_C = elements_C * sizeof(float);

    std::vector<float*> host_batches_A(kMicroBatches);
    std::vector<float*> host_batches_B(kMicroBatches);
    for (int batch = 0; batch < kMicroBatches; ++batch) {
        CUDA_CHECK(cudaMallocHost(&host_batches_A[batch], size_A));
        CUDA_CHECK(cudaMallocHost(&host_batches_B[batch], size_B));
    }
    float* host_result = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_result, size_C));

    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int batch = 0; batch < kMicroBatches; ++batch) {
        for (size_t i = 0; i < elements_A; ++i) {
            host_batches_A[batch][i] = dis(gen);
        }
        for (size_t i = 0; i < elements_B; ++i) {
            host_batches_B[batch][i] = dis(gen);
        }
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    dim3 block_size(16, 8);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);

    // Warmup one staged batch.
    CUDA_CHECK(cudaMemcpyAsync(d_A, host_batches_A[0], size_A, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, host_batches_B[0], size_B, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, size_C, stream));
    simple_gemm_kernel<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaMemcpyAsync(host_result, d_C, size_C, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int micro = 0; micro < kMicroBatches; ++micro) {
            const int batch_idx = (iter + micro) % kMicroBatches;
            CUDA_CHECK(cudaMemcpyAsync(d_A, host_batches_A[batch_idx], size_A, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_B,
                host_batches_B[(batch_idx * 5 + micro) % kMicroBatches],
                size_B,
                cudaMemcpyHostToDevice,
                stream));
            CUDA_CHECK(cudaMemsetAsync(d_C, 0, size_C, stream));
            simple_gemm_kernel<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaMemcpyAsync(host_result, d_C, size_C, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_total, start, stop));
    const float time_avg = time_total / static_cast<float>(kIterations * kMicroBatches);
    std::cout << "Host-staged GEMM (baseline): " << time_avg << " ms" << std::endl;
    std::cout << "Checksum sample: " << host_result[0] << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    for (int batch = 0; batch < kMicroBatches; ++batch) {
        CUDA_CHECK(cudaFreeHost(host_batches_A[batch]));
        CUDA_CHECK(cudaFreeHost(host_batches_B[batch]));
    }
    CUDA_CHECK(cudaFreeHost(host_result));

    return 0;
}
