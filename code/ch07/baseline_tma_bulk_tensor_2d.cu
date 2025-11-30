// baseline_tma_bulk_tensor_2d.cu
// Manual 2D global->shared->global copy (no TMA) for comparison with the TMA bulk tensor path.
//
// Targets Blackwell/Grace-Blackwell (sm_100+/sm_103+/sm_121) but runs on any SM_90+
// device because it uses only standard CUDA loads/stores. The optimized variant swaps
// in cp.async.bulk.tensor for the transfers.

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

// Keep the baseline conservative: smaller tiles increase block count and expose
// the overheads that TMA helps hide.
constexpr int TILE_M = 64;
// Use a narrower tile to keep static shared under 48 KB on SM100.
constexpr int TILE_N = 32;
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 2;   // Two warps to highlight overhead vs. TMA
constexpr int ITERATIONS = 10;

__global__ void baseline_bulk_copy_kernel(const float* __restrict__ src,
                                          float* __restrict__ dst,
                                          int width,
                                          int height,
                                          int ld) {
    const int tile_row = blockIdx.y * TILE_M;
    const int tile_col = blockIdx.x * TILE_N;
    if (tile_row >= height || tile_col >= width) {
        return;
    }

    __shared__ alignas(128) float tile[TILE_M][TILE_N];

    for (int r = threadIdx.y; r < TILE_M; r += blockDim.y) {
        const int g_row = tile_row + r;
        if (g_row >= height) break;
        const float* src_row = src + g_row * ld;
        for (int c = threadIdx.x; c < TILE_N; c += blockDim.x) {
            const int g_col = tile_col + c;
            if (g_col < width) {
                tile[r][c] = src_row[g_col];
            }
        }
    }
    __syncthreads();

    // Trivial transformation to keep the compiler from optimizing the copy away.
    for (int r = threadIdx.y; r < TILE_M; r += blockDim.y) {
        const int g_row = tile_row + r;
        if (g_row >= height) break;
        float* dst_row = dst + g_row * ld;
        for (int c = threadIdx.x; c < TILE_N; c += blockDim.x) {
            const int g_col = tile_col + c;
            if (g_col < width) {
                const float v = tile[r][c];
                dst_row[g_col] = v * 1.0001f + 0.0001f;
            }
        }
    }
}

}  // namespace

int main() {
    const int width = 2048;
    const int height = 2048;
    const int ld = width;
    const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);

    std::vector<float> h_src(width * height);
    for (int i = 0; i < width * height; ++i) {
        h_src[i] = static_cast<float>((i % 113) - 57) * 0.01f;
    }

    float* d_src = nullptr;
    float* d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, bytes));

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((width + TILE_N - 1) / TILE_N, (height + TILE_M - 1) / TILE_M, 1);

    // Warmup
    baseline_bulk_copy_kernel<<<grid, block>>>(d_src, d_dst, width, height, ld);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        baseline_bulk_copy_kernel<<<grid, block>>>(d_src, d_dst, width, height, ld);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(ITERATIONS);
    std::printf("Baseline 2D tensor copy: %.3f ms\n", avg_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}
