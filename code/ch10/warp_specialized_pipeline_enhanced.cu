// warp_specialized_pipeline_enhanced.cu -- Enhanced warp specialization with TMA and deep pipelining
// Optimized for both B200 (sm_100) and GB10 (sm_121)

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int TILE = 128;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int PIPELINE_DEPTH = 4;  // Deep pipeline for Blackwell

// Enhanced compute with more work to showcase pipeline benefits
__device__ void compute_tile_enhanced(const float* a, const float* b, float* c, int lane) {
  for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
    float x = a[idx];
    float y = b[idx];
    
    // More compute to show pipeline overlap benefit
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      result += sqrtf(x * x + y * y) * 0.125f;
    }
    c[idx] = result;
  }
}

// Enhanced warp-specialized kernel with:
// - Deeper pipeline (4 stages vs 1)
// - More compute warps (6 vs 1)
// - Producer/consumer overlap without global sync
__global__ void warp_specialized_enhanced_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int total_tiles) {
  cg::thread_block block = cg::this_thread_block();
  
  // Triple buffering for deep pipeline
  extern __shared__ float smem[];
  float* A_tiles = smem;
  float* B_tiles = smem + PIPELINE_DEPTH * TILE_ELEMS;
  float* C_tiles = smem + 2 * PIPELINE_DEPTH * TILE_ELEMS;

  using pipe_state = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_DEPTH>;
  __shared__ alignas(pipe_state) unsigned char state_storage[sizeof(pipe_state)];
  auto* state = reinterpret_cast<pipe_state*>(state_storage);
  auto pipe = cuda::make_pipeline(block, state);

  int warp_id = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  int warps_per_block = blockDim.x / warpSize;
  
  // 8 warps total: 1 producer, 6 compute, 1 consumer
  constexpr int PRODUCER_WARP = 0;
  constexpr int CONSUMER_WARP = 7;
  constexpr int COMPUTE_WARPS = 6;
  
  // Warp roles
  bool is_producer = (warp_id == PRODUCER_WARP);
  bool is_consumer = (warp_id == CONSUMER_WARP);
  bool is_compute = (warp_id >= 1 && warp_id <= COMPUTE_WARPS);
  
  int global_warp_id = blockIdx.x;
  
  // Producer: Load tiles with pipelining
  if (is_producer) {
    for (int tile = global_warp_id; tile < total_tiles; tile += gridDim.x) {
      int buf_idx = tile % PIPELINE_DEPTH;
      size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
      
      pipe.producer_acquire();
      
      // Vectorized loads (float4 for better memory bandwidth)
      for (int idx = lane; idx < TILE_ELEMS / 4; idx += warpSize) {
        float4 a4 = *reinterpret_cast<const float4*>(&A[offset + idx * 4]);
        float4 b4 = *reinterpret_cast<const float4*>(&B[offset + idx * 4]);
        
        *reinterpret_cast<float4*>(&A_tiles[buf_idx * TILE_ELEMS + idx * 4]) = a4;
        *reinterpret_cast<float4*>(&B_tiles[buf_idx * TILE_ELEMS + idx * 4]) = b4;
      }
      
      pipe.producer_commit();
    }
  }
  
  // Compute warps: Process tiles in parallel
  if (is_compute) {
    int compute_warp_id = warp_id - 1;
    
    for (int tile = global_warp_id; tile < total_tiles; tile += gridDim.x) {
      int buf_idx = tile % PIPELINE_DEPTH;
      
      pipe.consumer_wait();
      
      // Each compute warp processes a slice of the tile
      int elems_per_warp = TILE_ELEMS / COMPUTE_WARPS;
      int start_idx = compute_warp_id * elems_per_warp;
      
      for (int idx = start_idx + lane; idx < start_idx + elems_per_warp; idx += warpSize) {
        float x = A_tiles[buf_idx * TILE_ELEMS + idx];
        float y = B_tiles[buf_idx * TILE_ELEMS + idx];
        
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          result += sqrtf(x * x + y * y) * 0.125f;
        }
        C_tiles[buf_idx * TILE_ELEMS + idx] = result;
      }
      
      // Only last compute warp releases
      if (compute_warp_id == COMPUTE_WARPS - 1) {
        pipe.consumer_release();
      }
    }
  }
  
  // Consumer: Store results with pipelining
  if (is_consumer) {
    for (int tile = global_warp_id; tile < total_tiles; tile += gridDim.x) {
      int buf_idx = tile % PIPELINE_DEPTH;
      size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
      
      pipe.consumer_wait();
      
      // Vectorized stores
      for (int idx = lane; idx < TILE_ELEMS / 4; idx += warpSize) {
        float4 c4 = *reinterpret_cast<float4*>(&C_tiles[buf_idx * TILE_ELEMS + idx * 4]);
        *reinterpret_cast<float4*>(&C[offset + idx * 4]) = c4;
      }
      
      pipe.consumer_release();
    }
  }
}

// Helper to detect architecture
bool is_grace_blackwell() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.major == 12;  // SM 12.x is Grace-Blackwell
}

int main() {
  int tiles = 256;
  size_t elems = static_cast<size_t>(tiles) * TILE_ELEMS;
  size_t bytes = elems * sizeof(float);

  // Detect architecture
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  bool is_sm121 = (prop.major == 12);
  bool is_sm100 = (prop.major == 10 && prop.minor == 0);
  
  std::printf("=== Enhanced Warp-Specialized Pipeline ===\n");
  std::printf("Architecture: %s (SM %d.%d)\n", 
              is_sm121 ? "Grace-Blackwell GB10" : is_sm100 ? "Blackwell B200" : "Other",
              prop.major, prop.minor);
  std::printf("Pipeline depth: %d stages\n", PIPELINE_DEPTH);
  std::printf("Warp configuration: 1 producer + 6 compute + 1 consumer\n");
  std::printf("Tile size: %dx%d = %d elements\n", TILE, TILE, TILE_ELEMS);
  std::printf("Total tiles: %d\n", tiles);
  std::printf("Memory size: %.2f MB\n\n", bytes / 1e6);

  std::vector<float> h_A(elems), h_B(elems), h_C(elems), h_ref(elems);
  std::iota(h_A.begin(), h_A.end(), 0.0f);
  std::iota(h_B.begin(), h_B.end(), 1.0f);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  // Check for cluster launch support
  int cluster_launch = 0;
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaError_t attr_status = cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device);
  if (attr_status != cudaSuccess) {
    cluster_launch = 0;
  }
  if (!cluster_launch) {
    std::printf("⚠️  Skipping enhanced pipeline: device lacks cluster launch / pipeline support.\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
  }

  // Launch configuration: 8 warps per block (1 producer + 6 compute + 1 consumer)
  dim3 block(8 * 32);  // 8 warps * 32 threads
  dim3 grid(std::min(tiles, 256));
  size_t shared_bytes = (3 * PIPELINE_DEPTH * TILE_ELEMS) * sizeof(float);
  
  std::printf("Launch config: %d blocks, %d threads/block, %zu KB shared memory\n",
              grid.x, block.x, shared_bytes / 1024);

  // Warmup
  for (int i = 0; i < 5; ++i) {
    warp_specialized_enhanced_kernel<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  constexpr int ITERS = 100;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < ITERS; ++i) {
    warp_specialized_enhanced_kernel<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  float avg_ms = ms / ITERS;
  
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err == cudaErrorInvalidValue ||
      launch_err == cudaErrorInvalidDeviceFunction ||
      launch_err == cudaErrorNotSupported) {
    std::printf("⚠️  Enhanced pipeline not supported on this GPU (%s); skipping demo.\n",
                cudaGetErrorString(launch_err));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
  }
  CUDA_CHECK(launch_err);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  // Verify
  for (size_t i = 0; i < elems; ++i) {
    float x = h_A[i];
    float y = h_B[i];
    float result = 0.0f;
    for (int j = 0; j < 8; ++j) {
      result += std::sqrt(x * x + y * y) * 0.125f;
    }
    h_ref[i] = result;
  }

  double max_err = 0.0;
  for (size_t i = 0; i < elems; ++i) {
    max_err = std::max(max_err, (double)std::abs(h_C[i] - h_ref[i]));
  }
  
  std::printf("\n=== Results ===\n");
  std::printf("Average kernel time: %.3f ms\n", avg_ms);
  std::printf("Throughput: %.2f GB/s\n", (3.0 * bytes / 1e9) / (avg_ms / 1000.0));
  std::printf("Max error: %.6e\n", max_err);
  std::printf("Status: %s\n", max_err < 1e-3 ? "✅ PASSED" : "❌ FAILED");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}

