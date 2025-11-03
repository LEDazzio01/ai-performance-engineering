/*
 * Memory Pipeline Simulator - Chapter 10
 * =======================================
 * 
 * This demonstrates MEMORY-TO-MEMORY pipelining concepts using cuda::memcpy_async.
 * It simulates pipeline patterns that apply to storage I/O but uses GPU memory
 * operations for universal compatibility (works on any CUDA GPU).
 * 
 * ** THIS IS NOT STORAGE I/O **
 * This example transfers data between GPU memory regions to teach pipelining concepts.
 * 
 * For REAL GPUDirect Storage (storage-to-GPU transfers), see:
 *   - cufile_gds_example.py (production cuFile implementation)
 *   - CUFILE_GDS_GUIDE.md (complete implementation guide)
 *   - CUFILE_README.md (quick start)
 * 
 * Two Complementary Examples:
 * ---------------------------
 * THIS FILE (CUDA):     Memory pipeline simulator - teaches concepts (universal)
 * cufile_gds_example.py: Real GDS implementation - production code (GDS hardware)
 * 
 * Key Techniques Demonstrated:
 * - Double-buffered pipelining with cuda::pipeline
 * - Async memory copies with cuda::memcpy_async
 * - 32-byte vectorized loads on Blackwell (SM 10.0+)
 * - Persistent kernels for large problems (2048³+)
 * - Architecture-adaptive optimization
 * 
 * Hardware Target: NVIDIA B200/B300 (SM 10.0) and Grace-Blackwell (SM 12.1)
 * Also works on: Any CUDA GPU with compute capability 7.0+
 */

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <vector>

#include "../common/headers/arch_detection.cuh"

namespace cg = cooperative_groups;

constexpr int PIPELINE_STAGES = 2;

namespace {

struct ExecutionOptions {
    bool force_persistent = false;
    bool verbose = false;
};

ExecutionOptions g_exec_options{};

inline bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    if (value[0] == '0') {
        return false;
    }
    if (value[0] == '\0') {
        return false;
    }
    return !(value[0] == 'f' || value[0] == 'F' || value[0] == 'n' || value[0] == 'N');
}

struct OccupancyInfo {
    int active_blocks = 0;
    double occupancy = 0.0;
    cudaError_t status = cudaSuccess;
};

inline OccupancyInfo compute_occupancy(const void* kernel,
                                       int threads_per_block,
                                       size_t dynamic_shared_bytes) {
    OccupancyInfo info{};
    if (!kernel) {
        info.status = cudaErrorInvalidDeviceFunction;
        return info;
    }
    int active_blocks = 0;
    info.status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, threads_per_block, dynamic_shared_bytes);
    if (info.status != cudaSuccess) {
        return info;
    }
    int max_threads_per_sm = 0;
    if (cudaDeviceGetAttribute(&max_threads_per_sm,
                               cudaDevAttrMaxThreadsPerMultiProcessor, 0) != cudaSuccess ||
        max_threads_per_sm == 0) {
        max_threads_per_sm = 2048;  // sensible default for modern GPUs
    }
    info.active_blocks = active_blocks;
    info.occupancy =
        static_cast<double>(active_blocks * threads_per_block) / static_cast<double>(max_threads_per_sm);
    return info;
}

inline void print_usage(const char* program) {
    std::printf("Usage: %s [M N K] [--force-persistent] [--verbose] [--help]\n", program);
    std::printf("  M, N, K           Matrix dimensions (default: 1024 1024 1024)\n");
    std::printf("  --force-persistent  Force the persistent pipeline kernel when resources allow\n");
    std::printf("                      (or set DB_PIPELINE_FORCE_PERSISTENT=1)\n");
    std::printf("  --verbose           Print occupancy diagnostics (or DB_PIPELINE_VERBOSE=1)\n");
    std::printf("  --help              Display this message\n");
}

}  // namespace

namespace nvtx_helper {

struct NvtxSymbols {
    void* handle = nullptr;
    int (*range_push_a)(const char*) = nullptr;
    int (*range_pop)() = nullptr;
};

inline NvtxSymbols& symbols() {
    static NvtxSymbols sym{};
    static std::once_flag once{};
    std::call_once(once, []() {
        for (const char* candidate : {"libnvToolsExt.so", "libnvToolsExt.so.1"}) {
            sym.handle = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
            if (sym.handle != nullptr) {
                break;
            }
        }
        if (!sym.handle) {
            return;
        }
        sym.range_push_a = reinterpret_cast<int (*)(const char*)>(
            dlsym(sym.handle, "nvtxRangePushA"));
        sym.range_pop = reinterpret_cast<int (*)()>(
            dlsym(sym.handle, "nvtxRangePop"));
        if (!sym.range_push_a || !sym.range_pop) {
            dlclose(sym.handle);
            sym.handle = nullptr;
            sym.range_push_a = nullptr;
            sym.range_pop = nullptr;
        }
    });
    return sym;
}

class ScopedRange {
public:
    explicit ScopedRange(const char* name) {
        auto& sym = symbols();
        if (sym.range_push_a) {
            sym.range_push_a(name);
            active_ = true;
        }
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;

    ~ScopedRange() {
        auto& sym = symbols();
        if (active_ && sym.range_pop) {
            sym.range_pop();
        }
    }

private:
    bool active_ = false;
};

}  // namespace nvtx_helper

template <int TILE_M_, int TILE_N_, int TILE_K_, int BLOCK_THREADS_M_,
          int BLOCK_THREADS_N_, int THREAD_TILE_M_, int THREAD_TILE_N_,
          int CHUNK_K_>
struct TileConfig {
    static constexpr int TILE_M = TILE_M_;
    static constexpr int TILE_N = TILE_N_;
    static constexpr int TILE_K = TILE_K_;
    static constexpr int CHUNK_K = CHUNK_K_;
    static constexpr int BLOCK_THREADS_M = BLOCK_THREADS_M_;
    static constexpr int BLOCK_THREADS_N = BLOCK_THREADS_N_;
    static constexpr int THREAD_TILE_M = THREAD_TILE_M_;
    static constexpr int THREAD_TILE_N = THREAD_TILE_N_;
    static constexpr int BLOCK_THREADS = BLOCK_THREADS_M * BLOCK_THREADS_N_;
    static constexpr size_t SHARED_BYTES =
        PIPELINE_STAGES *
        (static_cast<size_t>(TILE_M) * CHUNK_K + static_cast<size_t>(CHUNK_K) * TILE_N) *
        sizeof(float);

    static_assert((TILE_M * TILE_K) % BLOCK_THREADS == 0,
                  "BLOCK_THREADS must evenly divide TILE_M * TILE_K");
    static_assert((TILE_K * TILE_N) % BLOCK_THREADS == 0,
                  "BLOCK_THREADS must evenly divide TILE_K * TILE_N");
    static_assert(TILE_K % CHUNK_K == 0,
                  "CHUNK_K must divide TILE_K");
};

// Pipeline configurations with different tile and chunk sizes
// Larger chunks reduce async overhead for big problems
using Config64Chunk64 = TileConfig<64, 64, 64, 16, 16, 4, 4, 64>;  // Full tile, large K
using Config64Chunk32 = TileConfig<64, 64, 64, 16, 16, 4, 4, 32>;  // Default (best for most cases)
using Config32 = TileConfig<32, 32, 32, 16, 16, 2, 2, 8>;
using Config16 = TileConfig<16, 16, 16, 16, 16, 1, 1, 8>;

// EDUCATIONAL: Large tile configs for 2048³+
// Fewer tiles = less pipeline overhead
using Config128Chunk32 = TileConfig<128, 128, 64, 16, 16, 8, 8, 32>;   // Large tile, moderate chunk
using Config128Chunk64 = TileConfig<128, 128, 64, 16, 16, 8, 8, 64>;   // Large tile, full chunk
using Config64Persistent = TileConfig<64, 64, 64, 16, 16, 4, 4, 32>;   // Persistent kernel (educational)

template <typename Config>
__device__ void zero_tile(float* tile, int elements) {
    const int linear = threadIdx.y * Config::BLOCK_THREADS_N + threadIdx.x;
    const int stride = Config::BLOCK_THREADS;
    for (int idx = linear; idx < elements; idx += stride) {
        tile[idx] = 0.0f;
    }
}

template <typename Config>
__device__ void copy_chunk_async(cg::thread_block cta,
                                 cuda::pipeline<cuda::thread_scope_block>& pipe,
                                 float* A_tile,
                                 float* B_tile,
                                 const float* A,
                                 const float* B,
                                 int M, int N, int K,
                                 int block_row,
                                 int block_col,
                                 int chunk_idx) {
    const int chunk_base = chunk_idx * Config::CHUNK_K;
    const int tile_rows_a = max(0, min(Config::TILE_M, M - block_row));
    const int tile_cols_b = max(0, min(Config::TILE_N, N - block_col));
    const int chunk_k = max(0, min(Config::CHUNK_K, K - chunk_base));

    // Use 32-byte (float8) vectorized copies on Blackwell for maximum coalescing
    // Falls back to 16-byte (float4) if not aligned
#if __CUDA_ARCH__ >= 1000  // Blackwell (SM 10.0+) and Grace-Blackwell (SM 12.1)
    constexpr int VEC_SIZE_LARGE = 8;  // float8 = 32 bytes (full cache line)
    constexpr int VEC_SIZE_SMALL = 4;  // float4 = 16 bytes
#else
    constexpr int VEC_SIZE_LARGE = 4;  // Older GPUs: max float4
    constexpr int VEC_SIZE_SMALL = 4;
#endif
    
    const int chunk_k_vec32 = (chunk_k / VEC_SIZE_LARGE) * VEC_SIZE_LARGE;
    const int chunk_k_vec16 = ((chunk_k - chunk_k_vec32) / VEC_SIZE_SMALL) * VEC_SIZE_SMALL;
    const int tile_cols_b_vec32 = (tile_cols_b / VEC_SIZE_LARGE) * VEC_SIZE_LARGE;
    const int tile_cols_b_vec16 = ((tile_cols_b - tile_cols_b_vec32) / VEC_SIZE_SMALL) * VEC_SIZE_SMALL;

    // Copy A tile with tiered vectorized loads (32B > 16B > 4B)
    for (int row = 0; row < tile_rows_a; ++row) {
        const float* src = A + static_cast<size_t>(block_row + row) * K + chunk_base;
        float* dst = A_tile + row * Config::CHUNK_K;
        int offset = 0;
        
        // 32-byte (float8) copies on Blackwell
        if (chunk_k_vec32 > 0) {
            cuda::memcpy_async(cta, dst, src, 
                              cuda::aligned_size_t<32>(chunk_k_vec32 * sizeof(float)), pipe);
            offset = chunk_k_vec32;
        }
        
        // 16-byte (float4) for remainder
        if (chunk_k_vec16 > 0) {
            cuda::memcpy_async(cta, dst + offset, src + offset,
                              cuda::aligned_size_t<16>(chunk_k_vec16 * sizeof(float)), pipe);
            offset += chunk_k_vec16;
        }
        
        // Scalar (4-byte) for final elements
        if (offset < chunk_k) {
            cuda::memcpy_async(cta, dst + offset, src + offset,
                              cuda::aligned_size_t<4>((chunk_k - offset) * sizeof(float)), pipe);
        }
    }

    // Copy B tile with tiered vectorized loads (32B > 16B > 4B)
    for (int row = 0; row < chunk_k; ++row) {
        const float* src = B + static_cast<size_t>(chunk_base + row) * N + block_col;
        float* dst = B_tile + row * Config::TILE_N;
        int offset = 0;
        
        // 32-byte (float8) copies on Blackwell
        if (tile_cols_b_vec32 > 0) {
            cuda::memcpy_async(cta, dst, src,
                              cuda::aligned_size_t<32>(tile_cols_b_vec32 * sizeof(float)), pipe);
            offset = tile_cols_b_vec32;
        }
        
        // 16-byte (float4) for remainder
        if (tile_cols_b_vec16 > 0) {
            cuda::memcpy_async(cta, dst + offset, src + offset,
                              cuda::aligned_size_t<16>(tile_cols_b_vec16 * sizeof(float)), pipe);
            offset += tile_cols_b_vec16;
        }
        
        // Scalar (4-byte) for final elements
        if (offset < tile_cols_b) {
            cuda::memcpy_async(cta, dst + offset, src + offset,
                              cuda::aligned_size_t<4>((tile_cols_b - offset) * sizeof(float)), pipe);
        }
    }

    // Zero out tail if chunk is partial
    if (chunk_k < Config::CHUNK_K) {
        for (int row = chunk_k; row < Config::CHUNK_K; ++row) {
            float* dst = B_tile + row * Config::TILE_N;
            for (int col = 0; col < Config::TILE_N; ++col) {
                dst[col] = 0.0f;
            }
        }
    }
}

template <typename Config>
__device__ void zero_chunk_out_of_bounds(float* A_tile,
                                         float* B_tile,
                                         int M, int N, int K,
                                         int block_row,
                                         int block_col,
                                         int chunk_base,
                                         int chunk_k) {
    const int linear = threadIdx.y * Config::BLOCK_THREADS_N + threadIdx.x;
    const int stride = Config::BLOCK_THREADS;

    for (int idx = linear; idx < Config::TILE_M * Config::CHUNK_K; idx += stride) {
        int row = idx / Config::CHUNK_K;
        int col = idx % Config::CHUNK_K;
        int global_row = block_row + row;
        int global_col = chunk_base + col;
        if (!(global_row < M && col < chunk_k && global_col < K)) {
            A_tile[row * Config::CHUNK_K + col] = 0.0f;
        }
    }

    for (int idx = linear; idx < Config::CHUNK_K * Config::TILE_N; idx += stride) {
        int row = idx / Config::TILE_N;
        int col = idx % Config::TILE_N;
        int global_row = chunk_base + row;
        int global_col = block_col + col;
        if (!(row < chunk_k && global_row < K && global_col < N)) {
            B_tile[row * Config::TILE_N + col] = 0.0f;
        }
    }
}

template <typename Config>
__global__ void gemm_tiled_pipeline_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    cg::thread_block cta = cg::this_thread_block();
    const int block_row = blockIdx.y * Config::TILE_M;
    const int block_col = blockIdx.x * Config::TILE_N;

    extern __shared__ float shared[];
    float* stage_ptr = shared;
    float* A_tiles[PIPELINE_STAGES];
    float* B_tiles[PIPELINE_STAGES];
    for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
        A_tiles[stage] = stage_ptr;
        stage_ptr += Config::TILE_M * Config::CHUNK_K;
        B_tiles[stage] = stage_ptr;
        stage_ptr += Config::CHUNK_K * Config::TILE_N;
    }

    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>;
    __shared__ alignas(pipeline_state_t) unsigned char pipe_storage[sizeof(pipeline_state_t)];
    auto* pipe_state = reinterpret_cast<pipeline_state_t*>(pipe_storage);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        new (pipe_state) pipeline_state_t();
    }
    cta.sync();
    auto pipe = cuda::make_pipeline(cta, pipe_state);

    const int total_chunks = (K + Config::CHUNK_K - 1) / Config::CHUNK_K;
    if (total_chunks == 0) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            pipe_state->~pipeline_state_t();
        }
        return;
    }

    for (int stage = 0; stage < PIPELINE_STAGES && stage < total_chunks; ++stage) {
        pipe.producer_acquire();
        copy_chunk_async<Config>(cta, pipe,
                                 A_tiles[stage], B_tiles[stage],
                                 A, B,
                                 M, N, K,
                                 block_row, block_col, stage);
        pipe.producer_commit();
    }

    const int thread_row = threadIdx.y * Config::THREAD_TILE_M;
    const int thread_col = threadIdx.x * Config::THREAD_TILE_N;
    float accum[Config::THREAD_TILE_M][Config::THREAD_TILE_N] = {0.0f};

    for (int chunk = 0; chunk < total_chunks; ++chunk) {
        const int stage = chunk % PIPELINE_STAGES;
        const int chunk_base = chunk * Config::CHUNK_K;
        const int chunk_k = max(0, min(Config::CHUNK_K, K - chunk_base));
        const int next_chunk = chunk + PIPELINE_STAGES;

        pipe.consumer_wait();
        cta.sync();
        zero_chunk_out_of_bounds<Config>(A_tiles[stage], B_tiles[stage],
                                         M, N, K,
                                         block_row, block_col,
                                         chunk_base, chunk_k);
        cta.sync();

        for (int kk = 0; kk < chunk_k; ++kk) {
            float a_frag[Config::THREAD_TILE_M];
            for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
                int row = thread_row + im;
                a_frag[im] = A_tiles[stage][row * Config::CHUNK_K + kk];
            }
            for (int jn = 0; jn < Config::THREAD_TILE_N; ++jn) {
                float b_val = B_tiles[stage][kk * Config::TILE_N + thread_col + jn];
                for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
                    accum[im][jn] += a_frag[im] * b_val;
                }
            }
        }

        cta.sync();
        pipe.consumer_release();

        if (next_chunk < total_chunks) {
            const int next_stage = next_chunk % PIPELINE_STAGES;
            pipe.producer_acquire();
            copy_chunk_async<Config>(cta, pipe,
                                     A_tiles[next_stage], B_tiles[next_stage],
                                     A, B,
                                     M, N, K,
                                     block_row, block_col, next_chunk);
            pipe.producer_commit();
        }
    }

    const int global_row = block_row + thread_row;
    const int global_col = block_col + thread_col;
    for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
        for (int jn = 0; jn < Config::THREAD_TILE_N; ++jn) {
            if (global_row + im < M && global_col + jn < N) {
                C[static_cast<size_t>(global_row + im) * N + (global_col + jn)] = accum[im][jn];
            }
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        pipe_state->~pipeline_state_t();
    }
}

// EDUCATIONAL: Persistent kernel for large problems
// Shows how to properly handle 2048³+ by keeping blocks alive
// This amortizes pipeline setup/teardown costs across multiple tiles
template <typename Config>
__global__ void gemm_tiled_persistent_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int M, int N, int K) {
    cg::thread_block cta = cg::this_thread_block();
    
    extern __shared__ float shared[];
    float* A_tiles[PIPELINE_STAGES];
    float* B_tiles[PIPELINE_STAGES];
    float* stage_ptr = shared;
    for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
        A_tiles[stage] = stage_ptr;
        stage_ptr += Config::TILE_M * Config::CHUNK_K;
        B_tiles[stage] = stage_ptr;
        stage_ptr += Config::CHUNK_K * Config::TILE_N;
    }

    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>;
    __shared__ alignas(pipeline_state_t) unsigned char pipe_storage[sizeof(pipeline_state_t)];
    auto* pipe_state = reinterpret_cast<pipeline_state_t*>(pipe_storage);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        new (pipe_state) pipeline_state_t();
    }
    cta.sync();
    auto pipe = cuda::make_pipeline(cta, pipe_state);

    const int thread_row = threadIdx.y * Config::THREAD_TILE_M;
    const int thread_col = threadIdx.x * Config::THREAD_TILE_N;
    
    // Grid-stride loop: process multiple output tiles per block
    const int tiles_m = (M + Config::TILE_M - 1) / Config::TILE_M;
    const int tiles_n = (N + Config::TILE_N - 1) / Config::TILE_N;
    const int total_output_tiles = tiles_m * tiles_n;
    const int blocks_per_grid = gridDim.x * gridDim.y;
    const int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Each block processes multiple output tiles
    for (int tile_id = block_id; tile_id < total_output_tiles; tile_id += blocks_per_grid) {
        const int tile_y = tile_id / tiles_n;
        const int tile_x = tile_id % tiles_n;
        const int block_row = tile_y * Config::TILE_M;
        const int block_col = tile_x * Config::TILE_N;
        
        float accum[Config::THREAD_TILE_M][Config::THREAD_TILE_N] = {0.0f};
        const int total_chunks = (K + Config::CHUNK_K - 1) / Config::CHUNK_K;
        
        // Preload pipeline
        for (int stage = 0; stage < PIPELINE_STAGES && stage < total_chunks; ++stage) {
            pipe.producer_acquire();
            copy_chunk_async<Config>(cta, pipe, A_tiles[stage], B_tiles[stage],
                                    A, B, M, N, K, block_row, block_col, stage);
            pipe.producer_commit();
        }
        
        // Main compute loop
        for (int chunk = 0; chunk < total_chunks; ++chunk) {
            const int stage = chunk % PIPELINE_STAGES;
            const int chunk_base = chunk * Config::CHUNK_K;
            const int chunk_k = min(Config::CHUNK_K, K - chunk_base);
            
            pipe.consumer_wait();
            cta.sync();

            zero_chunk_out_of_bounds<Config>(A_tiles[stage], B_tiles[stage],
                                             M, N, K,
                                             block_row, block_col,
                                             chunk_base, chunk_k);
            cta.sync();

            // Compute with this chunk
            #pragma unroll
            for (int kk = 0; kk < Config::CHUNK_K; ++kk) {
                if (kk < chunk_k) {
                    float a_frag[Config::THREAD_TILE_M];
                    #pragma unroll
                    for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
                        a_frag[im] = A_tiles[stage][(thread_row + im) * Config::CHUNK_K + kk];
                    }
                    #pragma unroll
                    for (int jn = 0; jn < Config::THREAD_TILE_N; ++jn) {
                        float b_val = B_tiles[stage][kk * Config::TILE_N + thread_col + jn];
                        #pragma unroll
                        for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
                            accum[im][jn] += a_frag[im] * b_val;
                        }
                    }
                }
            }
            
            pipe.consumer_release();
            
            // Preload next chunk
            const int next_chunk = chunk + PIPELINE_STAGES;
            if (next_chunk < total_chunks) {
                const int next_stage = next_chunk % PIPELINE_STAGES;
                pipe.producer_acquire();
                copy_chunk_async<Config>(cta, pipe, A_tiles[next_stage], B_tiles[next_stage],
                                        A, B, M, N, K, block_row, block_col, next_chunk);
                pipe.producer_commit();
            }
        }
        
        // Write results for this tile
        const int global_row = block_row + thread_row;
        const int global_col = block_col + thread_col;
        #pragma unroll
        for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
            #pragma unroll
            for (int jn = 0; jn < Config::THREAD_TILE_N; ++jn) {
                if (global_row + im < M && global_col + jn < N) {
                    C[static_cast<size_t>(global_row + im) * N + (global_col + jn)] = accum[im][jn];
                }
            }
        }
        
        // Only sync if processing another tile
        if (tile_id + blocks_per_grid < total_output_tiles) {
            cta.sync();
        }
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        pipe_state->~pipeline_state_t();
    }
}

template <typename Config>
__global__ void gemm_tiled_naive_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    cg::thread_block cta = cg::this_thread_block();
    const int block_row = blockIdx.y * Config::TILE_M;
    const int block_col = blockIdx.x * Config::TILE_N;

    extern __shared__ float shared[];
    float* A_tile = shared;
    float* B_tile = shared + Config::TILE_M * Config::TILE_K;

    const int thread_row = threadIdx.y * Config::THREAD_TILE_M;
    const int thread_col = threadIdx.x * Config::THREAD_TILE_N;
    float accum[Config::THREAD_TILE_M][Config::THREAD_TILE_N] = {0.0f};

    const int total_tiles = (K + Config::TILE_K - 1) / Config::TILE_K;
    for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
        const int k_base = tile_idx * Config::TILE_K;
        const int tile_k = max(0, min(Config::TILE_K, K - k_base));

        const int linear = threadIdx.y * Config::BLOCK_THREADS_N + threadIdx.x;
        const int stride = Config::BLOCK_THREADS;

        for (int idx = linear; idx < Config::TILE_M * Config::TILE_K; idx += stride) {
            int row = idx / Config::TILE_K;
            int col = idx % Config::TILE_K;
            int global_row = block_row + row;
            int global_col = k_base + col;
            A_tile[row * Config::TILE_K + col] =
                (global_row < M && global_col < K) ?
                A[static_cast<size_t>(global_row) * K + global_col] : 0.0f;
        }

        for (int idx = linear; idx < Config::TILE_K * Config::TILE_N; idx += stride) {
            int row = idx / Config::TILE_N;
            int col = idx % Config::TILE_N;
            int global_row = k_base + row;
            int global_col = block_col + col;
            B_tile[row * Config::TILE_N + col] =
                (global_row < K && global_col < N) ?
                B[static_cast<size_t>(global_row) * N + global_col] : 0.0f;
        }

        cta.sync();

        for (int kk = 0; kk < tile_k; ++kk) {
            float a_frag[Config::THREAD_TILE_M];
            for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
                int row = thread_row + im;
                a_frag[im] = A_tile[row * Config::TILE_K + kk];
            }
            for (int jn = 0; jn < Config::THREAD_TILE_N; ++jn) {
                float b_val = B_tile[kk * Config::TILE_N + thread_col + jn];
                for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
                    accum[im][jn] += a_frag[im] * b_val;
                }
            }
        }

        cta.sync();
    }

    const int global_row = block_row + thread_row;
    const int global_col = block_col + thread_col;
    for (int im = 0; im < Config::THREAD_TILE_M; ++im) {
        for (int jn = 0; jn < Config::THREAD_TILE_N; ++jn) {
            if (global_row + im < M && global_col + jn < N) {
                C[static_cast<size_t>(global_row + im) * N + (global_col + jn)] = accum[im][jn];
            }
        }
    }
}

template <typename Config, typename PipelineKernel>
void run_gemm_case_impl(int M, int N, int K,
                        const float* d_A,
                        const float* d_B,
                        float* d_C_naive,
                        float* d_C_pipeline,
                        float* naive_ms,
                        float* pipeline_ms,
                        PipelineKernel pipeline_kernel,
                        const char* pipeline_label) {
    dim3 block(Config::BLOCK_THREADS_N, Config::BLOCK_THREADS_M);
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N,
              (M + Config::TILE_M - 1) / Config::TILE_M);

    const size_t naive_smem = (Config::TILE_M * Config::TILE_K +
                               Config::TILE_K * Config::TILE_N) * sizeof(float);
    const size_t pipeline_smem = Config::SHARED_BYTES;

    // Set dynamic shared memory for both kernels if needed
    if (naive_smem > 49152) {
        cudaFuncSetAttribute(gemm_tiled_naive_kernel<Config>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(naive_smem));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C_naive, 0, static_cast<size_t>(M) * N * sizeof(float));
    cudaMemset(d_C_pipeline, 0, static_cast<size_t>(M) * N * sizeof(float));

    {
        nvtx_helper::ScopedRange range("naive_gemm");
        cudaEventRecord(start);
        for (int i = 0; i < 10; ++i) {
            gemm_tiled_naive_kernel<Config><<<grid, block, naive_smem>>>(
                d_A, d_B, d_C_naive, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(naive_ms, start, stop);
        *naive_ms /= 10.0f;
    }

    {
        nvtx_helper::ScopedRange range("pipeline_warmup");
        pipeline_kernel<<<grid, block, pipeline_smem>>>(
            d_A, d_B, d_C_pipeline, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "%s launch error: %s\n",
                         pipeline_label,
                         cudaGetErrorString(err));
            *pipeline_ms = 0.0f;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return;
        }
        cudaDeviceSynchronize();
    }

    {
        nvtx_helper::ScopedRange range("pipeline_gemm");
        cudaEventRecord(start);
        for (int i = 0; i < 10; ++i) {
            pipeline_kernel<<<grid, block, pipeline_smem>>>(
                d_A, d_B, d_C_pipeline, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(pipeline_ms, start, stop);
        *pipeline_ms /= 10.0f;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (g_exec_options.verbose) {
        const int threads_per_block = block.x * block.y;
        auto naive_occ = compute_occupancy(
            reinterpret_cast<const void*>(gemm_tiled_naive_kernel<Config>),
            threads_per_block, naive_smem);
        auto pipe_occ = compute_occupancy(
            reinterpret_cast<const void*>(pipeline_kernel),
            threads_per_block, pipeline_smem);
        if (naive_occ.status == cudaSuccess) {
            std::printf("  [occupancy] naive: %d blocks/SM (~%.1f%%)\n",
                        naive_occ.active_blocks, naive_occ.occupancy * 100.0);
        } else {
            std::printf("  [occupancy] naive: failed (%s)\n",
                        cudaGetErrorString(naive_occ.status));
        }
        if (pipe_occ.status == cudaSuccess) {
            std::printf("  [occupancy] pipeline: %d blocks/SM (~%.1f%%)\n",
                        pipe_occ.active_blocks, pipe_occ.occupancy * 100.0);
        } else {
            std::printf("  [occupancy] pipeline: failed (%s)\n",
                        cudaGetErrorString(pipe_occ.status));
        }
    }
}

template <typename Config>
void run_gemm_case(int M, int N, int K,
                   const float* d_A,
                   const float* d_B,
                   float* d_C_naive,
                   float* d_C_pipeline,
                   float* naive_ms,
                   float* pipeline_ms) {
    if (Config::SHARED_BYTES > 49152) {
        cudaError_t attr_err = cudaFuncSetAttribute(gemm_tiled_pipeline_kernel<Config>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(Config::SHARED_BYTES));
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "cudaFuncSetAttribute failed for %zu bytes: %s\n",
                    Config::SHARED_BYTES, cudaGetErrorString(attr_err));
            *pipeline_ms = -1.0f;
            return;
        }
    }
    run_gemm_case_impl<Config>(
        M, N, K, d_A, d_B, d_C_naive, d_C_pipeline, naive_ms, pipeline_ms,
        gemm_tiled_pipeline_kernel<Config>, "Pipeline kernel");
}

template <typename Config>
void run_gemm_case_persistent(int M, int N, int K,
                              const float* d_A,
                              const float* d_B,
                              float* d_C_naive,
                              float* d_C_pipeline,
                              float* naive_ms,
                              float* pipeline_ms) {
    if (Config::SHARED_BYTES > 49152) {
        cudaFuncSetAttribute(gemm_tiled_persistent_kernel<Config>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(Config::SHARED_BYTES));
    }
    
    // Use smaller grid for persistent kernel (grid-stride loop)
    dim3 block(Config::BLOCK_THREADS_N, Config::BLOCK_THREADS_M);
    
    const int tiles_m = (M + Config::TILE_M - 1) / Config::TILE_M;
    const int tiles_n = (N + Config::TILE_N - 1) / Config::TILE_N;
    
    dim3 grid_naive(tiles_n, tiles_m);
    
    // For 2048³: grid would be 32×32 = 1024 blocks normally
    // Use moderately smaller grid to balance parallelism and amortization
    // Too small (4×4) → each block does too many tiles (overhead)
    // Too large (32×32) → no amortization benefit
    const int total_tiles = tiles_m * tiles_n;
    const int persistent_grid_dim = (total_tiles >= 256) ? 16 : 8;
    dim3 grid_persistent(persistent_grid_dim, persistent_grid_dim);
    
    const size_t naive_smem = (Config::TILE_M * Config::TILE_K +
                               Config::TILE_K * Config::TILE_N) * sizeof(float);
    const size_t pipeline_smem = Config::SHARED_BYTES;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMemset(d_C_naive, 0, static_cast<size_t>(M) * N * sizeof(float));
    cudaMemset(d_C_pipeline, 0, static_cast<size_t>(M) * N * sizeof(float));
    
    // Run naive kernel
    {
        nvtx_helper::ScopedRange range("naive_gemm");
        cudaEventRecord(start);
        for (int i = 0; i < 10; ++i) {
            gemm_tiled_naive_kernel<Config><<<grid_naive, block, naive_smem>>>(
                d_A, d_B, d_C_naive, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(naive_ms, start, stop);
        *naive_ms /= 10.0f;
    }
    
    // Warmup persistent kernel
    {
        nvtx_helper::ScopedRange range("pipeline_warmup");
        gemm_tiled_persistent_kernel<Config><<<grid_persistent, block, pipeline_smem>>>(
            d_A, d_B, d_C_pipeline, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Persistent kernel launch error: %s\n",
                         cudaGetErrorString(err));
            *pipeline_ms = 0.0f;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return;
        }
        cudaDeviceSynchronize();
    }
    
    // Benchmark persistent kernel
    {
        nvtx_helper::ScopedRange range("pipeline_gemm");
        cudaEventRecord(start);
        for (int i = 0; i < 10; ++i) {
            gemm_tiled_persistent_kernel<Config><<<grid_persistent, block, pipeline_smem>>>(
                d_A, d_B, d_C_pipeline, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(pipeline_ms, start, stop);
        *pipeline_ms /= 10.0f;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (g_exec_options.verbose) {
        const int threads_per_block = block.x * block.y;
        auto naive_occ = compute_occupancy(
            reinterpret_cast<const void*>(gemm_tiled_naive_kernel<Config>),
            threads_per_block, naive_smem);
        auto persistent_occ = compute_occupancy(
            reinterpret_cast<const void*>(gemm_tiled_persistent_kernel<Config>),
            threads_per_block, pipeline_smem);
        if (naive_occ.status == cudaSuccess) {
            std::printf("  [occupancy] naive: %d blocks/SM (~%.1f%%)\n",
                        naive_occ.active_blocks, naive_occ.occupancy * 100.0);
        } else {
            std::printf("  [occupancy] naive: failed (%s)\n",
                        cudaGetErrorString(naive_occ.status));
        }
        if (persistent_occ.status == cudaSuccess) {
            std::printf("  [occupancy] persistent: %d blocks/SM (~%.1f%%)\n",
                        persistent_occ.active_blocks, persistent_occ.occupancy * 100.0);
        } else {
            std::printf("  [occupancy] persistent: failed (%s)\n",
                        cudaGetErrorString(persistent_occ.status));
        }
    }
}

template <typename Config>
bool config_fits(const cuda_arch::ArchitectureLimits& limits) {
    if (Config::BLOCK_THREADS > limits.max_threads_per_block) {
        return false;
    }
    return Config::SHARED_BYTES <= static_cast<size_t>(limits.max_shared_mem_per_block);
}

inline bool always_true(int, int, int) {
    return true;
}

// Chunk32 provides best balance - chunk64 reduces occupancy too much
inline bool prefer_chunk32(int, int, int) {
    return true;  // Fallback when other heuristics reject current problem
}

inline bool prefer_chunk64(int M, int N, int K) {
    if (g_exec_options.force_persistent) {
        return false;
    }
    if (K < Config64Chunk64::CHUNK_K) {
        return false;
    }
    if (M < Config64Chunk64::TILE_M || N < Config64Chunk64::TILE_N) {
        return false;
    }
    const int tiles_m = (M + Config64Chunk64::TILE_M - 1) / Config64Chunk64::TILE_M;
    const int tiles_n = (N + Config64Chunk64::TILE_N - 1) / Config64Chunk64::TILE_N;
    const int total_tiles = tiles_m * tiles_n;
    const bool large_k = K >= 1024;
    const bool enough_tiles = total_tiles >= 64;
    return large_k && enough_tiles;
}

inline bool prefer_persistent_large(int M, int N, int K) {
    if (!g_exec_options.force_persistent) {
        return false;
    }
    if (M < Config64Persistent::TILE_M || N < Config64Persistent::TILE_N) {
        return false;
    }
    if (K < Config64Persistent::CHUNK_K * 2) {
        return false;
    }
    const int tiles_m = (M + Config64Persistent::TILE_M - 1) / Config64Persistent::TILE_M;
    const int tiles_n = (N + Config64Persistent::TILE_N - 1) / Config64Persistent::TILE_N;
    const int total_tiles = tiles_m * tiles_n;
    if (total_tiles < 64) {
        return false;
    }
    const auto& limits = cuda_arch::get_architecture_limits();
    if (Config64Persistent::SHARED_BYTES > static_cast<size_t>(limits.max_shared_mem_per_block)) {
        return false;
    }
    return true;
}


int main(int argc, char** argv) {
    g_exec_options.force_persistent |= env_flag_enabled("DB_PIPELINE_FORCE_PERSISTENT");
    g_exec_options.verbose |= env_flag_enabled("DB_PIPELINE_VERBOSE");

    bool show_help = false;
    std::vector<int> dims;
    dims.reserve(3);

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--force-persistent") == 0) {
            g_exec_options.force_persistent = true;
            continue;
        }
        if (std::strcmp(arg, "--verbose") == 0) {
            g_exec_options.verbose = true;
            continue;
        }
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            show_help = true;
            continue;
        }
        char* endptr = nullptr;
        long value = std::strtol(arg, &endptr, 10);
        if (endptr && *endptr == '\0') {
            dims.push_back(static_cast<int>(value));
            continue;
        }
        std::fprintf(stderr, "Unrecognized argument: %s\n", arg);
        print_usage(argv[0]);
        return 1;
    }

    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }

    if (dims.size() > 3) {
        std::fprintf(stderr, "Too many dimension arguments provided.\n");
        print_usage(argv[0]);
        return 1;
    }

    int M = (dims.size() > 0) ? dims[0] : 1024;
    int N = (dims.size() > 1) ? dims[1] : 1024;
    int K = (dims.size() > 2) ? dims[2] : 1024;

    if (M <= 0 || N <= 0 || K <= 0) {
        std::fprintf(stderr, "Matrix dimensions must be positive integers.\n");
        return 1;
    }

    const auto& limits = cuda_arch::get_architecture_limits();

    struct Candidate {
        const char* name;
        bool (*fits)(const cuda_arch::ArchitectureLimits&);
        bool (*suitable)(int, int, int);
        void (*runner)(int, int, int, const float*, const float*, float*, float*, float*, float*);
    };

    // Heuristic for 128×128 tiles: very large problems only
    auto prefer_128x128 = [](int M, int N, int K) {
        if (g_exec_options.force_persistent) {
            return false;
        }
        return M >= 2048 && N >= 2048 && K >= 512;
    };
    
    const Candidate candidates[] = {
        {"128x128 (chunk64)", &config_fits<Config128Chunk64>, prefer_128x128, &run_gemm_case<Config128Chunk64>},
        {"128x128 (chunk32)", &config_fits<Config128Chunk32>, prefer_128x128, &run_gemm_case<Config128Chunk32>},
        {"64x64 (chunk64, vectorized)", &config_fits<Config64Chunk64>, &prefer_chunk64, &run_gemm_case<Config64Chunk64>},
        {"64x64 persistent (chunk32)", &config_fits<Config64Persistent>, &prefer_persistent_large, &run_gemm_case_persistent<Config64Persistent>},
        {"64x64 (chunk32, vectorized)", &config_fits<Config64Chunk32>, &prefer_chunk32, &run_gemm_case<Config64Chunk32>},
        {"32x32", &config_fits<Config32>, &always_true, &run_gemm_case<Config32>},
        {"16x16", &config_fits<Config16>, &always_true, &run_gemm_case<Config16>}
    };

    const Candidate* selected = nullptr;
    for (const auto& cand : candidates) {
        if (cand.fits(limits) && cand.suitable(M, N, K)) {
            selected = &cand;
            break;
        }
    }
    
    // Debug: print device limits for Config128
    if (M >= 2048) {
        printf("Device limits: max_shared_mem_per_block=%d bytes (%d KB)\n",
               limits.max_shared_mem_per_block, limits.max_shared_mem_per_block / 1024);
        printf("Config128Chunk32 needs: %zu bytes (%zu KB)\n",
               Config128Chunk32::SHARED_BYTES, Config128Chunk32::SHARED_BYTES / 1024);
    }
    if (!selected) {
        std::fprintf(stderr, "No suitable tile configuration found for this GPU.\n");
        return 1;
    }

    if (g_exec_options.force_persistent &&
        std::strcmp(selected->name, "64x64 persistent (chunk32)") != 0) {
        std::printf("Warning: persistent kernel not selected (resource or size limits). "
                    "Falling back to %s.\n", selected->name);
    }

    std::printf("=== Double-Buffered Pipeline GEMM ===\n");
    std::printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Selected tile configuration: %s\n", selected->name);

    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_B(static_cast<size_t>(K) * N);
    std::vector<float> h_C_naive(static_cast<size_t>(M) * N);
    std::vector<float> h_C_pipeline(static_cast<size_t>(M) * N);

    for (auto& v : h_A) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : h_B) v = static_cast<float>(rand()) / RAND_MAX;

    float *d_A = nullptr, *d_B = nullptr, *d_C_naive = nullptr, *d_C_pipeline = nullptr;
    cudaMalloc(&d_A, h_A.size() * sizeof(float));
    cudaMalloc(&d_B, h_B.size() * sizeof(float));
    cudaMalloc(&d_C_naive, h_C_naive.size() * sizeof(float));
    cudaMalloc(&d_C_pipeline, h_C_pipeline.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    float naive_ms = 0.0f;
    float pipeline_ms = 0.0f;
    selected->runner(M, N, K, d_A, d_B, d_C_naive, d_C_pipeline, &naive_ms, &pipeline_ms);

    cudaMemcpy(h_C_naive.data(), d_C_naive, h_C_naive.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_pipeline.data(), d_C_pipeline, h_C_pipeline.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double max_diff = 0.0;
    size_t max_index = 0;
    for (size_t i = 0; i < h_C_naive.size(); ++i) {
        double diff = std::abs(static_cast<double>(h_C_naive[i] - h_C_pipeline[i]));
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    if (max_diff > 1e-3) {
        size_t row = max_index / static_cast<size_t>(N);
        size_t col = max_index % static_cast<size_t>(N);
        std::printf("Max diff at (%zu, %zu): naive=%.6f pipeline=%.6f\n",
                    row, col,
                    static_cast<double>(h_C_naive[max_index]),
                    static_cast<double>(h_C_pipeline[max_index]));
    }

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double naive_gflops = flops / (naive_ms * 1e6);
    const double pipeline_gflops = (pipeline_ms > 0.0f) ? flops / (pipeline_ms * 1e6) : 0.0;

    std::printf("Naive time: %.2f ms (%.1f GFLOPS)\n", naive_ms, naive_gflops);
    if (pipeline_ms > 0.0f) {
        std::printf("Pipeline time: %.2f ms (%.1f GFLOPS)\n", pipeline_ms, pipeline_gflops);
        std::printf("Speedup: %.2fx\n", naive_ms / pipeline_ms);
    } else {
        std::printf("Pipeline kernel failed to launch.\n");
    }
    std::printf("Max difference: %.3e\n", max_diff);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_pipeline);

    return 0;
}
