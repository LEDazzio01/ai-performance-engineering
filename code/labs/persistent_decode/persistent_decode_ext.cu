#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <algorithm>
#include <stdexcept>

#ifdef prefetch
#undef prefetch
#endif

#include <cute/algorithm/copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>

namespace {

__device__ inline float dot_tile_fallback(const float* q, const float* k, int head_dim) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        acc += q[d] * k[d];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
// TMEM-capable epilogue is only built on SM100-class GPUs.
constexpr bool kTmemAvailable = true;
#else
constexpr bool kTmemAvailable = false;
#endif

__global__ void persistent_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int seq_len,
    int head_dim
) {
    constexpr int MAX_HEAD_DIM = 128;
    const int seq_id = blockIdx.x;
    if (seq_id >= batch) {
        return;
    }
    constexpr int MAX_SEQ_LEN = 48;
    if (seq_len > MAX_SEQ_LEN) {
        return;
    }

    // shared layout: K0|K1|V0|V1|reduce|tmep_stage
    extern __shared__ float smem_f[];
    float* smem_k0 = smem_f;
    float* smem_k1 = smem_k0 + MAX_HEAD_DIM;
    float* smem_v0 = smem_k1 + MAX_HEAD_DIM;
    float* smem_v1 = smem_v0 + MAX_HEAD_DIM;
    float* red = smem_v1 + MAX_HEAD_DIM;
    float* tmep_stage = red + blockDim.x;  // up to MAX_SEQ_LEN x head_dim

    for (int t = 0; t < seq_len; ++t) {
        const float* q_ptr = q + (seq_id * seq_len + t) * head_dim;
        const float* k_ptr = k + (seq_id * seq_len + t) * head_dim;
        const float* v_ptr = v + (seq_id * seq_len + t) * head_dim;

        // fallback dot + scale
        float dot = dot_tile_fallback(q_ptr, k_ptr, head_dim);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            tmep_stage[t * head_dim + d] = v_ptr[d] * dot;
        }
        __syncthreads();
        // stash V into shared for possible future cp.async re-enable
        if (head_dim <= MAX_HEAD_DIM) {
            float* v_smem_curr = (t & 1) ? smem_v1 : smem_v0;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                v_smem_curr[d] = v_ptr[d];
            }
        }
        __syncthreads();
    }
    // TMEM epilogue when the tile matches 32x64 and TMEM is available.
    bool use_tmem = kTmemAvailable && (head_dim == 64) && (seq_len == 32);
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    if (use_tmem) {
        __shared__ uint32_t tmem_base_ptr;
        if (threadIdx.x == 0) {
            cute::TMEM::Allocator1Sm allocator{};
            allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base_ptr);
        }
        __syncthreads();

        auto tmem_tensor = cute::make_tensor(
            cute::make_tmem_ptr<float>(tmem_base_ptr),
            cute::make_layout(
                cute::make_shape(cute::Int<32>{}, cute::Int<64>{}),
                cute::make_stride(cute::TMEM::DP<float>{}, cute::Int<1>{})));

        auto smem_tensor = cute::make_tensor(
            cute::make_smem_ptr(tmep_stage),
            cute::make_layout(
                cute::make_shape(cute::Int<32>{}, cute::Int<64>{}),
                cute::make_stride(cute::Int<64>{}, cute::Int<1>{})));

        auto gmem_tensor = cute::make_tensor(
            cute::make_gmem_ptr(out + seq_id * seq_len * head_dim),
            cute::make_layout(
                cute::make_shape(cute::Int<32>{}, cute::Int<64>{}),
                cute::make_stride(cute::Int<64>{}, cute::Int<1>{})));

        auto tmem_store = cute::make_tmem_copy(cute::SM100_TMEM_STORE_32dp32b2x{}, tmem_tensor);
        auto tmem_load = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b2x{}, tmem_tensor);

        if (threadIdx.x < 32) {
            auto store_thr = tmem_store.get_slice(threadIdx.x);
            auto src = store_thr.partition_S(smem_tensor);
            auto dst = store_thr.partition_D(tmem_tensor);
            cute::copy(tmem_store, src, dst);
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            auto load_thr = tmem_load.get_slice(threadIdx.x);
            auto src = load_thr.partition_S(tmem_tensor);
            auto dst = load_thr.partition_D(gmem_tensor);
            cute::copy(tmem_load, src, dst);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            cute::TMEM::Allocator1Sm allocator{};
            allocator.release_allocation_lock();
            allocator.free(tmem_base_ptr, cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
        }
        return;
    }
#endif

    // Fallback store for all other shapes/hardware.
    const int total = seq_len * head_dim;
    const int base = seq_id * seq_len * head_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        out[base + idx] = tmep_stage[idx];
    }
}

} // namespace

void persistent_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, int blocks) {
    if (!q.is_cuda()) {
        throw std::runtime_error("q must be CUDA");
    }
    if (q.scalar_type() != torch::kFloat) {
        throw std::runtime_error("q must be float32");
    }
    if (!(q.sizes() == k.sizes() && q.sizes() == v.sizes())) {
        throw std::runtime_error("q/k/v shapes must match");
    }
    if (out.sizes() != q.sizes()) {
        throw std::runtime_error("out shape mismatch");
    }
    if (q.size(2) > 128) {
        throw std::runtime_error("head_dim exceeds MAX_HEAD_DIM=128");
    }

    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(2));
    const int threads = 64;

    constexpr int MAX_HEAD_DIM = 128;
    constexpr int MAX_SEQ_LEN = 48;
    const size_t smem_bytes = (4 * MAX_HEAD_DIM + threads + MAX_SEQ_LEN * MAX_HEAD_DIM) * sizeof(float);

    c10::cuda::CUDAGuard guard(q.get_device());
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, q.get_device());
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties failed");
    }
    if (prop.major < 8) {
        throw std::runtime_error("persistent_decode requires SM80+");
    }

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    const int blocks_per_batch = std::min(blocks, batch);
    persistent_decode_kernel<<<blocks_per_batch, threads, smem_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("persistent_decode_kernel launch failed");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persistent_decode", &persistent_decode_cuda, "Persistent decode (CUDA)");
}
