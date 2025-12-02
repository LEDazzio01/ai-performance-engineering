/**
 * Persistent decode kernel for SM100+ (Blackwell).
 *
 * This kernel computes attention scores and weighted value aggregation.
 * Uses shared memory for efficient data movement.
 *
 * TMEM NOTE: TMEM is for Tensor Core MMA accumulators, not general data copies.
 * See labs/custom_vs_cublas/tcgen05_*.cu for proper TMEM+MMA usage.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace {

constexpr int MAX_HEAD_DIM = 128;
constexpr int MAX_SEQ_LEN = 64;

__device__ inline float dot_product(const float* q, const float* k, int head_dim, float* smem) {
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

__global__ void persistent_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int seq_len,
    int head_dim
) {
    const int seq_id = blockIdx.x;
    if (seq_id >= batch) return;
    if (seq_len > MAX_SEQ_LEN) return;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* accum = smem + blockDim.x;  // head_dim floats

    // Initialize accumulator
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        accum[d] = 0.0f;
    }
    __syncthreads();

    // Compute attention scores and accumulate weighted values
    for (int t = 0; t < seq_len; ++t) {
        const float* q_ptr = q + (seq_id * seq_len + t) * head_dim;
        const float* k_ptr = k + (seq_id * seq_len + t) * head_dim;
        const float* v_ptr = v + (seq_id * seq_len + t) * head_dim;

        float score = dot_product(q_ptr, k_ptr, head_dim, reduce_buf);
        
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            accum[d] += v_ptr[d] * score;
        }
        __syncthreads();
    }

    // Write output
    float* out_ptr = out + seq_id * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out_ptr[d] = accum[d];
    }
}

} // namespace

void persistent_decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, int blocks) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat, "q must be float32");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
    
    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(2));
    
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM, "head_dim exceeds MAX_HEAD_DIM");
    TORCH_CHECK(seq_len <= MAX_SEQ_LEN, "seq_len exceeds MAX_SEQ_LEN");
    TORCH_CHECK(out.size(0) == batch && out.size(1) == head_dim, "out shape mismatch");

    c10::cuda::CUDAGuard guard(q.get_device());
    
    const int threads = 64;
    const size_t smem_bytes = (threads + head_dim) * sizeof(float);
    
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    const int grid = std::min(blocks, batch);
    
    persistent_decode_kernel<<<grid, threads, smem_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim);
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persistent_decode", &persistent_decode_cuda, "Persistent decode (CUDA)");
}
