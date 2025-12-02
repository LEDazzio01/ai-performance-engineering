// Book sample (Chapter 18): illustrative FlashMLA-like decode sketch for Blackwell (sm_120).
// For production use FlashMLA/ThunderMLA or PyTorch FlexDecoding kernels; error handling omitted.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void flashmla_decode(const half* __restrict__ q,
                                const half* __restrict__ k_cache,
                                const half* __restrict__ v_cache,
                                half* __restrict__ out,
                                const int* __restrict__ lengths,
                                int num_heads,
                                int head_dim,
                                int stride) {
  int batch = blockIdx.x / num_heads;
  int head = blockIdx.x % num_heads;
  if (threadIdx.x >= head_dim) return;
  const half* q_head = q + (batch * num_heads + head) * head_dim;
  const half* k_head = k_cache + batch * stride + head * head_dim;
  const half* v_head = v_cache + batch * stride + head * head_dim;
  half acc = __float2half(0.f);
  half denom = __float2half(0.f);
  for (int pos = 0; pos < lengths[batch]; ++pos) {
    half score = __float2half(0.f);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      score = __hfma(q_head[d], k_head[pos * num_heads * head_dim + d], score);
    }
    score = __hdiv(score, __float2half(sqrtf((float)head_dim)));
    half weight = hexp(score);
    denom = __hadd(denom, weight);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      acc = __hfma(weight, v_head[pos * num_heads * head_dim + d], acc);
    }
  }
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    out[(batch * num_heads + head) * head_dim + d] = __hdiv(acc, denom);
  }
}

int main() {
  printf("FlashMLA decode sketch\n");
  return 0;
}
