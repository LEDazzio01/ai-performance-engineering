#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/algorithm/copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>

#ifdef prefetch
#undef prefetch
#endif

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)

template <typename T>
struct TmemCopyOps;

template <>
struct TmemCopyOps<float> {
  using Store = cute::SM100_TMEM_STORE_32dp32b4x;
  using Load = cute::SM100_TMEM_LOAD_32dp32b4x;
};

template <>
struct TmemCopyOps<at::Half> {
  using Store = cute::SM100_TMEM_STORE_32dp32b4x_16b;
  using Load = cute::SM100_TMEM_LOAD_32dp32b4x_16b;
};

template <>
struct TmemCopyOps<at::BFloat16> {
  using Store = cute::SM100_TMEM_STORE_32dp32b4x_16b;
  using Load = cute::SM100_TMEM_LOAD_32dp32b4x_16b;
};

template <typename T>
__global__ void tmem_cache_copy_kernel(const T* __restrict__ src,
                                       T* __restrict__ dst,
                                       int ld,
                                       int rows) {
  constexpr int TILE_M = 32;
  constexpr int TILE_N = 128;
  const int tile_row = blockIdx.x * TILE_M;
  if (tile_row + TILE_M > rows) {
    return;
  }

  __shared__ uint32_t tmem_base_ptr;
  if (threadIdx.x == 0) {
    cute::TMEM::Allocator1Sm allocator{};
    allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }
  __syncthreads();

  auto tmem_tensor = cute::make_tensor(
      cute::make_tmem_ptr<T>(tmem_base_ptr),
      cute::make_layout(
          cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
          cute::make_stride(cute::TMEM::DP<T>{}, cute::Int<1>{})));

  auto src_tensor = cute::make_tensor(
      cute::make_gmem_ptr(src + tile_row * ld),
      cute::make_layout(
          cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
          cute::make_stride(cute::Int<TILE_N>{}, cute::Int<1>{})));

  auto dst_tensor = cute::make_tensor(
      cute::make_gmem_ptr(dst + tile_row * ld),
      cute::make_layout(
          cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
          cute::make_stride(cute::Int<TILE_N>{}, cute::Int<1>{})));

  auto store_copy = cute::make_tmem_copy(typename TmemCopyOps<T>::Store{}, tmem_tensor);
  auto load_copy = cute::make_tmem_copy(typename TmemCopyOps<T>::Load{}, tmem_tensor);

  if (threadIdx.x < 32) {
    auto store_thr = store_copy.get_slice(threadIdx.x);
    auto src_frag = store_thr.partition_S(src_tensor);
    auto dst_frag = store_thr.partition_D(tmem_tensor);
    cute::copy(store_copy, src_frag, dst_frag);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    auto load_thr = load_copy.get_slice(threadIdx.x);
    auto src_frag = load_thr.partition_S(tmem_tensor);
    auto dst_frag = load_thr.partition_D(dst_tensor);
    cute::copy(load_copy, src_frag, dst_frag);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    cute::TMEM::Allocator1Sm allocator{};
    allocator.release_allocation_lock();
    allocator.free(tmem_base_ptr, cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
  }
}

template <typename T>
void launch_tmem_copy(const torch::Tensor& src, const torch::Tensor& dst, int rows, int ld) {
  dim3 block(32);
  dim3 grid(rows / 32);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  tmem_cache_copy_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(src.data_ptr<T>()),
      reinterpret_cast<T*>(dst.data_ptr<T>()),
      ld,
      rows);
  AT_CUDA_CHECK(cudaGetLastError());
}

#endif  // CUTE_ARCH_TCGEN05_TMEM_ENABLED

void tmem_cache_copy(torch::Tensor src, torch::Tensor dst) {
#if !defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
  TORCH_CHECK(
      false,
      "TMEM cache epilogue requires building with tcgen05/TMEM support (SM100+)—"
      "current build lacks CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#else
  TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "TMEM cache copy expects CUDA tensors");
  TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "Source/destination dtypes must match");
  TORCH_CHECK(src.dim() == 2 && dst.dim() == 2, "TMEM cache copy expects 2D tensors");
  TORCH_CHECK(src.sizes() == dst.sizes(), "Source/destination shapes must match");

  const int64_t rows = src.size(0);
  const int64_t cols = src.size(1);
  TORCH_CHECK(cols == 128, "TMEM cache copy currently supports head_dim == 128");
  TORCH_CHECK(rows % 32 == 0, "TMEM cache copy requires rows to be a multiple of 32");

  auto src_contig = src.contiguous();
  auto dst_contig = dst.contiguous();
  const int64_t ld = src_contig.stride(0);
  TORCH_CHECK(ld == cols, "TMEM cache copy requires contiguous row-major layout");

  c10::cuda::CUDAGuard guard(src_contig.get_device());
  cudaDeviceProp prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, src_contig.get_device()));
  TORCH_CHECK(prop.major >= 10, "TMEM cache copy requires SM100-class GPU");

  switch (src_contig.scalar_type()) {
    case torch::kFloat32:
      launch_tmem_copy<float>(src_contig, dst_contig, static_cast<int>(rows), static_cast<int>(ld));
      break;
    case torch::kBFloat16:
      launch_tmem_copy<at::BFloat16>(src_contig, dst_contig, static_cast<int>(rows), static_cast<int>(ld));
      break;
    case torch::kFloat16:
      launch_tmem_copy<at::Half>(src_contig, dst_contig, static_cast<int>(rows), static_cast<int>(ld));
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for TMEM cache copy");
  }

  if (!dst.is_contiguous()) {
    dst.copy_(dst_contig);
  }
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tmem_cache_copy", &tmem_cache_copy, "TMEM-backed tile copy for KV cache tensors");
}
