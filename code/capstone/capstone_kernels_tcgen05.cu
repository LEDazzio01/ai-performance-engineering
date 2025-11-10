#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <type_traits>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>

#include <cute/algorithm/axpby.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

namespace cg = cooperative_groups;
using namespace cute;

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;
using Alpha = float;
using Beta = float;

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

template <typename Allocator>
struct ClusterTmemHelper;

template <>
struct ClusterTmemHelper<cute::TMEM::Allocator1Sm> {
  __device__ static uint32_t* slot(uint32_t* local) { return local; }
  __device__ static void sync_before_alloc() {}
  __device__ static void sync_after_alloc() {}
  __device__ static void sync_before_free() {}
  __device__ static void sync_after_free() {}
};

template <>
struct ClusterTmemHelper<cute::TMEM::Allocator2Sm> {
  __device__ static uint32_t* slot(uint32_t* local) {
    auto cluster = cg::this_cluster();
    return cluster.map_shared_rank(local, 0);
  }

  __device__ static void sync_before_alloc() { cg::this_cluster().sync(); }
  __device__ static void sync_after_alloc() { cg::this_cluster().sync(); }
  __device__ static void sync_before_free() { cg::this_cluster().sync(); }
  __device__ static void sync_after_free() { cg::this_cluster().sync(); }
};

template <class MmaTag, int ClusterM>
struct TcgenVariantConfig {
  using Mma = MmaTag;
  static constexpr int kClusterM = ClusterM;
  static constexpr int kClusterN = 1;
  static constexpr int kClusterK = 1;
  using TmemAllocator = std::conditional_t<
      ClusterM == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  static constexpr auto cluster_shape() {
    return make_shape(Int<ClusterM>{}, Int<1>{}, Int<1>{});
  }
};

using VariantCTA1 = TcgenVariantConfig<
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                         UMMA::Major::K, UMMA::Major::K>,
    1>;

using VariantCTA2 = TcgenVariantConfig<
    SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256,
                               UMMA::Major::K, UMMA::Major::K>,
    2>;

template <class Variant,
          class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB,
          class AlphaT, class BetaT>
__global__ void gemm_device_variant(ATensor mA,
                                    BTensor mB,
                                    CTensor mC,
                                    DTensor mD,
                                    MmaTiler_MNK mma_tiler,
                                    TiledMMA tiled_mma,
                                    ClusterShape_MNK cluster_shape,
                                    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                                    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
                                    AlphaT alpha,
                                    BetaT beta) {
  Layout cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename TiledMMA::AtomThrID{}));

  auto mma_coord_vmnk =
      make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                 blockIdx.x / size<0>(cluster_layout_vmnk),
                 blockIdx.y,
                 _);

  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  auto cluster = cg::this_cluster();
  SharedStorageT &shared_storage =
      *reinterpret_cast<SharedStorageT*>(shared_memory);

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  auto mma_v = get<0>(mma_coord_vmnk);
  auto cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = typename Variant::TmemAllocator;
  TmemAllocator tmem_allocator{};
  auto *tmem_slot =
      ClusterTmemHelper<TmemAllocator>::slot(&shared_storage.tmem_base_ptr);

  ClusterTmemHelper<TmemAllocator>::sync_before_alloc();
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            tmem_slot);
  }
  ClusterTmemHelper<TmemAllocator>::sync_after_alloc();
  __syncthreads();
  uint32_t tmem_base = *tmem_slot;
  tCtAcc.data() = tmem_base;

  auto [tAgA, tAsA] = tma_partition(
      tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));
  auto [tBgB, tBsB] = tma_partition(
      tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

  int tma_transaction_bytes =
      sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));

  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, 1);
    cute::initialize_barrier(shared_storage.tma_barrier, 1);
  }
  int mma_barrier_phase_bit = 0;
  int tma_barrier_phase_bit = 0;
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier,
                                          tma_transaction_bytes);
      copy(tma_atom_A.with(shared_storage.tma_barrier), tAgA(_, k_tile), tAsA);
      copy(tma_atom_B.with(shared_storage.tma_barrier), tBgB(_, k_tile), tBsB);
    }

    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

    if (elect_one_warp) {
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }

    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
  Tensor tDrC = make_fragment_like(tDgC);
  copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  axpby(alpha, tDrAcc, beta, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();
  ClusterTmemHelper<TmemAllocator>::sync_before_free();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
  ClusterTmemHelper<TmemAllocator>::sync_after_free();
}

template <class Variant>
torch::Tensor run_tcgen05_variant(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D inputs");
  TORCH_CHECK(a.size(1) == b.size(0), "incompatible matmul shapes");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "tcgen05 kernels expect float16 inputs");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must reside on CUDA");

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();

  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_contig.size(1);

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::zeros({m, n}, options);
  auto d_buffer = torch::empty_like(c_buffer);

  auto cluster_shape = Variant::cluster_shape();
  auto tiled_mma = make_tiled_mma(typename Variant::Mma{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  TORCH_CHECK(evenly_divides(shape(mma_tiler), tile_shape(tiled_mma)),
              "tcgen05 MMA tile must divide instruction tile");
  TORCH_CHECK(
      evenly_divides(make_shape(m, n, k), mma_tiler),
      "Problem size must be divisible by the tcgen05 tile (128x256x64 or "
      "256x256x64 depending on the variant)");

  auto mma_shape_A =
      partition_shape_A(tiled_mma,
                        make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B =
      partition_shape_B(tiled_mma,
                        make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{},
                              mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{},
                              mma_shape_B);

  using SharedStorageT =
      SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  Tensor mA = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeA const*>(
          a_contig.data_ptr<at::Half>())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(
      make_gmem_ptr(reinterpret_cast<TypeB const*>(
          b_contig.data_ptr<at::Half>())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(
      make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(
      make_gmem_ptr(d_buffer.data_ptr<TypeD>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto tma_atom_A =
      make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout, select<0, 2>(mma_tiler));
  auto tma_atom_B =
      make_tma_atom(SM90_TMA_LOAD{}, mB, sB_layout, select<1, 2>(mma_tiler));

  auto cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  auto cluster_m_tiles = size<1>(cluster_layout_vmnk);
  auto cluster_n_tiles = size<2>(cluster_layout_vmnk);

  int tile_m = size(bM) * cluster_m_tiles;
  int tile_n = size(bN) * cluster_n_tiles;

  dim3 dimBlock(128);
  dim3 dimCluster(Variant::kClusterM,
                  Variant::kClusterN,
                  Variant::kClusterK);
  dim3 dimGrid(
      (m + tile_m - 1) / tile_m * dimCluster.x,
      (n + tile_n - 1) / tile_n * dimCluster.y,
      1);

  int smem_bytes = sizeof(SharedStorageT);

  auto *kernel_ptr = &gemm_device_variant<
      Variant,
      SharedStorageT,
      decltype(mA), decltype(mB),
      decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma),
      decltype(cluster_shape),
      decltype(tma_atom_A), decltype(tma_atom_B),
      Alpha, Beta>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  cutlass::ClusterLaunchParams params{
      dimGrid, dimBlock, dimCluster, smem_bytes};
  params.cuda_stream = at::cuda::getCurrentCUDAStream();

  auto status = cutlass::launch_kernel_on_cluster(
      params, (void const*)kernel_ptr,
      mA, mB, mC, mD,
      mma_tiler, tiled_mma, cluster_shape,
      tma_atom_A, tma_atom_B,
      Alpha(1.0f), Beta(0.0f));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "tcgen05 inline kernel launch failed");

  return d_buffer.to(torch::kFloat16);
}

torch::Tensor optimized_matmul_tcgen05(torch::Tensor a, torch::Tensor b) {
  return run_tcgen05_variant<VariantCTA1>(a, b);
}

torch::Tensor optimized_matmul_tcgen05_cta2(torch::Tensor a, torch::Tensor b) {
  return run_tcgen05_variant<VariantCTA2>(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("optimized_matmul_tcgen05", &optimized_matmul_tcgen05);
  m.def("optimized_matmul_tcgen05_cta2", &optimized_matmul_tcgen05_cta2);
}

TORCH_LIBRARY(capstone_tcgen05, m) {
  m.def("optimized_matmul_tcgen05(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_matmul_tcgen05_cta2(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(capstone_tcgen05, CUDA, m) {
  m.impl("optimized_matmul_tcgen05", optimized_matmul_tcgen05);
  m.impl("optimized_matmul_tcgen05_cta2", optimized_matmul_tcgen05_cta2);
}
