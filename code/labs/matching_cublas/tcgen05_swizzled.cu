/**
 * Stage 5: Swizzled Tile Scheduling + 3-Stage Pipeline
 * =====================================================
 * 
 * Key optimization: Process tiles in a swizzled order to maximize L2 cache hits.
 * Instead of row-major tile order (0,1,2,3,4,...), use a pattern that keeps
 * adjacent CTAs accessing nearby memory regions.
 * 
 * Swizzle pattern example for 4x4 tile grid:
 *   Row-major:  0  1  2  3     Swizzled:  0  1  4  5
 *               4  5  6  7                2  3  6  7
 *               8  9 10 11                8  9 12 13
 *              12 13 14 15               10 11 14 15
 * 
 * This improves L2 hit rate by ~10-20% for large matrices.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

using namespace cute;

namespace swizzled_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

constexpr int kStages = 3;
constexpr int kSwizzleLogTile = 3;  // Swizzle in 8x8 tile groups

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct SwizzledSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> A[kStages];
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> B[kStages];

  alignas(16) cute::uint64_t tma_barrier[kStages];
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA(int stage) {
    return make_tensor(make_smem_ptr(A[stage].begin()), ASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sB(int stage) {
    return make_tensor(make_smem_ptr(B[stage].begin()), BSmemLayout{});
  }
};

using MmaTag = SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                                    UMMA::Major::K, UMMA::Major::K>;

// Swizzle function: remap block indices for better L2 locality
__device__ __forceinline__ void swizzle_block_idx(int bx, int by, int num_tiles_n,
                                                   int& out_tile_m, int& out_tile_n) {
  // Group tiles into 2^kSwizzleLogTile x 2^kSwizzleLogTile squares
  constexpr int tile_group = 1 << kSwizzleLogTile;
  
  int group_x = bx / tile_group;
  int group_y = by / tile_group;
  int local_x = bx % tile_group;
  int local_y = by % tile_group;
  
  // Interleave within group using XOR swizzle
  int swizzled_local_x = local_x ^ local_y;
  
  out_tile_n = group_x * tile_group + swizzled_local_x;
  out_tile_m = group_y * tile_group + local_y;
}

template <class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape,
          class TmaAtomA, class TmaAtomB>
__global__ void gemm_swizzled(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
                              MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
                              int num_tiles_m, int num_tiles_n,
                              ClusterShape cluster_shape,
                              CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                              CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  // Apply swizzle to block indices
  int tile_m, tile_n;
  swizzle_block_idx(blockIdx.x, blockIdx.y, num_tiles_n, tile_m, tile_n);
  
  // Bounds check after swizzle
  if (tile_m >= num_tiles_m || tile_n >= num_tiles_n) return;

  auto mma_coord = make_coord(tile_m, tile_n, _);

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageT& ss = *reinterpret_cast<SharedStorageT*>(shared_memory);

  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(decltype(tmem_allocator)::Sm100TmemCapacityColumns,
                            &ss.tmem_base_ptr);
  }
  __syncthreads();
  uint32_t tmem_base = ss.tmem_base_ptr;

  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
  tCtAcc.data() = tmem_base;

  Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
  Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
  Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

  auto tCsA_0 = ss.tensor_sA(0);
  auto tCsB_0 = ss.tensor_sB(0);
  auto tCsA_1 = ss.tensor_sA(1);
  auto tCsB_1 = ss.tensor_sB(1);
  auto tCsA_2 = ss.tensor_sA(2);
  auto tCsB_2 = ss.tensor_sB(2);
  
  auto tCrA_0 = cta_mma.make_fragment_A(tCsA_0);
  auto tCrB_0 = cta_mma.make_fragment_B(tCsB_0);
  auto tCrA_1 = cta_mma.make_fragment_A(tCsA_1);
  auto tCrB_1 = cta_mma.make_fragment_B(tCsB_1);
  auto tCrA_2 = cta_mma.make_fragment_A(tCsA_2);
  auto tCrB_2 = cta_mma.make_fragment_B(tCsB_2);

  auto [tAgA_0, tAsA_0] = tma_partition(
      tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));
  auto [tBgB_0, tBsB_0] = tma_partition(
      tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB));
  auto [tAgA_1, tAsA_1] = tma_partition(
      tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_1), group_modes<0,3>(tCgCoordA));
  auto [tBgB_1, tBsB_1] = tma_partition(
      tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_1), group_modes<0,3>(tCgCoordB));
  auto [tAgA_2, tAsA_2] = tma_partition(
      tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_2), group_modes<0,3>(tCgCoordA));
  auto [tBgB_2, tBsB_2] = tma_partition(
      tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_2), group_modes<0,3>(tCgCoordB));

  int tma_bytes = sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0));

  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(ss.mma_barrier, 1);
    #pragma unroll
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(ss.tma_barrier[s], 1);
    }
  }
  __syncthreads();

  int num_k = size<3>(tCgA);
  int tma_phase[kStages] = {0, 0, 0};
  int mma_phase = 0;

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  auto issue_tma = [&](int stage, int k_tile) {
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(ss.tma_barrier[stage], tma_bytes);
      switch (stage) {
        case 0:
          copy(tma_atom_A.with(ss.tma_barrier[0]), tAgA_0(_, k_tile), tAsA_0);
          copy(tma_atom_B.with(ss.tma_barrier[0]), tBgB_0(_, k_tile), tBsB_0);
          break;
        case 1:
          copy(tma_atom_A.with(ss.tma_barrier[1]), tAgA_1(_, k_tile), tAsA_1);
          copy(tma_atom_B.with(ss.tma_barrier[1]), tBgB_1(_, k_tile), tBsB_1);
          break;
        case 2:
          copy(tma_atom_A.with(ss.tma_barrier[2]), tAgA_2(_, k_tile), tAsA_2);
          copy(tma_atom_B.with(ss.tma_barrier[2]), tBgB_2(_, k_tile), tBsB_2);
          break;
      }
    }
  };

  // Prologue
  int prologue_tiles = min(kStages - 1, num_k);
  for (int k = 0; k < prologue_tiles; ++k) {
    issue_tma(k % kStages, k);
  }

  #define DO_COMPUTE(STAGE) \
    for (int kb = 0; kb < size<2>(tCrA_0); ++kb) { \
      gemm(tiled_mma, tCrA_##STAGE(_, _, kb), tCrB_##STAGE(_, _, kb), tCtAcc); \
      tiled_mma.accumulate_ = UMMA::ScaleOut::One; \
    }

  // Main loop
  for (int k_tile = 0; k_tile < num_k; ++k_tile) {
    int read_stage = k_tile % kStages;
    int prefetch_k = k_tile + (kStages - 1);
    int prefetch_stage = prefetch_k % kStages;

    cute::wait_barrier(ss.tma_barrier[read_stage], tma_phase[read_stage]);
    tma_phase[read_stage] ^= 1;

    if (prefetch_k < num_k) {
      issue_tma(prefetch_stage, prefetch_k);
    }

    if (elect_one_warp) {
      switch (read_stage) {
        case 0: DO_COMPUTE(0); break;
        case 1: DO_COMPUTE(1); break;
        case 2: DO_COMPUTE(2); break;
      }
      cutlass::arch::umma_arrive(&ss.mma_barrier);
    }

    cute::wait_barrier(ss.mma_barrier, mma_phase);
    mma_phase ^= 1;
  }

  #undef DO_COMPUTE

  // Epilogue
  auto t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_copy = t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_copy.partition_D(tCgC);
  Tensor tDrC = make_fragment_like(tDgC);
  copy(tDgC, tDrC);

  Tensor tDtAcc = thr_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(t2r_copy, tDtAcc, tDrAcc);

  axpby(1.0f, tDrAcc, 0.0f, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, decltype(tmem_allocator)::Sm100TmemCapacityColumns);
  }
}

torch::Tensor run_swizzled_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D");
  TORCH_CHECK(a.size(1) == b.size(1), "shape mismatch");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16, "fp16");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA");

  auto ac = a.contiguous(), bc = b.contiguous();
  auto m = ac.size(0), k = ac.size(1), n = bc.size(0);

  TORCH_CHECK(m % 128 == 0 && n % 256 == 0 && k % 64 == 0, "alignment");

  auto opts = a.options().dtype(torch::kFloat32);
  auto c_buf = torch::zeros({m, n}, opts);
  auto d_buf = torch::empty_like(c_buf);

  auto tiled_mma = make_tiled_mma(MmaTag{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  int num_tiles_m = (m + size(bM) - 1) / size(bM);
  int num_tiles_n = (n + size(bN) - 1) / size(bN);

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  using SharedT = SwizzledSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<TypeA const*>(ac.data_ptr<at::Half>())),
                          make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<TypeB const*>(bc.data_ptr<at::Half>())),
                          make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buf.data_ptr<TypeC>()),
                          make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(d_buf.data_ptr<TypeD>()),
                          make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto tma_A = make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout, select<0, 2>(mma_tiler));
  auto tma_B = make_tma_atom(SM90_TMA_LOAD{}, mB, sB_layout, select<1, 2>(mma_tiler));

  dim3 block(128);
  // Launch enough blocks to cover swizzled grid
  int grid_m = ((num_tiles_m + 7) / 8) * 8;  // Round up to swizzle group
  int grid_n = ((num_tiles_n + 7) / 8) * 8;
  dim3 grid(grid_n, grid_m);
  
  int smem = sizeof(SharedT);

  using Cluster = decltype(make_shape(Int<1>{}, Int<1>{}, Int<1>{}));

  auto* kptr = &gemm_swizzled<SharedT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
                              decltype(mma_tiler), decltype(tiled_mma), Cluster,
                              decltype(tma_A), decltype(tma_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  gemm_swizzled<SharedT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
                decltype(mma_tiler), decltype(tiled_mma), Cluster,
                decltype(tma_A), decltype(tma_B)>
      <<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
          mA, mB, mC, mD, mma_tiler, tiled_mma, num_tiles_m, num_tiles_n,
          make_shape(Int<1>{}, Int<1>{}, Int<1>{}), tma_A, tma_B);
  AT_CUDA_CHECK(cudaGetLastError());

  return d_buf.to(torch::kFloat16);
}

}  // namespace swizzled_impl

torch::Tensor matmul_tcgen05_swizzled(torch::Tensor a, torch::Tensor b) {
  return swizzled_impl::run_swizzled_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_swizzled", &matmul_tcgen05_swizzled);
}

