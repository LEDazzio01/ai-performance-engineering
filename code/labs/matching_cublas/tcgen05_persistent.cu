/**
 * Stage 7: Persistent Kernel tcgen05 GEMM
 * =======================================
 * 
 * Key optimization: CTAs stay resident and process multiple tiles.
 * 
 * Instead of launching one CTA per output tile, we launch a fixed
 * number of CTAs (one per SM) that loop and process tiles until done.
 * 
 * Benefits:
 * - Amortized kernel launch overhead
 * - Better L2 locality (consecutive tiles hit same cache lines)
 * - Work stealing for load balancing
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

namespace persistent_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

constexpr int kStages = 2;

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct PersistentSharedStorage {
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

using MmaTag =
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                         UMMA::Major::K, UMMA::Major::K>;

// Global tile counter for persistent scheduling
__device__ unsigned int g_tile_idx;

template <class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB,
          class ASmemLayout, class BSmemLayout>
__global__ void __launch_bounds__(128, 1)
gemm_persistent(ATensor mA,
                BTensor mB,
                CTensor mC,
                DTensor mD,
                MmaTiler_MNK mma_tiler,
                TiledMMA tiled_mma,
                int num_tiles_m,
                int num_tiles_n,
                CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  
  extern __shared__ char shared_memory[];
  SharedStorageT& shared_storage =
      *reinterpret_cast<SharedStorageT*>(shared_memory);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  // TMEM allocation (done once per CTA, reused across tiles)
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(
        decltype(tmem_allocator)::Sm100TmemCapacityColumns,
        &shared_storage.tmem_base_ptr);
  }
  __syncthreads();
  uint32_t tmem_base = shared_storage.tmem_base_ptr;

  // Initialize barriers once
  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, 1);
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(shared_storage.tma_barrier[s], 1);
    }
  }
  __syncthreads();

  int total_tiles = num_tiles_m * num_tiles_n;
  
  // Persistent loop - each CTA processes multiple tiles
  while (true) {
    // Fetch next tile atomically
    unsigned int tile_idx;
    if (elect_one_warp && elect_one_thr) {
      tile_idx = atomicAdd(&g_tile_idx, 1);
    }
    tile_idx = __shfl_sync(0xffffffff, tile_idx, 0);
    
    if (tile_idx >= total_tiles) break;
    
    // Convert linear index to 2D coordinates
    int tile_m = tile_idx / num_tiles_n;
    int tile_n = tile_idx % num_tiles_n;
    
    auto mma_coord = make_coord(tile_m, tile_n, _);

    Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
    Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

    auto cta_mma = tiled_mma.get_slice(Int<0>{});
    Tensor tCgA = cta_mma.partition_A(gA);
    Tensor tCgB = cta_mma.partition_B(gB);
    Tensor tCgC = cta_mma.partition_C(gC);
    Tensor tCgD = cta_mma.partition_C(gD);

    Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
    tCtAcc.data() = tmem_base;

    // Setup TMA
    Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
    Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
    Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
    Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
    Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
    Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

    Tensor tCsA_0 = shared_storage.tensor_sA(0);
    Tensor tCsB_0 = shared_storage.tensor_sB(0);
    Tensor tCsA_1 = shared_storage.tensor_sA(1);
    Tensor tCsB_1 = shared_storage.tensor_sB(1);

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

    Tensor tCrA_0 = cta_mma.make_fragment_A(tCsA_0);
    Tensor tCrB_0 = cta_mma.make_fragment_B(tCsB_0);
    Tensor tCrA_1 = cta_mma.make_fragment_A(tCsA_1);
    Tensor tCrB_1 = cta_mma.make_fragment_B(tCsB_1);

    int tma_transaction_bytes =
        sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0));

    int num_k_tiles = size<3>(tCgA);
    int tma_phase[kStages] = {0, 0};
    int mma_barrier_phase = 0;

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    // Load first tile
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(
          shared_storage.tma_barrier[0], tma_transaction_bytes);
      copy(tma_atom_A.with(shared_storage.tma_barrier[0]), tAgA_0(_, 0), tAsA_0);
      copy(tma_atom_B.with(shared_storage.tma_barrier[0]), tBgB_0(_, 0), tBsB_0);
    }

    // Pipelined K-loop
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      int curr_stage = k_tile % kStages;
      int next_stage = (k_tile + 1) % kStages;

      cute::wait_barrier(shared_storage.tma_barrier[curr_stage], tma_phase[curr_stage]);
      tma_phase[curr_stage] ^= 1;

      if (k_tile + 1 < num_k_tiles) {
        if (elect_one_warp && elect_one_thr) {
          cute::set_barrier_transaction_bytes(
              shared_storage.tma_barrier[next_stage], tma_transaction_bytes);
          if (next_stage == 0) {
            copy(tma_atom_A.with(shared_storage.tma_barrier[0]), tAgA_0(_, k_tile + 1), tAsA_0);
            copy(tma_atom_B.with(shared_storage.tma_barrier[0]), tBgB_0(_, k_tile + 1), tBsB_0);
          } else {
            copy(tma_atom_A.with(shared_storage.tma_barrier[1]), tAgA_1(_, k_tile + 1), tAsA_1);
            copy(tma_atom_B.with(shared_storage.tma_barrier[1]), tBgB_1(_, k_tile + 1), tBsB_1);
          }
        }
      }

      if (elect_one_warp) {
        auto& tCrA = (curr_stage == 0) ? tCrA_0 : tCrA_1;
        auto& tCrB = (curr_stage == 0) ? tCrB_0 : tCrB_1;

        for (int k_block = 0; k_block < size<2>(tCrA_0); ++k_block) {
          gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
      }

      cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase);
      mma_barrier_phase ^= 1;
    }

    // Epilogue
    auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
    auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

    Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
    Tensor tDrC = make_fragment_like(tDgC);
    copy(tDgC, tDrC);

    Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
    Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
    Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
    copy(tiled_t2r_copy, tDtAcc, tDrAcc);

    axpby(1.0f, tDrAcc, 0.0f, tDrC);
    copy(tDrC, tDgD);
    
    __syncthreads();
  }

  // Cleanup TMEM
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(
        tmem_base,
        decltype(tmem_allocator)::Sm100TmemCapacityColumns);
  }
}

torch::Tensor run_persistent_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(1), "incompatible shapes");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "expected float16 inputs");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "expected CUDA tensors");

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_contig.size(0);

  TORCH_CHECK(m % 128 == 0 && n % 256 == 0 && k % 64 == 0,
              "Problem size must be divisible by tcgen05 tile (128x256x64)");

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::zeros({m, n}, options);
  auto d_buffer = torch::empty_like(c_buffer);

  auto tiled_mma = make_tiled_mma(MmaTag{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

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
      PersistentSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

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

  int num_tiles_m = (m + size(bM) - 1) / size(bM);
  int num_tiles_n = (n + size(bN) - 1) / size(bN);

  // Reset work counter
  unsigned int zero = 0;
  cudaMemcpyToSymbol(g_tile_idx, &zero, sizeof(unsigned int));

  // Launch with one CTA per SM (persistent)
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sms = prop.multiProcessorCount;
  
  dim3 dimBlock(128);
  dim3 dimGrid(num_sms);

  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_persistent<
      SharedStorageT,
      decltype(mA), decltype(mB),
      decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma),
      decltype(tma_atom_A), decltype(tma_atom_B),
      decltype(sA_layout), decltype(sB_layout)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  gemm_persistent<
      SharedStorageT,
      decltype(mA), decltype(mB),
      decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma),
      decltype(tma_atom_A), decltype(tma_atom_B),
      decltype(sA_layout), decltype(sB_layout)>
      <<<dimGrid, dimBlock, smem_bytes,
         at::cuda::getCurrentCUDAStream()>>>(
          mA, mB, mC, mD,
          mma_tiler, tiled_mma,
          num_tiles_m, num_tiles_n,
          tma_atom_A, tma_atom_B);
  AT_CUDA_CHECK(cudaGetLastError());

  return d_buffer.to(torch::kFloat16);
}

}  // namespace persistent_impl

torch::Tensor matmul_tcgen05_persistent(torch::Tensor a, torch::Tensor b) {
  return persistent_impl::run_persistent_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_persistent", &matmul_tcgen05_persistent);
}

