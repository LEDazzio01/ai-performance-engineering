#ifndef CUTLASS_ARCH_MMA_SM121_ENABLED
#define CUTLASS_ARCH_MMA_SM121_ENABLED 1
#endif
#ifndef CUTLASS_ARCH_MMA_SM121A_ENABLED
#define CUTLASS_ARCH_MMA_SM121A_ENABLED 1
#endif
#ifndef CUTLASS_ARCH_MMA_SM121F_ENABLED
#define CUTLASS_ARCH_MMA_SM121F_ENABLED 1
#endif

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "../third_party/cutlass/examples/common/helper.h"

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "CUTLASS was not built with tcgen05 (SM100) support enabled."
#endif

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using MmaTileShape_MNK = Shape<_64, _64, _32>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementC,
    LayoutC,
    AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

template <class Element>
void initialize_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed = 2025) {
  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, scope_max, scope_min, 0);
}

int main() {
  constexpr int M = 512;
  constexpr int N = 512;
  constexpr int K = 512;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

  block_A.reset(M * K);
  block_B.reset(K * N);
  block_C.reset(M * N);
  block_D.reset(M * N);

  initialize_block(block_A, 42);
  initialize_block(block_B, 43);
  initialize_block(block_C, 44);

  Gemm gemm;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{1.0f, 0.0f}, block_C.get(), stride_C, block_D.get(), stride_D}};

  args.scheduler.max_swizzle_size = 0;

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  std::cout << "Shared storage requirement: " << GemmKernel::SharedStorageSize
            << " bytes" << std::endl;
  CUTLASS_CHECK(gemm.can_implement(args));
  auto init_status = gemm.initialize(args, workspace.get());
  if (init_status != cutlass::Status::kSuccess) {
    std::cerr << "gemm.initialize failed: " << cutlassGetStatusString(init_status) << "\n";
    return -1;
  }
  auto run_status = gemm.run();
  if (run_status != cutlass::Status::kSuccess) {
    cudaError_t err = cudaGetLastError();
    std::cerr << "gemm.run failed: " << cutlassGetStatusString(run_status)
              << " (cuda: " << (err == cudaSuccess ? "cudaSuccess" : cudaGetErrorString(err))
              << ")\n";
    return -1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "tcgen05 probe completed for " << M << "x" << N << "x" << K << " GEMM\n";
  return 0;
}
