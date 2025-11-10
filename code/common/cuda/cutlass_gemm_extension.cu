// CUTLASS GEMM PyTorch extension for Chapter 14 benchmarks.
//
// Implements a thin wrapper around cutlass::gemm::device::Gemm to provide
// real CUTLASS kernels accessible from Python benchmarks. The extension
// accepts row-major FP16 matrices and produces FP16 output while
// accumulating in FP32 for numerical stability.

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/arch/arch.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(status));             \
        }                                                                     \
    } while (0)

namespace {

using ElementInput = cutlass::half_t;
using LayoutInput = cutlass::layout::RowMajor;
using ElementOutput = cutlass::half_t;
using LayoutOutput = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute = float;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInput,
    LayoutInput,
    ElementInput,
    LayoutInput,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

void validate_inputs(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be on CUDA device");
    TORCH_CHECK(B.is_cuda(), "Input B must be on CUDA device");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "Input B must be float16");
    TORCH_CHECK(A.dim() == 2, "Input A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input B must be 2D");
    TORCH_CHECK(
        A.size(1) == B.size(0),
        "Inner dimensions must match for GEMM (A: ",
        A.sizes(),
        ", B: ",
        B.sizes(),
        ")"
    );
}

}  // namespace

torch::Tensor cutlass_gemm_fp16(const torch::Tensor& A, const torch::Tensor& B) {
    validate_inputs(A, B);

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(1);

    auto C = torch::empty({m, n}, A.options());

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    ElementInput const* ptr_A = reinterpret_cast<ElementInput const*>(A.data_ptr<at::Half>());
    ElementInput const* ptr_B = reinterpret_cast<ElementInput const*>(B.data_ptr<at::Half>());
    ElementOutput* ptr_C = reinterpret_cast<ElementOutput*>(C.data_ptr<at::Half>());

    int lda = static_cast<int>(A.stride(0));
    int ldb = static_cast<int>(B.stride(0));
    int ldc = static_cast<int>(C.stride(0));

    ElementCompute alpha = 1.0f;
    ElementCompute beta = 0.0f;

    typename Gemm::Arguments args(
        problem_size,
        {ptr_A, lda},
        {ptr_B, ldb},
        {ptr_C, ldc},
        {ptr_C, ldc},
        {alpha, beta}
    );

    Gemm gemm_op;
    auto support = gemm_op.can_implement(args);
    TORCH_CHECK(support == cutlass::Status::kSuccess,
                "CUTLASS arguments unsupported: ",
                cutlassGetStatusString(support));
    auto status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        cudaError_t cuda_status = cudaGetLastError();
        std::string error_msg = cutlassGetStatusString(status);
        if (cuda_status != cudaSuccess) {
            error_msg += std::string(" | CUDA: ") + cudaGetErrorString(cuda_status);
        }
        TORCH_CHECK(false, "CUTLASS GEMM failed: ", error_msg);
    }
    CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cutlass_gemm_fp16",
        &cutlass_gemm_fp16,
        "CUTLASS GEMM (FP16 input/output, FP32 accumulate)"
    );
}
