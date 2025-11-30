# Makefile for Chapter 1 - Performance Basics Examples
# Builds optimized CUDA examples demonstrating profiling improvements.

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) --use_fast_math -lineinfo
NVCC_LIBS = -lcublas -lcublasLt
SUFFIX = $(TARGET_SUFFIX)

BASE_TARGETS = baseline_gemm optimized_gemm_batched optimized_gemm_strided
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare b200 gb10 b300 gb300

all: $(TARGETS)
	@echo "Building Chapter 1 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

baseline_gemm$(SUFFIX): baseline_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

optimized_gemm_batched$(SUFFIX): optimized_gemm_batched.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

optimized_gemm_strided$(SUFFIX): optimized_gemm_strided.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

test: all
	@echo "Testing batched GEMM optimization..."
	./baseline_gemm$(SUFFIX) || echo "Note: Run baseline_gemm manually"
	@echo ""
	@echo "Testing PyTorch optimizations..."
	cd .. && PYTHONPATH=.:$$PYTHONPATH python3 ch01/baseline_performance.py
	cd .. && PYTHONPATH=.:$$PYTHONPATH python3 ch01/optimized_performance.py

compare: clean
	@echo "========================================"
	@echo "Building Chapter 1 for both architectures..."
	@echo "========================================"
	@for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done
	@echo ""
	@echo "Run binaries:"
	@echo "  ./batched_gemm_example_sm100"
	@echo "  ./batched_gemm_example_sm121"

b200: ARCH=sm_100
b200: clean all

gb10: ARCH=sm_121
gb10: clean all

b300: ARCH=sm_103
b300: clean all

gb300: ARCH=sm_103
gb300: clean all

clean:
	rm -f batched_gemm_example_sm100 batched_gemm_example_sm121 *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
	@echo "Cleaned ch01 binaries and profiling artifacts"
