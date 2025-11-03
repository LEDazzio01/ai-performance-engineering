# Makefile for Chapter 1 - Performance Basics Examples
# Builds optimized CUDA examples demonstrating profiling improvements.

include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS)
NVCC_LIBS = -lcublas
SUFFIX = $(TARGET_SUFFIX)

TARGETS = batched_gemm_example$(SUFFIX) arithmetic_intensity_demo$(SUFFIX)

.PHONY: all clean test compare blackwell grace-blackwell

all: $(TARGETS)
	@echo "Building Chapter 1 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

batched_gemm_example$(SUFFIX): batched_gemm_example.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

arithmetic_intensity_demo$(SUFFIX): arithmetic_intensity_demo.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

test: all
	@echo "Testing batched GEMM optimization..."
	./batched_gemm_example$(SUFFIX)
	@echo ""
	@echo "Testing PyTorch optimizations..."
	cd .. && PYTHONPATH=.:$$PYTHONPATH python3 ch1/performance_basics_optimized.py

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

blackwell: ARCH=sm_100
blackwell: clean all

grace-blackwell: ARCH=sm_121
grace-blackwell: clean all

clean:
	rm -f batched_gemm_example_sm100 batched_gemm_example_sm121 *.o *.so *.a
	rm -f arithmetic_intensity_demo_sm100 arithmetic_intensity_demo_sm121
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
	@echo "Cleaned ch1 binaries and profiling artifacts"
