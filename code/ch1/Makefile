# Makefile for Chapter 1 - Performance Basics Examples
# Builds optimized CUDA examples demonstrating profiling improvements

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_100 --expt-relaxed-constexpr -DCUDA_VERSION=13.0
NVCC_LIBS = -lcublas

# Targets
TARGETS = batched_gemm_example

.PHONY: all clean test

all: $(TARGETS)
	@echo "Building all Chapter 1 examples with sm_100 architecture (CUDA 13.0)"
	@echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"

batched_gemm_example: batched_gemm_example.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

test: all
	@echo "Testing batched GEMM optimization..."
	./batched_gemm_example
	@echo ""
	@echo "Testing PyTorch optimizations..."
	cd .. && PYTHONPATH=.:$$PYTHONPATH python3 ch1/performance_basics_optimized.py

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
	@echo "Cleaned ch1 binaries and profiling artifacts"

