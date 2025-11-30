# Chapter 19 Makefile - Memory and precision optimizations with architecture-aware builds

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo
SUFFIX = $(TARGET_SUFFIX)

BASE_TARGETS = baseline_fp4_hardware_kernel optimized_fp4_hardware_kernel fp8_hardware_kernel baseline_kv_prefetch_overlap optimized_kv_prefetch_overlap
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS)

all: $(TARGETS)
	@echo "Building all Chapter 19 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

baseline_fp4_hardware_kernel$(SUFFIX): baseline_fp4_hardware_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

optimized_fp4_hardware_kernel$(SUFFIX): optimized_fp4_hardware_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -lcublasLt -lcublas

fp8_hardware_kernel$(SUFFIX): fp8_hardware_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

baseline_kv_prefetch_overlap$(SUFFIX): baseline_kv_prefetch_overlap.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

optimized_kv_prefetch_overlap$(SUFFIX): optimized_kv_prefetch_overlap.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

test: all
	@echo "Testing Chapter 19 examples..."
	@for exe in $(TARGETS); do \
		if [ -x "$$exe" ]; then \
			echo "Testing $$exe..."; \
			timeout 10s ./$$exe >/dev/null 2>&1 && echo "✓ $$exe runs successfully" || echo "✗ $$exe failed to run"; \
		fi; \
	done

compare: clean
	@echo "========================================"
	@echo "Building Chapter 19 samples for both architectures..."
	@echo "========================================"
	@for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done

b200: ARCH=sm_100
b200: clean all

gb10: ARCH=sm_121
gb10: clean all

b300: ARCH=sm_103
b300: clean all

gb300: ARCH=sm_103
gb300: clean all

profile-nsys: all
	@echo "Nsight Systems timeline profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o nsys_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-ncu: all
	@echo "Nsight Compute kernel profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency -o ncu_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "HTA profiling failed for $$exe"; \
	done

profile-perf: all
	@echo "Perf system-level profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe with perf..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || echo "perf failed for $$exe"; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

profile-all: all
	@echo "Comprehensive profiling with all tools..."
	@for exe in $(TARGETS); do \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency -o comprehensive_kernel_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "3. Perf system-level analysis..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || true; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

clean:
	rm -f $(addsuffix _sm100,$(BASE_TARGETS)) $(addsuffix _sm103,$(BASE_TARGETS)) $(addsuffix _sm121,$(BASE_TARGETS)) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
	rm -f __pycache__/*.pyc
	rm -rf validation_results/
	@echo "Cleaned ch19 binaries and profiling artifacts"
