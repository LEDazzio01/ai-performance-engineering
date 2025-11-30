# Chapter 6 Makefile - CUDA basics with architecture-aware builds

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo -Xptxas=-v $(CUDA_NVTX_CFLAGS)
NVCC_LIBS = -lcuda $(CUDA_NVTX_LDFLAGS)
SUFFIX = $(TARGET_SUFFIX)

# Explicit list of CUDA examples
BASE_TARGETS = baseline_add optimized_add_parallel baseline_add_tensors optimized_add_tensors \
               baseline_bank_conflicts optimized_bank_conflicts \
               optimized_ilp optimized_ilp_low_occupancy_vec4 optimized_ilp_extreme_low_occupancy_vec4 \
               baseline_launch_bounds optimized_launch_bounds \
               cuda13_demo_runner
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

$(TARGETS): $(CUDA_NVTX_DEPS)

.PHONY: all clean test compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS)

all: $(TARGETS)
	@echo "Building all Chapter 6 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

# Explicit build rules for each example
baseline_add$(SUFFIX): baseline_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_add_parallel$(SUFFIX): optimized_add_parallel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_add_tensors$(SUFFIX): baseline_add_tensors.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_add_tensors$(SUFFIX): optimized_add_tensors.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_coalescing_uncoalesced$(SUFFIX): baseline_coalescing_uncoalesced.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_coalescing$(SUFFIX): optimized_coalescing.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_bank_conflicts$(SUFFIX): baseline_bank_conflicts.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_bank_conflicts$(SUFFIX): optimized_bank_conflicts.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_ilp$(SUFFIX): baseline_ilp.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_ilp$(SUFFIX): optimized_ilp.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_ilp_low_occupancy_vec4$(SUFFIX): optimized_ilp_low_occupancy_vec4.cu optimized_ilp_low_occupancy_vec4_impl.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ optimized_ilp_low_occupancy_vec4.cu $(NVCC_LIBS)

optimized_ilp_extreme_low_occupancy_vec4$(SUFFIX): optimized_ilp_extreme_low_occupancy_vec4.cu optimized_ilp_low_occupancy_vec4_impl.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ optimized_ilp_extreme_low_occupancy_vec4.cu $(NVCC_LIBS)

baseline_launch_bounds$(SUFFIX): baseline_launch_bounds_gmem.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_launch_bounds$(SUFFIX): optimized_launch_bounds_gmem.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

cuda13_demo_runner$(SUFFIX): cuda13_demo_runner.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

test: all
	@echo "Testing all Chapter 6 examples..."
	@for exe in $(TARGETS); do \
		echo "Testing $$exe..."; \
		timeout 10s ./$$exe >/dev/null 2>&1 && echo "✓ $$exe runs successfully" || echo "✗ $$exe failed to run"; \
	done

compare: clean
	@echo "========================================"
	@echo "Building Chapter 6 samples for both architectures..."
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
	rm -f $(addsuffix _sm100,$(BASE_TARGETS)) $(addsuffix _sm121,$(BASE_TARGETS)) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
	rm -f test_bank test_baseline test_opt
