# Chapter 7 Makefile - Memory optimization examples with dual-arch support

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS)
SUFFIX = $(TARGET_SUFFIX)

CUDA_SOURCES = $(wildcard *.cu)
BASE_TARGETS = $(CUDA_SOURCES:.cu=)

TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: baseline_hbmcopy$(SUFFIX) optimized_hbmcopy$(SUFFIX) baseline_hbmpeak$(SUFFIX) optimized_hbmpeak$(SUFFIX)
.PHONY: optimized_scalar_copy$(SUFFIX) optimized_uncoalesced_copy$(SUFFIX) optimized_transpose$(SUFFIX) optimized_matmul_cuda$(SUFFIX)

baseline_hbmcopy$(SUFFIX): baseline_hbm_copy$(SUFFIX)
	@ln -sf $< $@

optimized_hbmcopy$(SUFFIX): optimized_hbm_copy$(SUFFIX)
	@ln -sf $< $@

baseline_hbmpeak$(SUFFIX): baseline_hbm_peak$(SUFFIX)
	@ln -sf $< $@

optimized_hbmpeak$(SUFFIX): optimized_hbm_peak$(SUFFIX)
	@ln -sf $< $@

# Aliases requested by the benchmark harness
optimized_scalar_copy$(SUFFIX): optimized_copy_vectorized$(SUFFIX)
	@ln -sf $< $@

optimized_uncoalesced_copy$(SUFFIX): optimized_copy_coalesced$(SUFFIX)
	@ln -sf $< $@

optimized_transpose$(SUFFIX): optimized_transpose_padded$(SUFFIX)
	@ln -sf $< $@

optimized_matmul_cuda$(SUFFIX): optimized_matmul_tiled$(SUFFIX)
	@ln -sf $< $@

.PHONY: all clean compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 7 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

$(TARGETS): %$(SUFFIX): %.cu
	@echo "Compiling $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

compare: clean
	@echo "========================================"
	@echo "Building Chapter 7 samples for both architectures..."
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
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o hta_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
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
	rm -f $(addsuffix _sm100,$(BASE_TARGETS)) $(addsuffix _sm121,$(BASE_TARGETS)) \
		baseline_hbmcopy$(SUFFIX) optimized_hbmcopy$(SUFFIX) baseline_hbmpeak$(SUFFIX) optimized_hbmpeak$(SUFFIX) \
		optimized_scalar_copy$(SUFFIX) optimized_uncoalesced_copy$(SUFFIX) optimized_transpose$(SUFFIX) optimized_matmul_cuda$(SUFFIX) \
		*.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
