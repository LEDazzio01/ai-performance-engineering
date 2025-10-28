# Architecture switching support
# Target architecture: Blackwell B200/B300 (SM100)
ARCH ?= sm_100

# CUDA configuration
CUDA_VERSION = 13.0
NVCC = nvcc
# CUTLASS 3.5 for Blackwell tensor cores
CUTLASS_PATH ?= $(shell pwd)/../external/cutlass
ifneq ($(wildcard $(CUTLASS_PATH)/include),)
CUTLASS_INCLUDE = -I$(CUTLASS_PATH)/include -DUSE_CUTLASS
else
CUTLASS_INCLUDE =
endif
# Use -gencode to embed both cubin (sm_100) and PTX (compute_100) for forward compatibility
NVCC_FLAGS = -O3 -std=c++17 -gencode arch=compute_100,code=[sm_100,compute_100] --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION) -Xcompiler -fPIC -lcuda $(CUTLASS_INCLUDE)

# Source files
SOURCES = double_buffered_pipeline.cu warp_specialized_pipeline.cu cooperative_persistent_kernel.cu
TARGETS = $(SOURCES:.cu=)

# Blackwell-only examples are built only when ARCH=sm_100
ifeq ($(ARCH),sm_100)
SOURCES += tma_2d_pipeline_blackwell.cu cluster_group_blackwell.cu tcgen05_blackwell.cu
TARGETS += tma_2d_pipeline_blackwell cluster_group_blackwell tcgen05_blackwell
endif

# Default target
all: $(TARGETS)
	@echo "Building with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"

# Build individual targets
double_buffered_pipeline: double_buffered_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

warp_specialized_pipeline: warp_specialized_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cooperative_persistent_kernel: cooperative_persistent_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Blackwell-only builds
tma_2d_pipeline_blackwell: tma_2d_pipeline_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

cluster_group_blackwell: cluster_group_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

tcgen05_blackwell: tcgen05_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Architecture-specific targets
blackwell: ARCH=sm_100
blackwell: $(TARGETS)
	@echo "Built for Blackwell B200/B300"

# Enhanced profiling targets
profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$(ARCH)_$$target ./$$target; \
	done

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$(ARCH)_$$target ./$$target; \
	done

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$(ARCH)_$$target ./$$target; \
	done

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		perf record -g -o perf.data_$(ARCH)_$$target ./$$target; \
		perf report -i perf.data_$(ARCH)_$$target; \
	done

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@for target in $(TARGETS); do \
		echo "Comprehensive profiling of $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$(ARCH)_$$target ./$$target; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$(ARCH)_$$target ./$$target; \
		nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$(ARCH)_$$target ./$$target; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$(ARCH)_$$target ./$$target; \
	done

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean $(TARGETS)
