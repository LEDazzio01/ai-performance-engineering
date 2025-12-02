# Chapter 10 Makefile - Memory Pipelines and GPUDirect Storage
# Auto-builds all *.cu files - no manual BASE_TARGETS list needed

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo -Xptxas=-v
SUFFIX = $(TARGET_SUFFIX)
INCLUDES = -I..
COMMON_HEADERS = ../core/common/headers/arch_detection.cuh ../core/common/headers/tma_helpers.cuh

# Auto-detect all .cu files and derive targets
CUDA_SOURCES := $(wildcard *.cu)
# Filter out files that require external build systems (PyTorch extensions)
EXTENSION_SOURCES := $(wildcard *_extension.cu) matmul_tcgen05.cu tcgen05_blackwell.cu
STANDALONE_SOURCES := $(filter-out $(EXTENSION_SOURCES),$(CUDA_SOURCES))
BASE_TARGETS := $(STANDALONE_SOURCES:.cu=)
TARGETS := $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all run test-gds demo help

all: $(TARGETS)
	@echo "Building all Chapter 10 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

# Default pattern rule for most .cu files
%$(SUFFIX): %.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Special rules for files needing -rdc=true (cluster/DSMEM support)
baseline_cluster_group$(SUFFIX): baseline_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_cluster_group$(SUFFIX): optimized_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

baseline_cluster_group_no_dsmem$(SUFFIX): baseline_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_cluster_group_no_dsmem$(SUFFIX): optimized_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_cluster_group_single_cta$(SUFFIX): optimized_cluster_group_single_cta.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_tma_multicast$(SUFFIX): optimized_tma_multicast.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_dsmem_reduction$(SUFFIX): optimized_dsmem_reduction.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_dsmem_cluster_atomic$(SUFFIX): optimized_dsmem_cluster_atomic.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_dsmem_warp_specialized$(SUFFIX): optimized_dsmem_warp_specialized.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

# Special rules for files needing C++20 or -lcuda
baseline_flash_attn_tma_micro_pipeline$(SUFFIX): baseline_flash_attn_tma_micro_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(INCLUDES) -o $@ $<

optimized_flash_attn_tma_micro_pipeline$(SUFFIX): optimized_flash_attn_tma_micro_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(INCLUDES) -lcuda -o $@ $<

tma_2d_pipeline_blackwell$(SUFFIX): tma_2d_pipeline_blackwell.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) -lcuda -o $@ $<

# Architecture-specific targets
b200: ARCH=sm_100
b200: clean all

gb10: ARCH=sm_121
gb10: clean all

b300: ARCH=sm_103
b300: clean all

gb300: ARCH=sm_103
gb300: clean all

compare: clean
	@for arch in $(ARCH_LIST); do \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done

test: all
	@echo "=== Testing Memory Pipeline (CUDA) ==="
	./baseline_double_buffered_pipeline$(SUFFIX) 128 128 128 || true

test-gds:
	@echo "=== Testing GPUDirect Storage (Python) ==="
	python3 cufile_gds_example.py || true

profile-nsys: all
	@for exe in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx -o nsys_$$exe ./$$exe 2>/dev/null || true; done

profile-ncu: all
	@for exe in $(TARGETS); do ncu --metrics achieved_occupancy -o ncu_$$exe ./$$exe 2>/dev/null || true; done

profile-hta: all
	@for exe in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx,nccl -o hta_$$exe ./$$exe 2>/dev/null || true; done

profile-perf: all
	@for exe in $(TARGETS); do perf record -g -o perf_$$exe ./$$exe || true; done

profile-all: profile-nsys profile-ncu profile-hta

help:
	@echo "Chapter 10: Memory Pipelines and GPUDirect Storage"
	@echo ""
	@echo "Targets: make all | make test | make clean | make b200 | make b300"

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_* ncu_* hta_* perf_* comprehensive_*
