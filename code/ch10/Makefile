# Chapter 10 Makefile - Memory Pipelines and GPUDirect Storage with architecture-aware builds

include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo -Xptxas=-v
SUFFIX = $(TARGET_SUFFIX)

# Include path for common headers
INCLUDES = -I..
COMMON_HEADERS = ../common/headers/arch_detection.cuh ../common/headers/tma_helpers.cuh

# Explicit list of CUDA examples
BASE_TARGETS = baseline_cluster_group optimized_cluster_group \
               baseline_cluster_group_no_dsmem optimized_cluster_group_no_dsmem \
               optimized_cluster_group_single_cta \
               baseline_cooperative_persistent optimized_cooperative_persistent \
               baseline_double_buffered_pipeline optimized_double_buffered_pipeline \
               baseline_flash_attn_tma_micro_pipeline optimized_flash_attn_tma_micro_pipeline \
               baseline_warp_specialized_pipeline optimized_warp_specialized_pipeline \
               optimized_warp_specialized_pipeline_enhanced \
               baseline_warp_specialized_cluster_pipeline optimized_warp_specialized_cluster_pipeline \
               tma_2d_pipeline_blackwell \
               baseline_tma_multicast optimized_tma_multicast \
               baseline_pipeline_3stage optimized_pipeline_3stage \
               baseline_dsmem_reduction optimized_dsmem_reduction \
               optimized_dsmem_cluster_atomic optimized_dsmem_warp_specialized \
               baseline_warp_spec_pingpong optimized_warp_spec_pingpong

# Guard optional demos that are not shipped on every branch.
ifneq ($(wildcard optimized_cluster_group_dram_partial_cluster_sync_no_dsmem.cu),)
BASE_TARGETS += optimized_cluster_group_dram_partial_cluster_sync_no_dsmem
else
$(info Skipping optimized_cluster_group_dram_partial_cluster_sync_no_dsmem – source missing)
endif

# Grace-Blackwell (sm_121) exposes clusters but not DSMEM/tensormap. Skip DSMEM binaries to
# keep the benchmark harness from treating ptxas failures as test failures.
ifeq ($(findstring sm_121,$(ARCH)),sm_121)
DSMEM_UNSUPPORTED := 1
DSMEM_TARGETS := baseline_cluster_group optimized_cluster_group optimized_cluster_group_single_cta baseline_warp_specialized_cluster_pipeline optimized_warp_specialized_cluster_pipeline
BASE_TARGETS := $(filter-out $(DSMEM_TARGETS),$(BASE_TARGETS))
$(info Skipping DSMEM targets ($(DSMEM_TARGETS)) on $(ARCH) – DSMEM unavailable)
endif

TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS) run test-gds demo help $(BASE_TARGETS)

# Convenience targets without architecture suffix (delegate to active ARCH)
$(BASE_TARGETS):
	@$(MAKE) $@$(SUFFIX)

all: $(TARGETS)
	@echo "Building all Chapter 10 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

# Explicit build rules for each example
baseline_cluster_group$(SUFFIX): baseline_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

baseline_cluster_group_no_dsmem$(SUFFIX): baseline_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_cluster_group$(SUFFIX): optimized_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_cluster_group_no_dsmem$(SUFFIX): optimized_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_cluster_group_single_cta$(SUFFIX): optimized_cluster_group_single_cta.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_cluster_group_dram_partial_cluster_sync_no_dsmem$(SUFFIX): optimized_cluster_group_dram_partial_cluster_sync_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

baseline_cooperative_persistent$(SUFFIX): baseline_cooperative_persistent.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_cooperative_persistent$(SUFFIX): optimized_cooperative_persistent.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

baseline_double_buffered_pipeline$(SUFFIX): baseline_double_buffered_pipeline.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_double_buffered_pipeline$(SUFFIX): optimized_double_buffered_pipeline.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

baseline_warp_specialized_pipeline$(SUFFIX): baseline_warp_specialized_pipeline.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_warp_specialized_pipeline$(SUFFIX): optimized_warp_specialized_pipeline.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_warp_specialized_pipeline_enhanced$(SUFFIX): optimized_warp_specialized_pipeline_enhanced.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

baseline_warp_specialized_cluster_pipeline$(SUFFIX): baseline_warp_specialized_cluster_pipeline.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_warp_specialized_cluster_pipeline$(SUFFIX): optimized_warp_specialized_cluster_pipeline.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

baseline_flash_attn_tma_micro_pipeline$(SUFFIX): baseline_flash_attn_tma_micro_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(INCLUDES) -o $@ $<

optimized_flash_attn_tma_micro_pipeline$(SUFFIX): optimized_flash_attn_tma_micro_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(INCLUDES) -lcuda -o $@ $<

tma_2d_pipeline_blackwell$(SUFFIX): tma_2d_pipeline_blackwell.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) -lcuda -o $@ $<

baseline_tma_multicast$(SUFFIX): baseline_tma_multicast.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_tma_multicast$(SUFFIX): optimized_tma_multicast.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

baseline_pipeline_3stage$(SUFFIX): baseline_pipeline_3stage.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_pipeline_3stage$(SUFFIX): optimized_pipeline_3stage.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

baseline_dsmem_reduction$(SUFFIX): baseline_dsmem_reduction.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_dsmem_reduction$(SUFFIX): optimized_dsmem_reduction.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_dsmem_cluster_atomic$(SUFFIX): optimized_dsmem_cluster_atomic.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

optimized_dsmem_warp_specialized$(SUFFIX): optimized_dsmem_warp_specialized.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $<

baseline_warp_spec_pingpong$(SUFFIX): baseline_warp_spec_pingpong.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

optimized_warp_spec_pingpong$(SUFFIX): optimized_warp_spec_pingpong.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Legacy target name for double_buffered_pipeline (without suffix)
double_buffered_pipeline: baseline_double_buffered_pipeline$(SUFFIX)
	cp baseline_double_buffered_pipeline$(SUFFIX) double_buffered_pipeline

run: double_buffered_pipeline
	./double_buffered_pipeline

test: all
	@echo "=== Testing Memory Pipeline (CUDA) ==="
	@echo "=== Testing small matrix (16x16x16) ==="
	./baseline_double_buffered_pipeline$(SUFFIX) 16 16 16 || true
	@echo ""
	@echo "=== Testing medium matrix (128x128x128) ==="
	./baseline_double_buffered_pipeline$(SUFFIX) 128 128 128 || true
	@echo ""
	@echo "=== Testing large matrix (1024x1024x1024) ==="
	./baseline_double_buffered_pipeline$(SUFFIX) 1024 1024 1024 || true

test-gds:
	@echo "=== Testing GPUDirect Storage (Python) ==="
	python3 cufile_gds_example.py || true

demo: all
	@echo "=== Running Both Examples ==="
	@./demo_both_examples.sh || true

compare: clean
	@echo "========================================"
	@echo "Building Chapter 10 samples for both architectures..."
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

help:
	@echo "Chapter 10: Memory Pipelines and GPUDirect Storage"
	@echo ""
	@echo "Available targets:"
	@echo "  make                - Build memory pipeline (CUDA)"
	@echo "  make test           - Test memory pipeline with multiple sizes"
	@echo "  make test-gds       - Test GPUDirect Storage (Python)"
	@echo "  make demo           - Run both examples back-to-back"
	@echo "  make run            - Run memory pipeline (default size)"
	@echo "  make clean          - Remove compiled binaries"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "Architecture targets:"
	@echo "  make b200           - Build for B200 (sm_100)"
	@echo "  make gb10           - Build for GB10 (sm_121)"
	@echo "  make b300           - Build for B300 (sm_103)"
	@echo "  make gb300          - Build for GB300 (sm_103)"
	@echo ""
	@echo "Examples:"
	@echo "  Memory Pipeline:       ./baseline_double_buffered_pipeline$(SUFFIX) 512 512 512"
	@echo "  GPUDirect Storage:     python3 cufile_gds_example.py"
	@echo "  Both:                  make demo"

clean:
	rm -f $(addsuffix _sm100,$(BASE_TARGETS)) $(addsuffix _sm121,$(BASE_TARGETS)) *.o *.so *.a double_buffered_pipeline
	rm -f warp_spec_cluster_test warp_specialized_cluster_pipeline
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
