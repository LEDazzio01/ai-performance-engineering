include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS)
SUFFIX = $(TARGET_SUFFIX)

# Base targets
BASE_TARGETS = basic_streams stream_ordered_allocator warp_specialized_pipeline_multistream warp_specialized_two_pipelines_multistream
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

# Default target
all: $(TARGETS)
	@echo "Building all examples with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

# Pattern rule for building targets with architecture suffix
%$(SUFFIX): %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Special rule for warp_specialized_two_pipelines_multistream (uses _driver.cu)
warp_specialized_two_pipelines_multistream$(SUFFIX): warp_specialized_two_pipelines_multistream_driver.cu warp_specialized_kernel_two_pipelines_multistream.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Architecture-specific targets
blackwell: ARCH=sm_100
blackwell: clean all
	@echo "✓ Built for Blackwell B200/B300 (discrete GPU)"

grace-blackwell: ARCH=sm_121
grace-blackwell: clean all
	@echo "✓ Built for Grace-Blackwell GB10 (integrated superchip)"

# Build BOTH versions for comparison benchmarking
compare: clean
	@echo "========================================"
	@echo "Building BOTH architectures for comparison..."
	@echo "========================================"
	@for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
		if [ "$${arch}" = "sm_100" ]; then \
			echo "✓ Blackwell B200 binaries built with _sm100 suffix"; \
		else \
			echo "✓ Grace-Blackwell GB10 binaries built with _sm121 suffix"; \
		fi; \
		echo ""; \
	done
	@echo ""
	@echo "========================================"
	@echo "Ready to benchmark! Run both versions:"
	@echo "  ./basic_streams_sm100  (discrete Blackwell)"
	@echo "  ./basic_streams_sm121  (Grace-Blackwell superchip)"
	@echo "========================================"

# Enhanced profiling targets
profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$$target_$(ARCH) ./$$target; \
	done

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$$target_$(ARCH) ./$$target; \
	done

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$$target_$(ARCH) ./$$target; \
	done

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		perf record -g -o perf.data_$$target_$(ARCH) ./$$target; \
		perf report -i perf.data_$$target_$(ARCH); \
	done

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@for target in $(TARGETS); do \
		echo "Comprehensive profiling of $$target..."; \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$$target_$(ARCH) ./$$target; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$$target_$(ARCH) ./$$target; \
		echo "3. Memory profiling..."; \
		nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$$target_$(ARCH) ./$$target; \
		echo "4. HTA analysis..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$$target_$(ARCH) ./$$target; \
	done

clean:
	rm -f *_sm100 *_sm121 *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

# Run benchmarks comparing both architectures
benchmark-compare: compare
	@echo ""
	@echo "========================================"
	@echo "Running Performance Comparison..."
	@echo "========================================"
	@echo ""
	@echo "--- Blackwell B200 (discrete GPU) ---"
	@./basic_streams_sm100 | tee benchmark_sm100.log
	@echo ""
	@echo "--- Grace-Blackwell GB10 (superchip) ---"
	@./basic_streams_sm121 | tee benchmark_sm121.log
	@echo ""
	@echo "========================================"
	@echo "Benchmark logs saved:"
	@echo "  benchmark_sm100.log - Discrete Blackwell"
	@echo "  benchmark_sm121.log - Grace-Blackwell superchip"
	@echo "========================================"

.PHONY: all blackwell grace-blackwell compare benchmark-compare profile-nsys profile-ncu profile-hta profile-perf profile-all clean
