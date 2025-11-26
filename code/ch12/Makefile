# Chapter 12 Makefile - CUDA graph and dynamic parallelism examples

include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS)
NVCC_LIBS = -lcuda $(CUDA_NVTX_LDFLAGS)
SUFFIX = $(TARGET_SUFFIX)

BASE_TARGETS = baseline_work_queue optimized_work_queue baseline_cuda_graphs optimized_cuda_graphs baseline_cuda_graphs_conditional optimized_cuda_graphs_conditional baseline_cuda_graphs_conditional_enhanced optimized_cuda_graphs_conditional_enhanced baseline_graph_conditional_runtime optimized_graph_conditional_runtime baseline_kernel_fusion optimized_kernel_fusion baseline_graph_bandwidth optimized_graph_bandwidth baseline_uneven_static optimized_uneven_static baseline_uneven_partition optimized_uneven_partition baseline_dynamic_parallelism_host optimized_dynamic_parallelism_host baseline_dynamic_parallelism_device optimized_dynamic_parallelism_device
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

$(TARGETS): $(CUDA_NVTX_DEPS)

.PHONY: all clean compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 12 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

baseline_work_queue$(SUFFIX): baseline_work_queue.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_work_queue$(SUFFIX): optimized_work_queue.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_uneven_static$(SUFFIX): baseline_uneven_static.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_uneven_static$(SUFFIX): optimized_uneven_static.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS)

baseline_uneven_partition$(SUFFIX): baseline_uneven_partition.cu uneven_partition_common.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ baseline_uneven_partition.cu $(NVCC_LIBS)

optimized_uneven_partition$(SUFFIX): optimized_uneven_partition.cu uneven_partition_common.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ optimized_uneven_partition.cu $(NVCC_LIBS)

baseline_dynamic_parallelism_host$(SUFFIX): baseline_dynamic_parallelism_host.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_dynamic_parallelism_host$(SUFFIX): optimized_dynamic_parallelism_host.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS)

baseline_dynamic_parallelism_device$(SUFFIX): baseline_dynamic_parallelism_device.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS)

optimized_dynamic_parallelism_device$(SUFFIX): optimized_dynamic_parallelism_device.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS)

baseline_cuda_graphs$(SUFFIX): baseline_cuda_graphs.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_cuda_graphs$(SUFFIX): optimized_cuda_graphs.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_cuda_graphs_conditional$(SUFFIX): baseline_cuda_graphs_conditional.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_cuda_graphs_conditional$(SUFFIX): optimized_cuda_graphs_conditional.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_cuda_graphs_conditional_enhanced$(SUFFIX): baseline_cuda_graphs_conditional_enhanced.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_cuda_graphs_conditional_enhanced$(SUFFIX): optimized_cuda_graphs_conditional_enhanced.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_graph_conditional_runtime$(SUFFIX): baseline_graph_conditional_runtime.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

optimized_graph_conditional_runtime$(SUFFIX): optimized_graph_conditional_runtime.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

baseline_kernel_fusion$(SUFFIX): baseline_kernel_fusion.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $@ $< $(NVCC_LIBS)

optimized_kernel_fusion$(SUFFIX): optimized_kernel_fusion.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $@ $< $(NVCC_LIBS)

baseline_graph_bandwidth$(SUFFIX): baseline_graph_bandwidth.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $@ $< $(NVCC_LIBS)

optimized_graph_bandwidth$(SUFFIX): optimized_graph_bandwidth.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $@ $< $(NVCC_LIBS)

compare: clean
	@echo "========================================"
	@echo "Building Chapter 12 samples for both architectures..."
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

profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	@for t in $(TARGETS); do \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$${t}_$(ARCH) ./$$t; \
	done

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	@for t in $(TARGETS); do \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$${t}_$(ARCH) ./$$t; \
	done

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for t in $(TARGETS); do \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$${t}_$(ARCH) ./$$t; \
	done

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	@for t in $(TARGETS); do \
		perf record -g -o perf.data_$(ARCH)_$$t ./$$t; \
		perf report -i perf.data_$(ARCH)_$$t; \
	done

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@for t in $(TARGETS); do \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$${t}_$(ARCH) ./$$t; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$${t}_$(ARCH) ./$$t; \
		echo "3. Memory profiling..."; \
		nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$${t}_$(ARCH) ./$$t; \
		echo "4. HTA analysis..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$${t}_$(ARCH) ./$$t; \
	done

clean:
	rm -f $(addsuffix _sm100,$(BASE_TARGETS)) $(addsuffix _sm121,$(BASE_TARGETS)) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
