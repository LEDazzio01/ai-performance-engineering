# Chapter 12 Makefile - CUDA graph and dynamic parallelism examples

include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS)
NVCC_LIBS = -lcuda
SUFFIX = $(TARGET_SUFFIX)

BASE_TARGETS = atomic_work_queue cuda_graphs dynamic_parallelism cuda_graphs_conditional cuda_graphs_conditional_enhanced
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean compare blackwell grace-blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 12 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

atomic_work_queue$(SUFFIX): atomic_work_queue.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

cuda_graphs$(SUFFIX): cuda_graphs.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

cuda_graphs_conditional$(SUFFIX): cuda_graphs_conditional.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

cuda_graphs_conditional_enhanced$(SUFFIX): cuda_graphs_conditional_enhanced.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

dynamic_parallelism$(SUFFIX): dynamic_parallelism.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS)

compare: clean
	@echo "========================================"
	@echo "Building Chapter 12 samples for both architectures..."
	@echo "========================================"
	@for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done

blackwell: ARCH=sm_100
blackwell: clean all

grace-blackwell: ARCH=sm_121
grace-blackwell: clean all

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
