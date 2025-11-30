# Chapter 2 Makefile (includes Blackwell and Grace-Blackwell NVLink demos)

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo -Xptxas -O3
NVCC_DEBUG_FLAGS = $(NVCC_FLAGS) -G
SUFFIX = $(TARGET_SUFFIX)

BASE_TARGETS = baseline_memory_transfer optimized_memory_transfer_zero_copy optimized_memory_transfer_nvlink cpu_gpu_grace_blackwell_coherency
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean compare profile-hta profile-perf profile-all b200 gb10 b300 gb300 debug $(TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 2 targets with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

baseline_memory_transfer$(SUFFIX): baseline_memory_transfer.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

optimized_memory_transfer_zero_copy$(SUFFIX): optimized_memory_transfer_zero_copy.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

optimized_memory_transfer_nvlink$(SUFFIX): optimized_memory_transfer_nvlink.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cpu_gpu_grace_blackwell_coherency$(SUFFIX): cpu_gpu_grace_blackwell_coherency.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

compare: clean
	@echo "========================================"
	@echo "Building Chapter 2 for both architectures..."
	@echo "========================================"
	@for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done
	@echo ""
	@echo "Run binaries:"
	@echo "  ./optimized_memory_transfer_nvlink_sm100"
	@echo "  ./optimized_memory_transfer_nvlink_sm121"

b200: ARCH=sm_100
b200: clean all

gb10: ARCH=sm_121
gb10: clean all

b300: ARCH=sm_103
b300: clean all

gb300: ARCH=sm_103
gb300: clean all

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(TARGETS); do \
		if [ -x "$$exe" ]; then \
			echo "Profiling $$exe..."; \
			nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl \
				-o hta_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || \
				echo "HTA profiling failed for $$exe"; \
		fi; \
	done

profile-perf: all
	@echo "Perf system-level profiling..."
	@for exe in $(TARGETS); do \
		if [ -x "$$exe" ]; then \
			echo "Profiling $$exe with perf..."; \
			perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe 2>/dev/null || \
				echo "perf failed for $$exe"; \
			perf report -i perf.data_$(ARCH)_$$exe 2>/dev/null || true; \
		fi; \
	done

profile-all: all
	@echo "Comprehensive profiling with all tools..."
	@for exe in $(TARGETS); do \
		if [ -x "$$exe" ]; then \
			echo "1. Nsight Systems timeline (HTA)..."; \
			nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl \
				-o comprehensive_hta_$$exe_$(ARCH) ./$$exe 2>/dev/null || true; \
			echo "2. Perf system-level analysis..."; \
			perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe 2>/dev/null || true; \
			perf report -i perf.data_$(ARCH)_$$exe 2>/dev/null || true; \
		fi; \
	done

clean:
	rm -f optimized_memory_transfer_nvlink_sm100 optimized_memory_transfer_nvlink_sm121
	rm -f cpu_gpu_grace_blackwell_coherency_sm100 cpu_gpu_grace_blackwell_coherency_sm121
	rm -f cpu_gpu_zero_copy_coherent_sm100 cpu_gpu_zero_copy_coherent_sm121
	rm -f *.o *.so *.a
	rm -f nsys_profile_* hta_profile_* perf.data_* comprehensive_*
debug:
	@echo "Building Chapter 2 debug targets with lineinfo and -G"
	$(NVCC) $(NVCC_DEBUG_FLAGS) -o optimized_memory_transfer_nvlink_debug optimized_memory_transfer_nvlink.cu
	$(NVCC) $(NVCC_DEBUG_FLAGS) -o cpu_gpu_grace_blackwell_coherency_debug cpu_gpu_grace_blackwell_coherency.cu
