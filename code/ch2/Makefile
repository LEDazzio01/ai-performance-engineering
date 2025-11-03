# Chapter 2 Makefile (includes Blackwell and Grace-Blackwell NVLink demos)

include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS)
SUFFIX = $(TARGET_SUFFIX)

BASE_TARGETS = nvlink_c2c_p2p_blackwell gb200_coherency cpu_gpu_zero_copy_coherent
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean compare profile-hta profile-perf profile-all blackwell grace-blackwell $(TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 2 targets with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

nvlink_c2c_p2p_blackwell$(SUFFIX): nvlink_c2c_p2p_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

gb200_coherency$(SUFFIX): gb200_grace_blackwell_coherency.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cpu_gpu_zero_copy_coherent$(SUFFIX): cpu_gpu_zero_copy_coherent.cu
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
	@echo "  ./nvlink_c2c_p2p_blackwell_sm100"
	@echo "  ./nvlink_c2c_p2p_blackwell_sm121"

blackwell: ARCH=sm_100
blackwell: clean all

grace-blackwell: ARCH=sm_121
grace-blackwell: clean all

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
	rm -f nvlink_c2c_p2p_blackwell_sm100 nvlink_c2c_p2p_blackwell_sm121
	rm -f gb200_coherency_sm100 gb200_coherency_sm121
	rm -f *.o *.so *.a
	rm -f nsys_profile_* hta_profile_* perf.data_* comprehensive_*
