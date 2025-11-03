# Chapter 8 Makefile - Control-flow and occupancy tuning examples

include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) --threads=0
SUFFIX = $(TARGET_SUFFIX)

CUDA_SOURCES = $(wildcard *.cu)
BASE_TARGETS = $(CUDA_SOURCES:.cu=)
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean compare blackwell grace-blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 8 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

$(TARGETS): %$(SUFFIX): %.cu
	@echo "Compiling $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

compare: clean
	@echo "========================================"
	@echo "Building Chapter 8 samples for both architectures..."
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

profile-nsys: all
	@echo "Nsight Systems timeline profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o nsys_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-ncu: all
	@echo "Nsight Compute kernel profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		ncu --section=SpeedOfLight --metrics breakdown:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all -o ncu_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o hta_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
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
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o comprehensive_timeline_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --section=SpeedOfLight --metrics breakdown:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all -o comprehensive_kernel_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "3. Perf system-level analysis..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || true; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

clean:
	rm -f $(addsuffix _sm100,$(BASE_TARGETS)) $(addsuffix _sm121,$(BASE_TARGETS)) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
