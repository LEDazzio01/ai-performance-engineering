# Makefile for Chapter 4: Multi-GPU Communication Examples
# Supports dual-architecture builds for Blackwell B200 and Grace-Blackwell GB10.

include ../common/cuda_arch.mk

NVSHMEM_HOME ?= /usr/local/nvshmem
NVSHMEM_INCLUDE = -I$(NVSHMEM_HOME)/include
NVSHMEM_LIB = -L$(NVSHMEM_HOME)/lib -lnvshmem
HAS_NVSHMEM := $(shell test -d $(NVSHMEM_HOME) && echo yes || echo no)

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS)
SUFFIX = $(TARGET_SUFFIX)

NVSHMEM_TARGETS = nvshmem_8gpu_examples nvshmem_advanced_8gpu nvshmem_multinode_example \
                  nvshmem_pipeline_patterns nvshmem_tensor_parallel
TARGETS = $(addsuffix $(SUFFIX), $(NVSHMEM_TARGETS))

.PHONY: all help nvshmem compare clean install-help blackwell grace-blackwell $(TARGETS)

all: help
	@echo ""
	@echo "NVSHMEM status: $(HAS_NVSHMEM)"
ifeq ($(HAS_NVSHMEM),no)
	@echo "  → NVSHMEM not found at $(NVSHMEM_HOME)"
	@echo "  → Set NVSHMEM_HOME to the correct path if installed"
endif

help:
	@echo "Available targets:"
	@echo "  make nvshmem         - Compile NVSHMEM examples (requires NVSHMEM)"
	@echo "  make compare         - Build both architectures back-to-back"
	@echo "  make clean           - Remove build outputs"
	@echo "  make install-help    - Show NVSHMEM installation tips"
	@echo ""
	@echo "Current configuration:"
	@echo "  ARCH        = $(ARCH)"
	@echo "  ARCH_NAME   = $(ARCH_NAME)"
	@echo "  Binary suffix = $(SUFFIX)"

nvshmem: $(TARGETS)
ifeq ($(HAS_NVSHMEM),yes)
	@echo "✓ NVSHMEM examples compiled successfully"
else
	@echo "⚠ NVSHMEM not found - compiling in educational mode"
endif

nvshmem_8gpu_examples$(SUFFIX): nvshmem_8gpu_examples.cu
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@
	@echo "✓ Basic examples compiled with NVSHMEM support"
	@echo "  Run: nvshmemrun -np 8 ./nvshmem_8gpu_examples$(SUFFIX)"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_multinode_example$(SUFFIX): nvshmem_multinode_example.cu
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@
	@echo "✓ Multi-node example compiled with NVSHMEM support"
	@echo "  Run: nvshmemrun -np <world> ./nvshmem_multinode_example$(SUFFIX) --gpus-per-node 8"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_pipeline_patterns$(SUFFIX): nvshmem_pipeline_patterns.cu
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@
	@echo "✓ Pipeline patterns compiled with NVSHMEM support"
	@echo "  Run: nvshmemrun -np 2 ./nvshmem_pipeline_patterns$(SUFFIX)"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_advanced_8gpu$(SUFFIX): nvshmem_advanced_8gpu.cu
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@
	@echo "✓ Advanced patterns compiled with NVSHMEM support"
	@echo "  Run: nvshmemrun -np 8 ./nvshmem_advanced_8gpu$(SUFFIX)"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_tensor_parallel$(SUFFIX): nvshmem_tensor_parallel.cu
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -lcuda -o $@
	@echo "✓ Tensor parallel kernels compiled with NVSHMEM support"
	@echo "  Run: nvshmemrun -np 8 ./nvshmem_tensor_parallel$(SUFFIX) --test all"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Compiled in educational mode (no NVSHMEM library)"
	@echo "This version shows API patterns but won't execute communication"
endif

compare: clean
	@echo "========================================"
	@echo "Building Chapter 4 NVSHMEM samples for both architectures..."
	@echo "========================================"
	@for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} nvshmem; \
	done

blackwell: ARCH=sm_100
blackwell: clean nvshmem

grace-blackwell: ARCH=sm_121
grace-blackwell: clean nvshmem

clean:
	rm -f $(addsuffix _sm100,$(NVSHMEM_TARGETS)) $(addsuffix _sm121,$(NVSHMEM_TARGETS))
	rm -f *.o
	@echo "Cleaned Chapter 4 NVSHMEM binaries"

install-help:
	@echo "To install NVSHMEM:"
	@echo "  1. Download from NVIDIA: https://developer.nvidia.com/nvshmem"
	@echo "  2. Extract and install to $(NVSHMEM_HOME)"
	@echo "  3. Set environment variable: export NVSHMEM_HOME=$(NVSHMEM_HOME)"
	@echo "  4. Run: make nvshmem"
