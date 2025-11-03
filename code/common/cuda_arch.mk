# Shared CUDA architecture configuration for Blackwell and Grace-Blackwell builds.
#
# Usage from chapter Makefiles (located under ch*/):
#   include ../common/cuda_arch.mk
#   NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) ...
#   # optional: USE_ARCH_SUFFIX := 0  # to disable suffixing targets
#
# Exposes:
#   ARCH                - Selected GPU architecture (default: sm_100)
#   ARCH_NAME           - Human-readable architecture label
#   ARCH_SUFFIX         - Suffix (_sm100 or _sm121) for architecture-specific binaries
#   TARGET_SUFFIX       - Suffix applied when USE_ARCH_SUFFIX is 1
#   CUDA_NVCC_ARCH_FLAGS- Baseline nvcc flags for the selected architecture
#   ARCH_LIST           - Ordered list of supported architectures (sm_100, sm_121)

ARCH ?= sm_100
CUDA_VERSION ?= 13.0
NVCC ?= nvcc

ARCH_LIST := sm_100 sm_121

ifeq ($(ARCH),sm_121)
ARCH_NAME := Grace-Blackwell GB10 (CC 12.1)
ARCH_SUFFIX := _sm121
CUDA_ARCH_GENCODE := -gencode arch=compute_121,code=[sm_121,compute_121]
HOST_ARCH_FLAGS := -Xcompiler -mcpu=native
else ifeq ($(ARCH),sm_100)
ARCH_NAME := Blackwell B200/B300 (CC 10.0)
ARCH_SUFFIX := _sm100
CUDA_ARCH_GENCODE := -gencode arch=compute_100,code=[sm_100,compute_100]
HOST_ARCH_FLAGS :=
else
$(error Unsupported ARCH=$(ARCH). Supported values: sm_100, sm_121)
endif

# Base nvcc flags shared across the project. Chapters may append additional flags as needed.
CUDA_CXX_STANDARD ?= 17
CUDA_NVCC_BASE_FLAGS ?= -O3 -std=c++$(CUDA_CXX_STANDARD) $(CUDA_ARCH_GENCODE) --expt-relaxed-constexpr -Xcompiler -fPIC
CUDA_NVCC_ARCH_FLAGS := $(CUDA_NVCC_BASE_FLAGS) $(HOST_ARCH_FLAGS)

# Control whether binaries get suffixed with architecture-specific suffixes.
USE_ARCH_SUFFIX ?= 1
ifeq ($(USE_ARCH_SUFFIX),1)
TARGET_SUFFIX := $(ARCH_SUFFIX)
else
TARGET_SUFFIX :=
endif
