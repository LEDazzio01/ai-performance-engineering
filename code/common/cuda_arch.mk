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
#   ARCH_SUFFIX         - Suffix (_sm100, _sm103, or _sm121) for architecture-specific binaries
#   TARGET_SUFFIX       - Suffix applied when USE_ARCH_SUFFIX is 1
#   CUDA_NVCC_ARCH_FLAGS- Baseline nvcc flags for the selected architecture
#   ARCH_LIST           - Ordered list of supported architectures (sm_100, sm_103, sm_121)

CUDA_VERSION ?= 13.0
NVCC ?= nvcc
PYTHON ?= python3

CUDA_ARCH_MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUDA_COMMON_DIR := $(dir $(CUDA_ARCH_MK_PATH))

DEFAULT_ARCH := sm_100
AUTO_ARCH_DETECTION ?= 1

ifeq ($(origin ARCH), undefined)
  ifeq ($(AUTO_ARCH_DETECTION),1)
    DETECTED_ARCH := $(strip $(shell $(PYTHON) $(CUDA_COMMON_DIR)/python/detect_sm.py 2>/dev/null))
  endif
endif

ifeq ($(strip $(DETECTED_ARCH)),)
  ARCH ?= $(DEFAULT_ARCH)
  ARCH_SOURCE := default
else
  ARCH := $(DETECTED_ARCH)
  ARCH_SOURCE := auto
endif

ifeq ($(origin ARCH), command line)
  ARCH_SOURCE := user
endif
ifeq ($(origin ARCH), environment)
  ARCH_SOURCE := user
endif

ifeq ($(ARCH_SOURCE),auto)
$(info [cuda_arch] Auto-detected GPU architecture $(ARCH) (override with ARCH=<sm_100|sm_103|sm_121>))
endif
NVTX_STUB_DIR := $(abspath $(CUDA_COMMON_DIR)/nvtx_stub)
NVTX_STUB_LIB := $(NVTX_STUB_DIR)/libnvToolsExt.a
NVTX_STUB_SCRIPT := $(abspath $(CUDA_COMMON_DIR)/python/nvtx_stub.py)

ARCH_LIST := sm_100 sm_103 sm_121

ifeq ($(ARCH),sm_121)
ARCH_NAME := Grace-Blackwell GB10 (CC 12.1)
ARCH_SUFFIX := _sm121
# CUDA 13.0's ptxas does not yet accept sm_121, so clamp to sm_120 targets.
# This keeps GB10 builds working until CUDA 13.1 introduces native support.
CUDA_ARCH_GENCODE := -gencode arch=compute_120,code=[sm_120,compute_120]
HOST_ARCH_FLAGS := -Xcompiler -mcpu=native
else ifeq ($(ARCH),sm_100)
ARCH_NAME := Blackwell B200/B300 (CC 10.0)
ARCH_SUFFIX := _sm100
CUDA_ARCH_GENCODE := -gencode arch=compute_100,code=[sm_100,compute_100]
HOST_ARCH_FLAGS :=
else ifeq ($(ARCH),sm_103)
ARCH_NAME := Blackwell Ultra B300 (CC 10.3)
ARCH_SUFFIX := _sm103
CUDA_ARCH_GENCODE := -gencode arch=compute_103,code=[sm_103,compute_103]
HOST_ARCH_FLAGS :=
else
$(error Unsupported ARCH=$(ARCH). Supported values: sm_100, sm_103, sm_121)
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

$(NVTX_STUB_LIB):
	$(PYTHON) $(NVTX_STUB_SCRIPT) --output $@

# Enable NVTX profiling helpers by default. Set NVTX_ENABLED=0 to disable.
NVTX_ENABLED ?= 1
ifeq ($(strip $(NVTX_ENABLED)),1)
CUDA_NVTX_CFLAGS := -DENABLE_NVTX_PROFILING
CUDA_NVTX_LDFLAGS := -L$(NVTX_STUB_DIR) -lnvToolsExt
CUDA_NVTX_DEPS := $(NVTX_STUB_LIB)
else
CUDA_NVTX_CFLAGS :=
CUDA_NVTX_LDFLAGS :=
CUDA_NVTX_DEPS :=
endif
