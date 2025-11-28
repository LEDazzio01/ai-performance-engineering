# Makefile for Matching cuBLAS Lab

NVCC = nvcc
ARCH ?= sm_100

# Detect architecture if not specified
ifeq ($(ARCH),)
    ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | sed 's/^/sm_/')
endif

NVCC_FLAGS = -O3 -std=c++17 \
    -gencode arch=compute_100a,code=[sm_100a,compute_100a] \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC \
    -lineinfo

TARGETS = kernels.so

all: $(TARGETS)

kernels.so: kernels.cu
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $<

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean

