# Makefile for double_buffered_pipeline.cu

# Detect CUDA architecture
CUDA_ARCH ?= native

# Compiler settings
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = -arch=$(CUDA_ARCH) -lineinfo --expt-relaxed-constexpr

# Include path for common headers
INCLUDES = -I..

# Common headers
COMMON_HEADERS = ../common/headers/arch_detection.cuh ../common/headers/tma_helpers.cuh

# Target executable
TARGET = double_buffered_pipeline
SOURCE = double_buffered_pipeline.cu

.PHONY: all clean run test test-gds demo help

all: $(TARGET)

$(TARGET): $(SOURCE) $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	@echo "=== Testing Memory Pipeline (CUDA) ==="
	@echo "=== Testing small matrix (16x16x16) ==="
	./$(TARGET) 16 16 16
	@echo ""
	@echo "=== Testing medium matrix (128x128x128) ==="
	./$(TARGET) 128 128 128
	@echo ""
	@echo "=== Testing large matrix (1024x1024x1024) ==="
	./$(TARGET) 1024 1024 1024

test-gds:
	@echo "=== Testing GPUDirect Storage (Python) ==="
	python3 cufile_gds_example.py

demo: $(TARGET)
	@echo "=== Running Both Examples ==="
	./demo_both_examples.sh

help:
	@echo "Chapter 10: Memory Pipelines and GPUDirect Storage"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build memory pipeline (CUDA)"
	@echo "  make test     - Test memory pipeline with multiple sizes"
	@echo "  make test-gds - Test GPUDirect Storage (Python)"
	@echo "  make demo     - Run both examples back-to-back"
	@echo "  make run      - Run memory pipeline (default size)"
	@echo "  make clean    - Remove compiled binaries"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  Memory Pipeline:       ./double_buffered_pipeline 512 512 512"
	@echo "  GPUDirect Storage:     python3 cufile_gds_example.py"
	@echo "  Both:                  make demo"

clean:
	rm -f $(TARGET)
