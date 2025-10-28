# Multi-stage Docker build for optimized GPU performance
FROM nvcr.io/nvidia/pytorch:25.05-py3 AS base

# Install additional dependencies for performance optimization
RUN apt-get update && apt-get install -y \
    numactl \
    libnuma-dev \
    htop \
    iotop \
    tcmalloc-minimal \
    jemalloc \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for optimized memory allocation
ENV MALLOC_CONF="narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000,background_thread:true"
ENV TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=536870912
ENV TCMALLOC_RELEASE_RATE=16

# Set CUDA environment variables for PyTorch 2.9 (cu130)
ENV PYTORCH_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Copy application code
COPY . /app
WORKDIR /app

# Install Python dependencies with latest versions
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu130 \
        'torch==2.9.*+cu130' 'torchvision==0.24.*+cu130' 'torchaudio==2.9.*+cu130' && \
    pip install triton==3.5.0 && \
    pip install nvidia-ml-py==12.560.30 psutil==6.1.0 GPUtil==1.4.0


# Runtime stage for smaller final image
FROM base AS runtime

# Set up runtime environment
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Entry point with NUMA awareness
ENTRYPOINT ["numactl", "--interleave=all", "python"]
CMD ["train.py"]
