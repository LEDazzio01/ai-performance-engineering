# Common Infrastructure

Shared utilities for all chapter examples to ensure consistent build system, profiling workflow, and benchmarking methodology.

## Directory Structure

```
common/
├── build/
│   └── cuda_common.mk          # Shared Makefile rules
├── headers/
│   ├── cuda_helpers.cuh        # CUDA error checking, timing
│   ├── profiling_helpers.cuh   # NVTX markers for profiling
│   ├── arch_detection.cuh      # GPU architecture detection & limits
│   ├── tma_helpers.cuh         # Tensor Memory Accelerator utilities
│   └── cuda13_demos.cuh        # CUDA 13.0 feature demos
├── profiling/
│   ├── profile_cuda.sh         # Standard CUDA profiling script
│   ├── profile_pytorch.sh      # PyTorch profiling script
│   └── compare_results.py      # Compare baseline vs optimized
└── python/
    ├── benchmark_utils.py      # Benchmarking utilities
    └── profiling_utils.py      # PyTorch profiling helpers
```

## Usage

### Build System

In your chapter Makefile:

```makefile
# Define your targets
TARGETS = my_kernel other_kernel

# Include common build rules
include ../../common/build/cuda_common.mk

# That's it! Common flags and rules are automatically applied
```

The common Makefile provides:
- Dual-architecture builds (sm_100 + sm_121)
- Consistent compiler flags (-O3, -std=c++17)
- Common headers included automatically
- Debug build support: `make DEBUG=1`

### CUDA Headers

#### Basic Helpers
```cpp
#include "../../common/headers/cuda_helpers.cuh"
#include "../../common/headers/profiling_helpers.cuh"

int main() {
    // GPU info
    printGpuInfo();
    
    // Error checking
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    
    // Timing
    CudaTimer timer;
    timer.start();
    myKernel<<<grid, block>>>(d_data);
    float ms = timer.stop();
    
    // Calculate metrics
    float bandwidth = calculateBandwidthGBs(bytes, ms);
    float tflops = calculateGFLOPS(flops, ms);
    
    return 0;
}
```

#### Architecture Detection
```cpp
#include "../../common/headers/arch_detection.cuh"

int main() {
    // Query GPU capabilities
    const auto& limits = cuda_arch::get_architecture_limits();
    
    // Check features
    if (limits.supports_clusters) {
        printf("Cluster size: %d\n", limits.max_cluster_size);
    }
    if (limits.has_grace_coherence) {
        printf("Grace-Blackwell coherence available\n");
    }
    
    // Select optimal tile size
    auto tile = cuda_arch::select_tensor_core_tile();
    printf("Tensor core tile: %dx%dx%d\n", tile.m, tile.n, tile.k);
    
    // Get TMA limits
    auto tma = cuda_arch::get_tma_limits();
    printf("TMA 2D box: %ux%u\n", tma.max_2d_box_width, tma.max_2d_box_height);
    
    return 0;
}
```

#### TMA (Tensor Memory Accelerator) Helpers
```cpp
#include "../../common/headers/arch_detection.cuh"
#include "../../common/headers/tma_helpers.cuh"

int main() {
    // Check TMA support
    if (!cuda_tma::device_supports_tma()) {
        printf("TMA not supported (requires SM 9.0+)\n");
        return 1;
    }
    
    // Create tensor map
    CUtensorMap desc;
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    bool ok = cuda_tma::make_2d_tensor_map(
        desc, encode, d_data, width, height, ld,
        box_width, box_height, CU_TENSOR_MAP_SWIZZLE_NONE);
    
    // Use in kernel with cp_async_bulk_tensor operations
    return 0;
}
```

#### CUDA 13.0 Demos
```cpp
#include "../../common/headers/cuda13_demos.cuh"

int main() {
    // Run stream-ordered memory allocation demo
    cuda13_demos::run_stream_ordered_memory_demo();
    
    // Run TMA copy demo
    cuda13_demos::run_simple_tma_demo();
    
    return 0;
}
```

### Profiling CUDA Examples

```bash
# From your chapter directory (e.g., ch7/)
../../common/profiling/profile_cuda.sh ./my_kernel baseline
../../common/profiling/profile_cuda.sh ./optimized_kernel optimized

# Results saved to ../../results/ch7/
```

### Profiling PyTorch Examples

```bash
# From your chapter directory
../../common/profiling/profile_pytorch.sh ./training.py --batch-size 32

# View Chrome trace in chrome://tracing
```

### Python Benchmarking

```python
import sys
sys.path.append('../../common/python')
from benchmark_utils import compare_implementations, print_gpu_info

def baseline():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

def optimized():
    # Your optimized version
    pass

print_gpu_info()
compare_implementations(baseline, optimized, "My Optimization")
```

### Python Profiling

```python
import sys
sys.path.append('../../common/python')
from profiling_utils import ProfilerContext, profile_with_chrome_trace

# Method 1: Context manager
with ProfilerContext("training", trace_dir="./traces"):
    for batch in dataloader:
        train_step(batch)

# Method 2: Single function
profile_with_chrome_trace(
    lambda: model(input),
    trace_path="./forward_pass.json"
)
```

## Benefits

1. **Consistency**: All chapters use the same profiling methodology
2. **Maintainability**: Bug fixes and improvements propagate to all chapters
3. **Pedagogy**: Students see the same patterns across all examples
4. **Quality**: Professional-grade error checking and profiling built-in

## Adding New Utilities

To add new common utilities:

1. Add header files to `headers/`
2. Add Python modules to `python/`
3. Add shell scripts to `profiling/`
4. Update this README with usage examples
5. Test with at least one chapter before rolling out

## Migration Guide

For existing chapters:

1. Replace custom Makefile with common include
2. Replace error checking with `CUDA_CHECK()` macro
3. Add NVTX markers for profiling hot spots
4. Use common benchmarking utilities in Python examples

See CHAPTER_AUDIT.md for detailed migration status per chapter.
