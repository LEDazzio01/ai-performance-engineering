#pragma once

/**
 * Minimal CUDA 13 teaser helpers shared by introductory chapters.
 *
 * These wrappers keep the chapter samples runnable on host compilers that
 * build against pre-CUDA 13 toolchains, while still giving Blackwell users
 * a quick way to see the new workflows in action.
 *
 * - stream_ordered_teaser(): wraps the cudaMallocAsync demo.
 * - tma_teaser(): wraps a small Tensor Memory Accelerator copy example.
 *
 * When CUDA 13 headers are unavailable, these simply print guidance directing
 * readers to the full implementations later in the book.
 */

#include <cstdio>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#include "headers/cuda13_demos.cuh"
#endif

namespace cuda13_teasers {

inline void stream_ordered_teaser() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    std::printf("\n[Teaser] Running cudaMallocAsync stream-ordered allocation demo…\n");
    cuda13_demos::run_stream_ordered_memory_demo();
#else
    std::printf(
        "\n[Teaser] cudaMallocAsync unavailable on this toolkit. "
        "See ch11/stream_ordered_allocator.cu for the full walkthrough.\n");
#endif
}

inline void tma_teaser() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    std::printf("\n[Teaser] Running Tensor Memory Accelerator copy demo…\n");
    cuda13_demos::run_simple_tma_demo();
#else
    std::printf(
        "\n[Teaser] Tensor Memory Accelerator requires CUDA 13+. "
        "See ch7/async_prefetch_tma.cu or ch10/tma_2d_pipeline_blackwell.cu for the full example.\n");
#endif
}

}  // namespace cuda13_teasers

