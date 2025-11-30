// NVSHMEM IBGDA microbenchmark (put vs scalar p)
//
// Mirrors the small-message experiments from the NVIDIA IBGDA blog:
// - Block put path stresses queue depth and batching.
// - Scalar p path shows how many tiny messages per second IBGDA can sustain.
//
// Build (requires NVSHMEM with IBGDA support):
//   make nvshmem_ibgda_microbench ARCH=sm_121   # via ch04/Makefile
//   # or direct:
//   nvcc -std=c++17 -O3 -arch=sm_90a -rdc=true \
//     -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib -lnvshmem -lcudart \
//     -o nvshmem_ibgda_microbench nvshmem_ibgda_microbench.cu
//
// Run (use nvshmemrun; flip IBGDA on/off via env):
//   NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=gpu \
//   nvshmemrun -np 2 ./nvshmem_ibgda_microbench --mode=p --bytes=1024 --ctas=8 --iters=1000
//   NVSHMEM_IB_ENABLE_IBGDA=0 ... # proxy baseline

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK_CUDA(cmd)                                                         \
    do {                                                                        \
        cudaError_t _e = (cmd);                                                 \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(_e));                               \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CHECK_NVSHMEM(cmd) (cmd)

enum class Mode { kBlockPut, kScalarP };

__global__ void block_put_kernel(float* buf, size_t n, int peer, int iters) {
    size_t block_span = (n + gridDim.x - 1) / gridDim.x;
    size_t offset = blockIdx.x * block_span;
    size_t count = 0;
    if (offset < n) {
        count = min(block_span, n - offset);
    }
    if (count == 0) return;

    for (int i = 0; i < iters; ++i) {
        nvshmemx_putmem_nbi_block(buf + offset, buf + offset,
                                  count * sizeof(float), peer);
    }
}

__global__ void scalar_p_kernel(float* buf, size_t n, int peer, int iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = buf[idx];
    for (int i = 0; i < iters; ++i) {
        nvshmem_float_p(buf + idx, val, peer);
    }
}

struct Options {
    size_t bytes = 1024;
    int iters = 1000;
    int ctas = 8;
    int threads = 256;
    Mode mode = Mode::kScalarP;
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--bytes=", 8) == 0) {
            opts.bytes = std::strtoull(argv[i] + 8, nullptr, 10);
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            opts.iters = std::atoi(argv[i] + 8);
        } else if (std::strncmp(argv[i], "--ctas=", 7) == 0) {
            opts.ctas = std::atoi(argv[i] + 7);
        } else if (std::strncmp(argv[i], "--threads=", 10) == 0) {
            opts.threads = std::atoi(argv[i] + 10);
        } else if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            std::string m = argv[i] + 7;
            if (m == "put") {
                opts.mode = Mode::kBlockPut;
            } else if (m == "p") {
                opts.mode = Mode::kScalarP;
            } else {
                std::fprintf(stderr, "Unknown mode %s (use put or p)\n", m.c_str());
                std::exit(1);
            }
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            std::exit(1);
        }
    }
    return opts;
}

int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        std::fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    // Tie PE to GPU round-robin.
    CHECK_NVSHMEM(nvshmem_init());
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) {
            std::printf("Single-PE run (npes=%d); skipping IBGDA measurement.\n", npes);
        }
        CHECK_NVSHMEM(nvshmem_finalize());
        return 0;
    }
    CHECK_CUDA(cudaSetDevice(mype % dev_count));

    int peer = (mype + 1) % npes;
    size_t elems = (opts.bytes + sizeof(float) - 1) / sizeof(float);
    if (elems == 0) elems = 1;

    float* buf = static_cast<float*>(nvshmem_malloc(elems * sizeof(float)));
    if (!buf) {
        std::fprintf(stderr, "nvshmem_malloc failed\n");
        return 1;
    }

    CHECK_CUDA(cudaMemset(buf, 0, elems * sizeof(float)));
    CHECK_NVSHMEM(nvshmem_barrier_all());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 grid(opts.ctas);
    dim3 block(opts.threads);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    if (opts.mode == Mode::kBlockPut) {
        block_put_kernel<<<grid, block>>>(buf, elems, peer, opts.iters);
    } else {
        scalar_p_kernel<<<grid, block>>>(buf, elems, peer, opts.iters);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_NVSHMEM(nvshmem_barrier_all());

    double seconds = ms / 1000.0;
    size_t active_threads = (size_t)opts.ctas * (size_t)opts.threads;
    if (opts.mode == Mode::kBlockPut) {
        size_t per_block_elems = (elems + opts.ctas - 1) / opts.ctas;
        size_t total_bytes = (size_t)opts.iters * per_block_elems * sizeof(float);
        double gbps = (total_bytes / seconds) / 1e9;
        if (mype == 0) {
            std::printf("mode=put bytes=%zu ctas=%d threads=%d iters=%d time=%.3f ms bw=%.2f GB/s\n",
                        opts.bytes, opts.ctas, opts.threads, opts.iters, ms, gbps);
        }
    } else {
        size_t ops = (size_t)opts.iters * active_threads;
        double mops = (ops / seconds) / 1e6;
        if (mype == 0) {
            std::printf("mode=p bytes=%zu ctas=%d threads=%d iters=%d time=%.3f ms rate=%.1f MOPS\n",
                        opts.bytes, opts.ctas, opts.threads, opts.iters, ms, mops);
        }
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    nvshmem_free(buf);
    CHECK_NVSHMEM(nvshmem_finalize());
    return 0;
}
