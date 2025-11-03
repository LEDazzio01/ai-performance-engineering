#include <cuda_runtime.h>
#include <cstdio>

#include "warp_specialized_kernel_two_pipelines_multistream.cu"

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    using namespace ch11;

    const int tiles = 128;
    const size_t elems = static_cast<size_t>(tiles) * kTileElems;
    const size_t bytes = elems * sizeof(float);

    std::vector<float> h_a(elems, 1.0f);
    std::vector<float> h_b(elems, 2.0f);
    std::vector<float> h_out(elems, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_a, bytes));
    check(cudaMalloc(&d_b, bytes));
    check(cudaMalloc(&d_out, bytes));

    check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t streams[2];
    check(cudaStreamCreate(&streams[0]));
    check(cudaStreamCreate(&streams[1]));

    const int tiles_per_stream = tiles / 2;
    const dim3 block(256);
    const dim3 grid(std::min(tiles_per_stream, 64));
    const size_t shared_bytes = 3 * kPipelineStages * kTileElems * sizeof(float);

    for (int s = 0; s < 2; ++s) {
        size_t tile_offset = static_cast<size_t>(s) * tiles_per_stream * kTileElems;
        const float* a_ptr = d_a + tile_offset;
        const float* b_ptr = d_b + tile_offset;
        float* out_ptr = d_out + tile_offset;

        warp_specialized_kernel_two_pipelines_multistream<<<grid, block, shared_bytes, streams[s]>>>(
            a_ptr, b_ptr, out_ptr, tiles_per_stream);
    }

    check(cudaGetLastError());
    check(cudaStreamSynchronize(streams[0]));
    check(cudaStreamSynchronize(streams[1]));

    check(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        double expected = h_a[i] * 2.0 + h_b[i];
        max_err = std::max(max_err, std::abs(expected - h_out[i]));
    }
    std::printf("warp_specialized_two_pipelines_multistream_driver complete. max_err=%.3e\n", max_err);

    check(cudaStreamDestroy(streams[0]));
    check(cudaStreamDestroy(streams[1]));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}

#else  // CUDA < 13

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::printf("warp_specialized_two_pipelines_multistream requires CUDA 13+ asynchronous "
                    "pipeline APIs. Detected compute capability %d.%d; skipping.\n",
                    prop.major,
                    prop.minor);
    } else {
        std::printf("warp_specialized_two_pipelines_multistream requires CUDA 13+ asynchronous "
                    "pipeline APIs. Unable to query device properties; skipping.\n");
    }
    return 0;
}

#endif  // CUDA toolkit check
