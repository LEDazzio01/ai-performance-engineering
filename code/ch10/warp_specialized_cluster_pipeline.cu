#include <cuda_runtime.h>
#include <cstdio>

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)

#include <cooperative_groups.h>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

__global__ void cluster_sum_kernel(const float* __restrict__ data,
                                   float* __restrict__ cluster_out,
                                   int elems_per_block) {
    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();

    extern __shared__ float shared[];
    float sum = 0.0f;
    int base = blockIdx.x * elems_per_block;
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        sum += data[base + i];
    }

    shared[threadIdx.x] = sum;
    block.sync();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        block.sync();
    }

    if (threadIdx.x == 0) {
        cluster_out[blockIdx.x] = shared[0];
    }

    cluster.sync();

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        int blocks_in_cluster = cluster.dim_blocks().x * cluster.dim_blocks().y * cluster.dim_blocks().z;
        int cluster_start = blockIdx.x - cluster.block_rank();
        float cluster_total = 0.0f;
        for (int b = 0; b < blocks_in_cluster && (cluster_start + b) < gridDim.x; ++b) {
            cluster_total += cluster_out[cluster_start + b];
        }
        cluster_out[cluster_start] = cluster_total;
    }
}

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

int main() {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));

    int cluster_launch = 0;
#ifdef cudaDevAttrClusterLaunch
    cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0);
#endif

    bool supports_clusters = cluster_launch > 0 || prop.major >= 9;
    if (!supports_clusters) {
        printf("Thread block clusters not supported on this GPU.\n");
        return 0;
    }

    int cluster_size = 1;
    if (prop.major >= 10) {
        cluster_size = 8;
    } else if (prop.major == 9) {
        cluster_size = 4;
    }

    const int blocks = cluster_size * 2;
    const int elems_per_block = 2048;
    const size_t total_elems = size_t(blocks) * elems_per_block;

    std::vector<float> h_in(total_elems, 1.0f);
    std::vector<float> h_out(blocks, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_in, total_elems * sizeof(float)));
    check(cudaMalloc(&d_out, blocks * sizeof(float)));
    check(cudaMemcpy(d_in, h_in.data(), total_elems * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemset(d_out, 0, blocks * sizeof(float)));

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = cfg.blockDim.x * sizeof(float);

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    check(cudaFuncSetAttribute(
        cluster_sum_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));

    check(cudaLaunchKernelEx(&cfg,
                             cluster_sum_kernel,
                             d_in,
                             d_out,
                             elems_per_block));
    check(cudaDeviceSynchronize());

    check(cudaMemcpy(h_out.data(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Cluster 0 aggregated sum: %.1f\n", h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

#else  // CUDA < 13 or header unavailable

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("warp_specialized_cluster_pipeline requires CUDA 13+ cluster/TMA APIs. "
               "Detected compute capability %d.%d; skipping.\n",
               prop.major,
               prop.minor);
    } else {
        printf("warp_specialized_cluster_pipeline requires CUDA 13+ cluster/TMA APIs. "
               "Unable to query device properties; skipping.\n");
    }
    return 0;
}

#endif  // CUDA toolkit check
