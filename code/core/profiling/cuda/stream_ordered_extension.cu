// CUDA extension demonstrating standard vs stream-ordered allocations.

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <array>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(status));             \
        }                                                                     \
    } while (0)

constexpr int NUM_STREAMS = 3;

__global__ void scale_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        #pragma unroll 4
        for (int i = 0; i < 8; ++i) {
            val = val * 1.0001f + 0.1f;
        }
        output[idx] = val;
    }
}

void run_allocation_workload(int64_t elements, int iterations, bool stream_ordered) {
    TORCH_CHECK(elements > 0, "elements must be > 0");
    TORCH_CHECK(iterations > 0, "iterations must be > 0");
    size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    
    std::array<cudaStream_t, NUM_STREAMS> streams{};
    for (auto& st : streams) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    }
    
    std::array<float*, NUM_STREAMS> h_buffers{};
    for (auto& host_ptr : h_buffers) {
        CUDA_CHECK(cudaMallocHost(&host_ptr, bytes));
        for (int64_t i = 0; i < elements; ++i) {
            host_ptr[i] = static_cast<float>(i % 1024) * 0.5f;
        }
    }
    
    for (int iter = 0; iter < iterations; ++iter) {
        std::array<float*, NUM_STREAMS> d_in{};
        std::array<float*, NUM_STREAMS> d_out{};
        
        for (int s = 0; s < NUM_STREAMS; ++s) {
            if (stream_ordered) {
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_in[s]), bytes, streams[s]));
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_out[s]), bytes, streams[s]));
            } else {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in[s]), bytes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out[s]), bytes));
            }
            
            CUDA_CHECK(cudaMemcpyAsync(
                d_in[s],
                h_buffers[s],
                bytes,
                cudaMemcpyHostToDevice,
                streams[s]
            ));
            
            int threads = 256;
            int blocks = static_cast<int>((elements + threads - 1) / threads);
            scale_kernel<<<blocks, threads, 0, streams[s]>>>(d_in[s], d_out[s], static_cast<int>(elements));
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaMemcpyAsync(
                h_buffers[s],
                d_out[s],
                bytes,
                cudaMemcpyDeviceToHost,
                streams[s]
            ));
        }
        
        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            if (stream_ordered) {
                CUDA_CHECK(cudaFreeAsync(d_in[s], streams[s]));
                CUDA_CHECK(cudaFreeAsync(d_out[s], streams[s]));
            } else {
                CUDA_CHECK(cudaFree(d_in[s]));
                CUDA_CHECK(cudaFree(d_out[s]));
            }
        }
    }
    
    for (auto& st : streams) {
        CUDA_CHECK(cudaStreamDestroy(st));
    }
    for (auto& host_ptr : h_buffers) {
        CUDA_CHECK(cudaFreeHost(host_ptr));
    }
}

}  // namespace

void run_standard_allocator(int64_t elements, int iterations) {
    run_allocation_workload(elements, iterations, false);
}

void run_stream_ordered_allocator(int64_t elements, int iterations) {
    run_allocation_workload(elements, iterations, true);
}

torch::Tensor run_standard_allocator_capture(int64_t elements, int iterations) {
    TORCH_CHECK(elements > 0, "elements must be > 0");
    TORCH_CHECK(iterations > 0, "iterations must be > 0");
    size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    std::array<cudaStream_t, NUM_STREAMS> streams{};
    for (auto& st : streams) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    }

    std::array<float*, NUM_STREAMS> h_buffers{};
    for (auto& host_ptr : h_buffers) {
        CUDA_CHECK(cudaMallocHost(&host_ptr, bytes));
        for (int64_t i = 0; i < elements; ++i) {
            host_ptr[i] = static_cast<float>(i % 1024) * 0.5f;
        }
    }

    for (int iter = 0; iter < iterations; ++iter) {
        std::array<float*, NUM_STREAMS> d_in{};
        std::array<float*, NUM_STREAMS> d_out{};

        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in[s]), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out[s]), bytes));

            CUDA_CHECK(cudaMemcpyAsync(d_in[s], h_buffers[s], bytes, cudaMemcpyHostToDevice, streams[s]));

            int threads = 256;
            int blocks = static_cast<int>((elements + threads - 1) / threads);
            scale_kernel<<<blocks, threads, 0, streams[s]>>>(d_in[s], d_out[s], static_cast<int>(elements));
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(h_buffers[s], d_out[s], bytes, cudaMemcpyDeviceToHost, streams[s]));
        }

        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            CUDA_CHECK(cudaFree(d_in[s]));
            CUDA_CHECK(cudaFree(d_out[s]));
        }
    }

    auto out = torch::empty({NUM_STREAMS}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* out_ptr = out.data_ptr<float>();
    for (int s = 0; s < NUM_STREAMS; ++s) {
        out_ptr[s] = h_buffers[s][0];
    }

    for (auto& st : streams) {
        CUDA_CHECK(cudaStreamDestroy(st));
    }
    for (auto& host_ptr : h_buffers) {
        CUDA_CHECK(cudaFreeHost(host_ptr));
    }

    return out;
}

torch::Tensor run_stream_ordered_allocator_capture(int64_t elements, int iterations) {
    TORCH_CHECK(elements > 0, "elements must be > 0");
    TORCH_CHECK(iterations > 0, "iterations must be > 0");
    size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    std::array<cudaStream_t, NUM_STREAMS> streams{};
    for (auto& st : streams) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    }

    std::array<float*, NUM_STREAMS> h_buffers{};
    for (auto& host_ptr : h_buffers) {
        CUDA_CHECK(cudaMallocHost(&host_ptr, bytes));
        for (int64_t i = 0; i < elements; ++i) {
            host_ptr[i] = static_cast<float>(i % 1024) * 0.5f;
        }
    }

    for (int iter = 0; iter < iterations; ++iter) {
        std::array<float*, NUM_STREAMS> d_in{};
        std::array<float*, NUM_STREAMS> d_out{};

        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_in[s]), bytes, streams[s]));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_out[s]), bytes, streams[s]));

            CUDA_CHECK(cudaMemcpyAsync(d_in[s], h_buffers[s], bytes, cudaMemcpyHostToDevice, streams[s]));

            int threads = 256;
            int blocks = static_cast<int>((elements + threads - 1) / threads);
            scale_kernel<<<blocks, threads, 0, streams[s]>>>(d_in[s], d_out[s], static_cast<int>(elements));
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(h_buffers[s], d_out[s], bytes, cudaMemcpyDeviceToHost, streams[s]));
        }

        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            CUDA_CHECK(cudaFreeAsync(d_in[s], streams[s]));
            CUDA_CHECK(cudaFreeAsync(d_out[s], streams[s]));
        }
    }

    auto out = torch::empty({NUM_STREAMS}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* out_ptr = out.data_ptr<float>();
    for (int s = 0; s < NUM_STREAMS; ++s) {
        out_ptr[s] = h_buffers[s][0];
    }

    for (auto& st : streams) {
        CUDA_CHECK(cudaStreamDestroy(st));
    }
    for (auto& host_ptr : h_buffers) {
        CUDA_CHECK(cudaFreeHost(host_ptr));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_standard_allocator", &run_standard_allocator, "cudaMalloc baseline workload");
    m.def("run_stream_ordered_allocator", &run_stream_ordered_allocator, "cudaMallocAsync workload");
    m.def("run_standard_allocator_capture", &run_standard_allocator_capture, "cudaMalloc baseline workload (returns a small output slice)");
    m.def("run_stream_ordered_allocator_capture", &run_stream_ordered_allocator_capture, "cudaMallocAsync workload (returns a small output slice)");
}
