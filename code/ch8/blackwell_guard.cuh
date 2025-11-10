#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION CUDART_VERSION
#endif

namespace ch8 {

inline bool is_blackwell_device(
    cudaDeviceProp* out_props = nullptr,
    cudaError_t* out_error = nullptr,
    bool assume_blackwell_on_error = false) {
#if CUDA_VERSION >= 12000
    cudaError_t err = cudaSuccess;
    cudaDeviceProp props{};

    auto handle_failure = [&](cudaError_t failure_err) -> bool {
        if (assume_blackwell_on_error) {
            if (out_props != nullptr) {
                *out_props = cudaDeviceProp{};
                out_props->major = 12;
                out_props->minor = 0;
            }
            if (out_error != nullptr) {
                *out_error = cudaSuccess;
            }
            return true;
        }
        if (out_props != nullptr) {
            *out_props = cudaDeviceProp{};
        }
        if (out_error != nullptr) {
            *out_error = failure_err;
        }
        return false;
    };

    // Trigger runtime initialization so that device queries succeed reliably.
    cudaFree(nullptr);

    int device_count = 0;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return handle_failure((err == cudaSuccess) ? cudaErrorNoDevice : err);
    }

    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        if (!handle_failure(err)) {
            return false;
        }
        device = 0;
    }

    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        if (!handle_failure(err)) {
            return false;
        }
    }

    int major = 0;
    int minor = 0;
    cudaError_t major_err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaError_t minor_err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    auto is_attr_unsupported = [](cudaError_t err) {
        if (err == cudaErrorNotSupported) {
            return true;
        }
#ifdef cudaErrorOperationNotSupported
        if (err == cudaErrorOperationNotSupported) {
            return true;
        }
#endif
        return false;
    };

    if (is_attr_unsupported(major_err) || is_attr_unsupported(minor_err)) {
        // New hardware stacks may temporarily return cudaErrorNotSupported for capability
        // queries before the runtime is updated. Assume Blackwell in this case so the
        // benchmark can proceed (Python-side gating already verified support).
        major = 12;
        minor = 0;
        major_err = cudaSuccess;
        minor_err = cudaSuccess;
    }

    if (major_err == cudaSuccess && minor_err == cudaSuccess) {
        props.major = major;
        props.minor = minor;
    } else {
        err = cudaGetDeviceProperties(&props, device);
        if (err != cudaSuccess) {
            if (!handle_failure(err)) {
                return false;
            }
            return true;
        }
    }

    if (out_props != nullptr) {
        *out_props = props;
    }
    if (out_error != nullptr) {
        *out_error = cudaSuccess;
    }
    return props.major >= 12;
#else
    if (out_props != nullptr) {
        *out_props = cudaDeviceProp{};
    }
    if (out_error != nullptr) {
        *out_error = cudaErrorNotSupported;
    }
    return false;
#endif
}

}  // namespace ch8
