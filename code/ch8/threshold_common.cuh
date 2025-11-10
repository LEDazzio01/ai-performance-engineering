#pragma once

#include <cuda_runtime.h>

namespace ch8 {

constexpr int kThresholdBaselineThreads = 32;
constexpr int kThresholdOptimizedThreads = 512;
constexpr float kThresholdSecondaryScale = 1.5f;
constexpr float kThresholdInnerScale = 0.85f;
constexpr float kThresholdOuterScale = 1.25f;

__device__ __forceinline__ float expensive_transform(float value, float sine, float cosine) {
    // Match baseline math path without affecting final output magnitude.
    return fabsf(value) + (sine * cosine) * 0.0001f;
}

__device__ __forceinline__ float evaluate_branch_with_redundancy(float value, float scale) {
    const float sine = __sinf(value);
    const float cosine = __cosf(value);
    const float magnitude = expensive_transform(value, sine, cosine);
    const float repeated = expensive_transform(value, sine, cosine);
    const float averaged = 0.5f * (magnitude + repeated);

    const float sine_neg = __sinf(-value);
    const float cosine_neg = __cosf(-value);
    volatile float redundant_eval = expensive_transform(-value, sine_neg, cosine_neg);

    float checksum = value * 0.5f;
    checksum = fmaf(checksum, 0.99991f, 0.0001f * value);
    checksum = fmaf(checksum, 0.99973f, -0.0001f * value);

    const float blended = averaged + (redundant_eval - redundant_eval) + (checksum - checksum);
    return blended * scale;
}

__device__ __forceinline__ float compute_threshold_scale(float value, float threshold) {
    const float abs_value = fabsf(value);
    if (abs_value <= threshold) {
        return 0.0f;
    }
    const bool outer = abs_value > threshold * kThresholdSecondaryScale;
    const float base = outer ? kThresholdOuterScale : kThresholdInnerScale;
    return value >= 0.0f ? base : -base;
}

__device__ __forceinline__ float transform_with_scale(float value, float threshold) {
    const float scale = compute_threshold_scale(value, threshold);
    if (scale == 0.0f) {
        return 0.0f;
    }
    float sine, cosine;
    __sincosf(value, &sine, &cosine);
    return expensive_transform(value, sine, cosine) * scale;
}

__global__ void threshold_naive_kernel(
    const float* __restrict__ inputs,
    float* __restrict__ output,
    float threshold,
    int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const volatile float* volatile_inputs = reinterpret_cast<const volatile float*>(inputs);
    const float branch_value = volatile_inputs[idx];
    float result = 0.0f;
    const float outer_threshold = threshold * kThresholdSecondaryScale;
    if (branch_value > outer_threshold) {
        const float value = volatile_inputs[idx];  // Naive path reloads from memory for the actual math.
        float accum = 0.0f;
        #pragma unroll 4
        for (int repeat = 0; repeat < 4; ++repeat) {
            accum += evaluate_branch_with_redundancy(value, kThresholdOuterScale);
        }
        result = 0.25f * accum;
    } else if (branch_value > threshold) {
        const float value = volatile_inputs[idx];
        float accum = 0.0f;
        #pragma unroll 4
        for (int repeat = 0; repeat < 4; ++repeat) {
            accum += evaluate_branch_with_redundancy(value, kThresholdInnerScale);
        }
        result = 0.25f * accum;
    } else if (branch_value < -outer_threshold) {
        const float value = volatile_inputs[idx];
        float accum = 0.0f;
        #pragma unroll 4
        for (int repeat = 0; repeat < 4; ++repeat) {
            accum += evaluate_branch_with_redundancy(value, -kThresholdOuterScale);
        }
        result = 0.25f * accum;
    } else if (branch_value < -threshold) {
        const float value = volatile_inputs[idx];
        float accum = 0.0f;
        #pragma unroll 4
        for (int repeat = 0; repeat < 4; ++repeat) {
            accum += evaluate_branch_with_redundancy(value, -kThresholdInnerScale);
        }
        result = 0.25f * accum;
    }
    output[idx] = result;
}

__global__ void threshold_predicated_fallback_kernel(
    const float* __restrict__ inputs,
    float* __restrict__ output,
    float threshold,
    int chunks,
    int count) {
    const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = chunk * 4;
    if (chunk >= chunks) {
        return;
    }

    if (base + 3 < count) {
        const float4* src = reinterpret_cast<const float4*>(inputs);
        float4 vec = src[chunk];
        float4 result;
        result.x = transform_with_scale(vec.x, threshold);
        result.y = transform_with_scale(vec.y, threshold);
        result.z = transform_with_scale(vec.z, threshold);
        result.w = transform_with_scale(vec.w, threshold);

        reinterpret_cast<float4*>(output)[chunk] = result;
    } else {
        for (int i = base; i < count; ++i) {
            output[i] = transform_with_scale(inputs[i], threshold);
        }
    }
}

inline int threshold_chunks(int count) {
    return (count + 3) / 4;
}

inline void launch_threshold_naive(
    const float* inputs,
    float* output,
    float threshold,
    int count,
    cudaStream_t stream) {
    const dim3 block(kThresholdBaselineThreads);
    const dim3 grid((count + block.x - 1) / block.x);
    threshold_naive_kernel<<<grid, block, 0, stream>>>(
        inputs,
        output,
        threshold,
        count);
}

inline void launch_threshold_predicated(
    const float* inputs,
    float* output,
    float threshold,
    int count,
    cudaStream_t stream) {
    const dim3 block(kThresholdOptimizedThreads);
    const int chunks = threshold_chunks(count);
    const dim3 grid((chunks + block.x - 1) / block.x);
    threshold_predicated_fallback_kernel<<<grid, block, 0, stream>>>(
        inputs,
        output,
        threshold,
        chunks,
        count);
}

}  // namespace ch8
