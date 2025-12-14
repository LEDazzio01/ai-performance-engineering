/**
 * cuda_verify.cuh - Header for CUDA verify mode
 * 
 * Build Contract:
 *   - Pass -DVERIFY=1 to nvcc for verify builds ONLY
 *   - Perf builds MUST NOT include -DVERIFY=1
 *   - Verified by symbol inspection (nm/objdump)
 * 
 * Runner Contract:
 *   - Harness provides fixed input buffer
 *   - Checksum returned via stdout or buffer
 *   - Output format: "VERIFY_CHECKSUM: <float>\n"
 * 
 * Usage:
 *   #include "cuda_verify.cuh"
 *   
 *   __global__ void myKernel(float* output, int size) {
 *       // ... kernel logic ...
 *       
 *       // At end of kernel, emit checksum for verification
 *       float checksum = 0.0f;
 *       VERIFY_CHECKSUM(output, size, &checksum);
 *   }
 *   
 *   // In host code after kernel:
 *   VERIFY_PRINT_CHECKSUM(checksum);
 */

#ifndef CUDA_VERIFY_CUH
#define CUDA_VERIFY_CUH

#include <cmath>
#include <cstring>
#include <cstdio>

// =============================================================================
// Device-side checksum computation
// =============================================================================

#ifdef VERIFY

/**
 * VERIFY_CHECKSUM - Compute checksum of buffer on device
 * 
 * Computes sum of all elements in buffer. Only active when VERIFY=1.
 * In perf mode (no VERIFY flag), this is a no-op with zero overhead.
 * 
 * @param buffer     Pointer to data buffer
 * @param size       Number of elements
 * @param checksum   Pointer to output checksum value
 */
#define VERIFY_CHECKSUM(buffer, size, checksum_out) \
    do { \
        float _verify_sum = 0.0f; \
        for (int _i = 0; _i < (size); _i++) { \
            _verify_sum += static_cast<float>((buffer)[_i]); \
        } \
        *(checksum_out) = _verify_sum; \
    } while(0)

/**
 * VERIFY_CHECKSUM_ATOMIC - Thread-safe checksum with atomicAdd
 * 
 * For use in parallel kernels where multiple threads contribute.
 * Uses atomicAdd for thread safety.
 */
#define VERIFY_CHECKSUM_ATOMIC(buffer, start_idx, count, global_checksum) \
    do { \
        float _local_sum = 0.0f; \
        for (int _i = 0; _i < (count); _i++) { \
            _local_sum += static_cast<float>((buffer)[(start_idx) + _i]); \
        } \
        atomicAdd((global_checksum), _local_sum); \
    } while(0)

/**
 * VERIFY_MAX - Compute max absolute value of buffer
 * 
 * Useful for checking kernel output bounds.
 */
#define VERIFY_MAX(buffer, size, max_out) \
    do { \
        float _verify_max = 0.0f; \
        for (int _i = 0; _i < (size); _i++) { \
            float _val = fabsf(static_cast<float>((buffer)[_i])); \
            if (_val > _verify_max) _verify_max = _val; \
        } \
        *(max_out) = _verify_max; \
    } while(0)

/**
 * VERIFY_PRINT_CHECKSUM - Print checksum to stdout
 * 
 * Called from host code after copying checksum back from device.
 * Format: "VERIFY_CHECKSUM: <float>\n"
 */
#define VERIFY_PRINT_CHECKSUM(checksum) \
    do { \
        printf("VERIFY_CHECKSUM: %.10e\n", (checksum)); \
        fflush(stdout); \
    } while(0)

/**
 * VERIFY_PRINT_CHECKSUMS - Print multiple named checksums
 * 
 * For kernels with multiple outputs.
 * Format: "VERIFY_CHECKSUM_<name>: <float>\n"
 */
#define VERIFY_PRINT_NAMED_CHECKSUM(name, checksum) \
    do { \
        printf("VERIFY_CHECKSUM_%s: %.10e\n", (name), (checksum)); \
        fflush(stdout); \
    } while(0)

/**
 * VERIFY_GUARD - Execute code block only in verify mode
 */
#define VERIFY_GUARD(code_block) code_block

/**
 * VERIFY_ONLY - Mark variable/buffer as verify-only
 */
#define VERIFY_ONLY(decl) decl

#else  // !VERIFY - Perf path: all macros become no-ops

#define VERIFY_CHECKSUM(buffer, size, checksum_out) ((void)0)
#define VERIFY_CHECKSUM_ATOMIC(buffer, start_idx, count, global_checksum) ((void)0)
#define VERIFY_MAX(buffer, size, max_out) ((void)0)
#define VERIFY_PRINT_CHECKSUM(checksum) ((void)0)
#define VERIFY_PRINT_NAMED_CHECKSUM(name, checksum) ((void)0)
#define VERIFY_GUARD(code_block) ((void)0)
#define VERIFY_ONLY(decl) ((void)0)

#endif  // VERIFY

// =============================================================================
// Host-side utilities
// =============================================================================

namespace cuda_verify {

/**
 * Parse VERIFY_CHECKSUM from stdout string
 * 
 * @param stdout_str  Output from CUDA binary execution
 * @param checksum    Output parameter for parsed checksum
 * @return true if checksum found and parsed successfully
 */
inline bool parse_checksum(const char* stdout_str, float* checksum) {
    const char* prefix = "VERIFY_CHECKSUM: ";
    const char* pos = strstr(stdout_str, prefix);
    if (pos == nullptr) return false;
    pos += strlen(prefix);
    return sscanf(pos, "%e", checksum) == 1;
}

/**
 * Compare two checksums within tolerance
 * 
 * @param baseline    Baseline checksum
 * @param optimized   Optimized checksum
 * @param rtol        Relative tolerance (default: 1e-5)
 * @param atol        Absolute tolerance (default: 1e-8)
 * @return true if checksums match within tolerance
 */
inline bool compare_checksums(float baseline, float optimized, 
                               float rtol = 1e-5f, float atol = 1e-8f) {
    float diff = fabsf(baseline - optimized);
    float tol = atol + rtol * fabsf(baseline);
    return diff <= tol;
}

}  // namespace cuda_verify

#endif  // CUDA_VERIFY_CUH






