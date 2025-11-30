// optimized_ilp.cu
// CUDA 13 + Blackwell: Float8 vectorization with 8-way ILP

#include "optimized_ilp_low_occupancy_vec4_impl.cuh"

int main() {
    constexpr int kDefaultCap = 48;
    return run_ilp_low_occupancy_vec4(
        "Optimized ILP (Float8, 8-way, 48 active blocks)",
        kDefaultCap);
}
