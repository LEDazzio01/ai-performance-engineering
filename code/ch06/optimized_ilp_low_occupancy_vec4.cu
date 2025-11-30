// optimized_ilp_low_occupancy_vec4.cu
// CUDA 13 + Blackwell: Float8 with low occupancy for higher ILP

#include "optimized_ilp_low_occupancy_vec4_impl.cuh"

int main() {
    constexpr int kLowOccupancyCap = 32;
    return run_ilp_low_occupancy_vec4(
        "Optimized ILP (Float8, 8-way, 32 active blocks)",
        kLowOccupancyCap);
}
