// optimized_ilp_extreme_low_occupancy_vec4.cu
// CUDA 13 + Blackwell: Float8 with extreme low occupancy for maximum ILP

#include "optimized_ilp_low_occupancy_vec4_impl.cuh"

int main() {
    constexpr int kExtremeCap = 16;
    return run_ilp_low_occupancy_vec4(
        "Optimized ILP (Float8, 8-way, 16 active blocks - extreme)",
        kExtremeCap);
}
