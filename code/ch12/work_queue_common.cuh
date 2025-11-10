#pragma once

#include <vector>

static inline std::vector<int> build_workloads(int n) {
    std::vector<int> work(n);
    const int heavy_tail = n / 8;
    const int medium_stride = std::max(1, n / 48);
    for (int i = 0; i < n; ++i) {
        int value = 24 + (i % 7) * 5;
        if (i >= n - heavy_tail) {
            value = 1024 + (i & 255) * 4;
        } else if ((i % medium_stride) == 0) {
            value = 320 + (i & 63) * 6;
        } else if ((i % 113) == 0) {
            value = 96;
        }
        work[i] = value;
    }
    return work;
}
