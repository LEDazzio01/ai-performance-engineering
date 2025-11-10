#pragma once

#include <algorithm>
#include <vector>

struct UnevenSegment {
    int offset;
    int length;
};

static inline std::vector<UnevenSegment> build_uneven_segments(int elems) {
    constexpr int kBase = 384;
    constexpr int kMin = 128;
    constexpr int kMax = 1024;

    std::vector<UnevenSegment> segments;
    segments.reserve(elems / kMin + 8);

    int cursor = 0;
    int pattern = 0;
    while (cursor < elems) {
        const int mod = (pattern * 23) % 11;
        int swing = mod * 64;
        int span = kBase + ((pattern & 1) ? -swing : swing);
        if ((pattern % 19) == 0) {
            span += 384;
        }
        span = std::max(kMin, std::min(kMax, span));

        UnevenSegment seg;
        seg.offset = cursor;
        seg.length = std::min(span, elems - cursor);
        segments.push_back(seg);

        cursor += seg.length;
        ++pattern;
    }

    return segments;
}
