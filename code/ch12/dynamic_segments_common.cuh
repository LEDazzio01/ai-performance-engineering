#pragma once

#include <algorithm>
#include <vector>

struct Segment {
    int offset;
    int length;
    float scale;
};

static inline std::vector<Segment> build_segments(int elements) {
    constexpr int kBaseSpan = 1024;
    constexpr int kMinSpan = 192;
    constexpr int kMaxSpan = 2048;

    std::vector<Segment> segments;
    segments.reserve(elements / kMinSpan + 8);

    int cursor = 0;
    int pattern = 0;
    while (cursor < elements) {
        const int swing = (pattern % 7) * 96;
        int span = kBaseSpan + ((pattern & 1) ? -swing : swing);
        span = std::max(kMinSpan, std::min(kMaxSpan, span));

        Segment seg;
        seg.offset = cursor;
        seg.length = std::min(span, elements - cursor);
        seg.scale = 1.0f + 0.0008f * static_cast<float>((pattern * 17) % 29);
        segments.push_back(seg);

        cursor += seg.length;
        ++pattern;
    }

    return segments;
}
