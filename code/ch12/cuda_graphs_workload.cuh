#pragma once

struct StageSpec {
    float scale;
    float bias;
    float frequency;
};

inline constexpr StageSpec kStageSpecs[] = {
    {1.01f, 0.05f, 0.10f},
    {0.97f, -0.02f, 0.25f},
    {1.08f, 0.12f, 0.40f},
    {0.92f, -0.04f, 0.60f},
    {1.04f, 0.03f, 0.85f},
    {0.99f, -0.08f, 1.05f},
};

inline constexpr int kStageCount = sizeof(kStageSpecs) / sizeof(StageSpec);
inline constexpr int kInnerPasses = 3;
