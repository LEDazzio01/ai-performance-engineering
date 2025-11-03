// profiling_helpers.cuh - NVTX markers and profiling utilities
// Integrates with Nsight Systems for visual timeline profiling

#ifndef PROFILING_HELPERS_CUH
#define PROFILING_HELPERS_CUH

#include <nvtx3/nvToolsExt.h>
#include <cstdint>

// Color palette for NVTX markers
namespace nvtx {
  constexpr uint32_t COLOR_RED     = 0xFFFF0000;
  constexpr uint32_t COLOR_GREEN   = 0xFF00FF00;
  constexpr uint32_t COLOR_BLUE    = 0xFF0000FF;
  constexpr uint32_t COLOR_YELLOW  = 0xFFFFFF00;
  constexpr uint32_t COLOR_CYAN    = 0xFF00FFFF;
  constexpr uint32_t COLOR_MAGENTA = 0xFFFF00FF;
  constexpr uint32_t COLOR_WHITE   = 0xFFFFFFFF;
  constexpr uint32_t COLOR_ORANGE  = 0xFFFFA500;
  constexpr uint32_t COLOR_PURPLE  = 0xFF800080;
}

// NVTX Range RAII wrapper
class NvtxRange {
private:
  nvtxRangeId_t range_id;

public:
  NvtxRange(const char* name, uint32_t color = nvtx::COLOR_GREEN) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    range_id = nvtxRangeStartEx(&eventAttrib);
  }

  ~NvtxRange() {
    nvtxRangeEnd(range_id);
  }

  // Prevent copying
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

// Convenience macros for profiling sections
#define NVTX_RANGE(name) NvtxRange nvtx_##__LINE__(name)
#define NVTX_RANGE_COLOR(name, color) NvtxRange nvtx_##__LINE__(name, color)

// Mark specific operations for profiling
#define NVTX_MARK_COMPUTE(name) nvtxMarkA(name)
#define NVTX_MARK_MEMORY(name) nvtxMarkA(name)

// Common profiling ranges
#define PROFILE_KERNEL_LAUNCH(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_BLUE)
#define PROFILE_MEMORY_COPY(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_YELLOW)
#define PROFILE_HOST_COMPUTE(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_CYAN)
#define PROFILE_DATA_PREP(name) NVTX_RANGE_COLOR(name, nvtx::COLOR_MAGENTA)

#endif // PROFILING_HELPERS_CUH

