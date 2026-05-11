#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vkt {

// GPU timestamp query pool for per-pass timing (Track E1).
//
// One VkQueryPool per frame-in-flight, 2 query slots per pass (begin + end).
// Retrieve results after the fence for that frame slot signals.
//
// Usage:
//   init(device, physDevice, graphicsQueueFamilyIndex, framesInFlight)
//   -- per frame --
//   retrieve(device, currentFrame, timestampPeriod)  // after fence wait
//   reset(cmd, currentFrame)                         // before first pass
//   -- (RenderGraph::execute calls beginPass/endPass automatically) --
//   -- per UI frame --
//   for (uint32_t i = 0; i < resultPassCount(); ++i)
//       ImGui::Text("%s  %.3f ms", passName(i), rollingAvgMs(i));
class GpuTimestamps {
public:
    // Maximum number of passes that can be timed per frame.
    // Must be >= RenderGraph::kMaxPasses.
    static constexpr uint32_t kMaxPasses = 32;

    GpuTimestamps()  = default;
    ~GpuTimestamps() = default;
    GpuTimestamps(const GpuTimestamps&)            = delete;
    GpuTimestamps& operator=(const GpuTimestamps&) = delete;

    // Initialize query pools. Returns false and degrades gracefully if the
    // device or queue family does not support timestamp queries.
    bool init(VkDevice device, VkPhysicalDevice physDevice,
              uint32_t graphicsQueueFamilyIndex, uint32_t framesInFlight);

    // Destroy query pools. Safe to call if init() was never called.
    void destroy();

    // Returns false if the device/queue does not support timestamp queries.
    // When false, all other methods are safe but produce zero results.
    bool supported() const { return m_supported; }

    // Reset this frame slot's query pool. Must be called at the start of
    // the command buffer, outside any render pass, before the first beginPass().
    void reset(VkCommandBuffer cmd, uint32_t frameIndex);

    // Write a begin timestamp for pass 'passIndex'. Called by RenderGraph::execute().
    // 'name' is stored for display; overwritten each frame so always current.
    void beginPass(VkCommandBuffer cmd, uint32_t frameIndex,
                   uint32_t passIndex, const char* name);

    // Write an end timestamp for pass 'passIndex'. Called by RenderGraph::execute().
    void endPass(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t passIndex);

    // Read GPU results for the given frame slot. Call after vkWaitForFences()
    // and before re-recording into this slot.
    // timestampPeriod: from VkPhysicalDeviceLimits (nanoseconds per tick).
    void retrieve(VkDevice device, uint32_t frameIndex, float timestampPeriod);

    // -- Result accessors (valid after at least one retrieve()) ----------------

    // Number of passes timed in the last retrieve() call.
    uint32_t resultPassCount() const { return m_uiPassCount; }

    // GPU duration in milliseconds for pass i (from last retrieve()).
    float lastMs(uint32_t i) const {
        return (m_supported && i < kMaxPasses) ? m_lastMs[i] : 0.0f;
    }

    // Exponential moving average GPU duration (alpha = 0.1).
    float rollingAvgMs(uint32_t i) const {
        return (m_supported && i < kMaxPasses) ? m_rollingAvg[i] : 0.0f;
    }

    // Name of pass i as set by the most recent beginPass() call.
    const char* passName(uint32_t i) const {
        return (i < kMaxPasses) ? m_names[i].c_str() : "";
    }

private:
    bool     m_supported      = false;
    uint32_t m_framesInFlight = 0;
    uint32_t m_maxPasses      = 0;   // actual maxPasses passed to init()
    uint32_t m_uiPassCount    = 0;   // pass count from last retrieve() -- used by UI

    VkDevice m_device = VK_NULL_HANDLE;

    // One query pool per frame-in-flight.
    std::vector<VkQueryPool> m_pools;

    // Per-frame-slot pass count (updated by reset/beginPass, used by retrieve).
    std::vector<uint32_t> m_slotPassCount;

    // Per-pass data (indexed by passIndex, not frameIndex).
    std::array<std::string, kMaxPasses> m_names;
    std::array<float,       kMaxPasses> m_lastMs{};
    std::array<float,       kMaxPasses> m_rollingAvg{};
};

} // namespace vkt
