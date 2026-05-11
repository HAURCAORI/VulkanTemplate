#include "render/GpuTimestamps.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>

namespace vkt {

bool GpuTimestamps::init(VkDevice device, VkPhysicalDevice physDevice,
                         uint32_t graphicsQueueFamilyIndex, uint32_t framesInFlight) {
    m_device         = device;
    m_framesInFlight = framesInFlight;
    m_maxPasses      = kMaxPasses;

    // Check timestamp period -- a zero value means device does not support timestamps.
    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(physDevice, &devProps);
    if (devProps.limits.timestampPeriod == 0.0f) {
        std::printf("[GpuTimestamps] Device does not support timestamp queries (period=0).\n");
        m_supported = false;
        return false;
    }

    // Check that the graphics queue family supports timestamp queries.
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueProps(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, queueProps.data());

    if (graphicsQueueFamilyIndex >= queueFamilyCount ||
        queueProps[graphicsQueueFamilyIndex].timestampValidBits == 0) {
        std::printf("[GpuTimestamps] Graphics queue family does not support timestamp queries.\n");
        m_supported = false;
        return false;
    }

    // Create one query pool per frame-in-flight.
    // Each pool holds kMaxPasses * 2 slots (begin + end per pass).
    const uint32_t queryCount = kMaxPasses * 2u;
    m_pools.resize(framesInFlight, VK_NULL_HANDLE);
    m_slotPassCount.assign(framesInFlight, 0u);

    VkQueryPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    poolInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = queryCount;

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        if (vkCreateQueryPool(device, &poolInfo, nullptr, &m_pools[i]) != VK_SUCCESS) {
            // Non-fatal: disable timestamps and clean up pools created so far.
            std::printf("[GpuTimestamps] Failed to create query pool (frame %u). "
                        "Timestamps disabled.\n", i);
            for (uint32_t j = 0; j < i; ++j) {
                vkDestroyQueryPool(device, m_pools[j], nullptr);
                m_pools[j] = VK_NULL_HANDLE;
            }
            m_pools.clear();
            m_supported = false;
            return false;
        }
    }

    m_names.fill({});
    m_lastMs.fill(0.0f);
    m_rollingAvg.fill(0.0f);

    m_supported = true;
    return true;
}

void GpuTimestamps::destroy() {
    if (m_device == VK_NULL_HANDLE) return;
    for (auto pool : m_pools)
        if (pool != VK_NULL_HANDLE)
            vkDestroyQueryPool(m_device, pool, nullptr);
    m_pools.clear();
    m_slotPassCount.clear();
    m_supported = false;
    m_device    = VK_NULL_HANDLE;
}

void GpuTimestamps::reset(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!m_supported) return;
    assert(frameIndex < m_framesInFlight);
    vkCmdResetQueryPool(cmd, m_pools[frameIndex], 0u, kMaxPasses * 2u);
    m_slotPassCount[frameIndex] = 0u;
}

void GpuTimestamps::beginPass(VkCommandBuffer cmd, uint32_t frameIndex,
                              uint32_t passIndex, const char* name) {
    if (!m_supported) return;
    assert(frameIndex < m_framesInFlight);
    assert(passIndex < kMaxPasses);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        m_pools[frameIndex], passIndex * 2u);

    if (passIndex + 1u > m_slotPassCount[frameIndex])
        m_slotPassCount[frameIndex] = passIndex + 1u;

    if (name)
        m_names[passIndex] = name;
}

void GpuTimestamps::endPass(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t passIndex) {
    if (!m_supported) return;
    assert(frameIndex < m_framesInFlight);
    assert(passIndex < kMaxPasses);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        m_pools[frameIndex], passIndex * 2u + 1u);
}

void GpuTimestamps::retrieve(VkDevice device, uint32_t frameIndex, float timestampPeriod) {
    if (!m_supported) return;
    assert(frameIndex < m_framesInFlight);

    const uint32_t passCount = m_slotPassCount[frameIndex];
    if (passCount == 0u) return;

    const uint32_t queryCount = passCount * 2u;
    // Allocate on the stack for small pass counts; heap for safety.
    std::vector<uint64_t> results(queryCount, 0u);

    // VK_QUERY_RESULT_WAIT_BIT: safe because we already waited for the fence.
    VkResult r = vkGetQueryPoolResults(
        device, m_pools[frameIndex],
        0u, queryCount,
        queryCount * sizeof(uint64_t), results.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (r != VK_SUCCESS) return;  // VK_NOT_READY on very first frame is harmless

    // Convert to milliseconds and update rolling average (EMA, alpha=0.1).
    const float nsToMs = timestampPeriod * 1e-6f;
    static constexpr float kAlpha = 0.10f;

    for (uint32_t i = 0; i < passCount; ++i) {
        const uint64_t t0 = results[i * 2u];
        const uint64_t t1 = results[i * 2u + 1u];
        const float ms = (t1 >= t0) ? static_cast<float>(t1 - t0) * nsToMs : 0.0f;
        m_lastMs[i]     = ms;
        m_rollingAvg[i] = m_rollingAvg[i] * (1.0f - kAlpha) + ms * kAlpha;
    }

    m_uiPassCount = passCount;
}

} // namespace vkt
