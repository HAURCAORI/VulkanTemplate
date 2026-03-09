#pragma once

#include "config/RenderLimits.h"
#include <volk.h>
#include <array>

namespace vkt {

inline constexpr uint32_t MAX_FRAMES_IN_FLIGHT = RenderLimits::kMaxFramesInFlight;

// Resources that must be unique per frame-in-flight so the CPU can record
// frame N+1 while the GPU is still executing frame N.
struct FrameData {
    VkCommandPool   commandPool   = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    // Signaled by vkAcquireNextImageKHR when the swapchain image is safe to write.
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;

    // NOTE: renderFinished semaphores are per-swapchain-image, not per frame slot.
    // The presentation engine holds the semaphore until the image is re-acquired;
    // reusing a frame-slot semaphore on a triple-buffered swapchain violates
    // VUID-vkQueueSubmit-pSignalSemaphores-00067.  See Application for details.

    // CPU blocks on this fence before recording into this slot again.
    VkFence inFlightFence = VK_NULL_HANDLE;
};

using FrameArray = std::array<FrameData, MAX_FRAMES_IN_FLIGHT>;

} // namespace vkt
