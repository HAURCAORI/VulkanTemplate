#pragma once

#include <volk.h>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

#include <string_view>

namespace vkt {

// Owns the core Vulkan objects needed by every other system.
// Call init() once before any other vkt object; call destroy() (or let the
// destructor run) only after vkDeviceWaitIdle has been called by the caller.
// Not thread-safe: all methods must be called from one thread.
class Context {
public:
    // Vulkan 1.2 is the minimum for timeline semaphores, buffer device address,
    // and the structured feature query structs used by vk-bootstrap.
    static constexpr uint32_t MIN_API_VERSION = VK_API_VERSION_1_2;

    struct CreateInfo {
        std::string_view appName    = "VulkanTemplate";
        bool             validation = false; // enable via VKT_VALIDATION in Debug builds
    };

    Context() = default;
    ~Context();
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    void init(GLFWwindow* window, const CreateInfo& info = {});
    void destroy();

    // -- Handles ---------------------------------------------------------------
    VkInstance       instance()       const { return m_instance.instance; }
    VkPhysicalDevice physicalDevice() const { return m_physDevice.physical_device; }
    VkDevice         device()         const { return m_device.device; }
    VkSurfaceKHR     surface()        const { return m_surface; }

    // -- Queues ----------------------------------------------------------------
    // transferQueue() may alias graphicsQueue() on devices with only one family.
    VkQueue  graphicsQueue()      const { return m_graphicsQueue; }
    VkQueue  presentQueue()       const { return m_presentQueue; }
    VkQueue  transferQueue()      const { return m_transferQueue; }
    uint32_t graphicsQueueIndex() const { return m_graphicsQueueIndex; }
    uint32_t presentQueueIndex()  const { return m_presentQueueIndex; }
    uint32_t transferQueueIndex() const { return m_transferQueueIndex; }

    // -- Info ------------------------------------------------------------------
    const vkb::PhysicalDevice& physDeviceInfo() const { return m_physDevice; }
    const char*                deviceName()     const { return m_physDevice.name.c_str(); }

private:
    vkb::Instance       m_instance{};
    vkb::PhysicalDevice m_physDevice{};
    vkb::Device         m_device{};
    VkSurfaceKHR        m_surface = VK_NULL_HANDLE;

    VkQueue  m_graphicsQueue      = VK_NULL_HANDLE;
    VkQueue  m_presentQueue       = VK_NULL_HANDLE;
    VkQueue  m_transferQueue      = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueIndex = 0;
    uint32_t m_presentQueueIndex  = 0;
    uint32_t m_transferQueueIndex = 0;

    bool m_initialized = false;
};

} // namespace vkt
