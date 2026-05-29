#include "Context.h"

#include <stdexcept>
#include <cstdio>

namespace vkt {

void Context::init(GLFWwindow* window, const CreateInfo& info) {
    // -- 1. Initialize volk (loads vkGetInstanceProcAddr from the Vulkan loader) --
    if (volkInitialize() != VK_SUCCESS)
        throw std::runtime_error("[Context] Failed to initialize volk -- is the Vulkan loader installed?");

    // -- 2. Instance -----------------------------------------------------------
    vkb::InstanceBuilder builder;
    auto instResult = builder
        .set_app_name(info.appName.data())
        .require_api_version(MIN_API_VERSION)
        .request_validation_layers(info.validation)
        .use_default_debug_messenger()
        .build();

    if (!instResult)
        throw std::runtime_error("[Context] Instance creation failed: " + instResult.error().message());

    m_instance = instResult.value();

    // volkLoadInstance populates instance-level entry points (vkCreateDevice,
    // vkEnumeratePhysicalDevices, etc.).  volkLoadDevice (called after device
    // creation) separately populates device-level entry points so each device
    // gets its own dispatch table  --  required when using multiple VkDevices.
    volkLoadInstance(m_instance.instance);

    // -- 3. Surface (via GLFW) -------------------------------------------------
    // Surface must exist before physical device selection so vk-bootstrap can
    // verify presentation support and swapchain format compatibility.
    if (glfwCreateWindowSurface(m_instance.instance, window, nullptr, &m_surface) != VK_SUCCESS)
        throw std::runtime_error("[Context] Failed to create window surface");

    // -- 4. Physical device ----------------------------------------------------
    // Require Vulkan 1.2; request discrete GPU when available.
    // Users can add feature requirements here (e.g., raytracing, mesh shaders).
    VkPhysicalDeviceFeatures requiredFeatures{};
    requiredFeatures.samplerAnisotropy = VK_TRUE;
    requiredFeatures.fillModeNonSolid  = VK_TRUE; // optional for wireframe rendering

    auto selectPhysical = [&](const VkPhysicalDeviceFeatures& features) {
        // Pass surface in the constructor (v1.4 API) to avoid the .set_surface() chain call.
        vkb::PhysicalDeviceSelector selector{m_instance, m_surface};
        return selector
            .set_minimum_version(1, 2)
            .set_required_features(features)
            .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
            .select();
    };

    auto physResult = selectPhysical(requiredFeatures);
    if (!physResult) {
        // Retry without fillModeNonSolid to maximize portability for baseline rendering.
        VkPhysicalDeviceFeatures portableFeatures{};
        portableFeatures.samplerAnisotropy = VK_TRUE;
        physResult = selectPhysical(portableFeatures);
        if (!physResult) {
            // Use detailed_failure_reasons() (v1.4 API) for a per-device diagnostic.
            std::string msg = "[Context] Physical device selection failed: " + physResult.error().message();
            for (const auto& reason : physResult.full_error().detailed_failure_reasons)
                msg += "\n  - " + reason;
            throw std::runtime_error(msg);
        }
        std::printf("[Context] fillModeNonSolid unsupported; wireframe is disabled.\n");
    }

    m_physDevice = physResult.value();

    // -- 5. Logical device -----------------------------------------------------
    // Query descriptor indexing features (Vulkan 1.2 core; required for bindless textures).
    // Forward the hardware-reported struct directly: only VK_TRUE fields get enabled.
    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    {
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &indexingFeatures;
        vkGetPhysicalDeviceFeatures2(m_physDevice.physical_device, &f2);
    }

    vkb::DeviceBuilder deviceBuilder{m_physDevice};
    deviceBuilder.add_pNext(&indexingFeatures);
    auto devResult = deviceBuilder.build();

    if (!devResult)
        throw std::runtime_error("[Context] Device creation failed: " + devResult.error().message());

    m_device = devResult.value();

    volkLoadDevice(m_device.device);

    // -- 6. Queues -------------------------------------------------------------
    // get_queue_and_index() (v1.4 API) returns both in one call, avoiding two
    // separate lookups and the risk of them returning inconsistent results.
    auto gqi = m_device.get_queue_and_index(vkb::QueueType::graphics);
    if (!gqi) throw std::runtime_error("[Context] No graphics queue: " + gqi.error().message());
    m_graphicsQueue      = gqi.value().first;
    m_graphicsQueueIndex = gqi.value().second;

    auto pqi = m_device.get_queue_and_index(vkb::QueueType::present);
    if (!pqi) throw std::runtime_error("[Context] No present queue: " + pqi.error().message());
    m_presentQueue      = pqi.value().first;
    m_presentQueueIndex = pqi.value().second;

    // A dedicated transfer queue enables DMA overlap with graphics work.
    // Fall back to graphics when unavailable  --  correct but loses the overlap.
    auto tqi = m_device.get_queue_and_index(vkb::QueueType::transfer);
    m_transferQueue      = tqi ? tqi.value().first  : m_graphicsQueue;
    m_transferQueueIndex = tqi ? tqi.value().second : m_graphicsQueueIndex;

    m_initialized = true;
    std::printf("[Context] GPU: %s\n", m_physDevice.name.c_str());
}

void Context::destroy() {
    if (!m_initialized) return;
    vkb::destroy_device(m_device);
    vkb::destroy_surface(m_instance, m_surface);
    vkb::destroy_instance(m_instance);
    m_initialized = false;
}

Context::~Context() { destroy(); }

} // namespace vkt
