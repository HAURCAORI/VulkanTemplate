// VMA implementation -- compiled in exactly one translation unit.
#define VMA_IMPLEMENTATION
// VMA resolves Vulkan entry points at runtime through vkGetInstanceProcAddr /
// vkGetDeviceProcAddr.  Because volk has already loaded them, this works with
// VK_NO_PROTOTYPES (no static Vulkan symbols needed).
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include "Allocator.h"
#include "Context.h"

#include <stdexcept>

namespace vkt {

void Allocator::init(const Context& ctx) {
    VmaVulkanFunctions vmaFunctions{};
    vmaFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFunctions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo info{};
    info.vulkanApiVersion = VK_API_VERSION_1_2;
    info.instance         = ctx.instance();
    info.physicalDevice   = ctx.physicalDevice();
    info.device           = ctx.device();
    info.pVulkanFunctions = &vmaFunctions;
    // VK_KHR_dedicated_allocation is core in Vulkan 1.1+; VMA auto-detects it.
    // VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT is not needed for 1.2+.

    if (vmaCreateAllocator(&info, &m_allocator) != VK_SUCCESS)
        throw std::runtime_error("[Allocator] Failed to create VmaAllocator");
}

void Allocator::destroy() {
    if (m_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
}

Allocator::~Allocator() { destroy(); }

} // namespace vkt
