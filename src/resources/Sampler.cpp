#include "Sampler.h"

#include <stdexcept>
#include <algorithm>

namespace vkt {

void Sampler::create(VkDevice device, VkPhysicalDevice physDevice, const Config& cfg) {
    m_device = device;

    // Clamp requested anisotropy to what the device actually supports
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physDevice, &props);

    bool  enableAnisotropy = cfg.maxAnisotropy > 1.0f;
    float maxAnisotropy    = std::min(cfg.maxAnisotropy,
                                      props.limits.maxSamplerAnisotropy);

    VkSamplerCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.minFilter               = cfg.minFilter;
    info.magFilter               = cfg.magFilter;
    info.mipmapMode              = cfg.mipmapMode;
    info.addressModeU            = cfg.addressMode;
    info.addressModeV            = cfg.addressMode;
    info.addressModeW            = cfg.addressMode;
    info.mipLodBias              = cfg.mipLodBias;
    info.anisotropyEnable        = enableAnisotropy ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy           = enableAnisotropy ? maxAnisotropy : 1.0f;
    info.compareEnable           = cfg.compareEnable ? VK_TRUE : VK_FALSE;
    info.compareOp               = cfg.compareOp;
    info.minLod                  = 0.0f;
    info.maxLod                  = cfg.maxLod;
    info.borderColor             = cfg.borderColor;
    info.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(device, &info, nullptr, &m_sampler) != VK_SUCCESS)
        throw std::runtime_error("[Sampler] Failed to create VkSampler");
}

void Sampler::destroy() {
    if (m_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
}

Sampler::~Sampler() { destroy(); }

} // namespace vkt
