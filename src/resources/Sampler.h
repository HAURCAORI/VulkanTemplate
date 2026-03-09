#pragma once

#include <volk.h>

namespace vkt {

// RAII wrapper around VkSampler.
// Clamps requested anisotropy to the device limit at creation time so callers
// can safely pass 16.0f on any hardware.
class Sampler {
public:
    struct Config {
        VkFilter             minFilter   = VK_FILTER_LINEAR;
        VkFilter             magFilter   = VK_FILTER_LINEAR;
        VkSamplerMipmapMode  mipmapMode  = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        float                maxAnisotropy = 16.0f; // clamped to device limit; 0 = disabled
        float                maxLod        = VK_LOD_CLAMP_NONE;
        float                mipLodBias    = 0.0f;
        VkBorderColor        borderColor   = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        bool                 compareEnable = false;
        VkCompareOp          compareOp     = VK_COMPARE_OP_ALWAYS;
    };

    Sampler() = default;
    ~Sampler();
    Sampler(const Sampler&)            = delete;
    Sampler& operator=(const Sampler&) = delete;

    // physDevice is needed to query maxSamplerAnisotropy from device limits.
    void create(VkDevice device, VkPhysicalDevice physDevice, const Config& cfg = {});
    void destroy();

    VkSampler handle() const { return m_sampler; }

    // -- Config presets --------------------------------------------------------

    // Linear filtering + trilinear mipmaps + max anisotropy. Good default.
    static Config linearConfig(float maxLod = VK_LOD_CLAMP_NONE) {
        return Config{VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                      VK_SAMPLER_MIPMAP_MODE_LINEAR,
                      VK_SAMPLER_ADDRESS_MODE_REPEAT, 16.0f, maxLod};
    }

    // Point (nearest) filtering, no mipmaps. For pixel-art or UI textures.
    static Config nearestConfig() {
        return Config{VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                      VK_SAMPLER_MIPMAP_MODE_NEAREST,
                      VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.0f, 0.0f};
    }

    // Linear, clamp to edge, no anisotropy. Suitable for render targets / post-fx.
    static Config clampConfig() {
        return Config{VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                      VK_SAMPLER_MIPMAP_MODE_LINEAR,
                      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f, VK_LOD_CLAMP_NONE};
    }

    // For shadow map sampling: linear, clamp-to-border (white = lit), no comparison.
    // The fragment shader does manual 3x3 PCF with explicit depth comparison.
    static Config shadowConfig() {
        Config c{};
        c.minFilter     = VK_FILTER_LINEAR;
        c.magFilter     = VK_FILTER_LINEAR;
        c.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        c.addressMode   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        c.maxAnisotropy = 0.0f; // disabled
        c.maxLod        = 0.0f;
        c.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // depth=1.0 at border = fully lit
        c.compareEnable = false;
        c.compareOp     = VK_COMPARE_OP_ALWAYS;
        return c;
    }

private:
    VkDevice  m_device  = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
};

} // namespace vkt
