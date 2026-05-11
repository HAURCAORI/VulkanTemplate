#pragma once

#include <volk.h>
#include <cstdint>

namespace vkt {

// Manages a variable-count sampler2D[] descriptor array for bindless texture access.
// Requires VkPhysicalDeviceDescriptorIndexingFeatures:
//   descriptorBindingPartiallyBound, runtimeDescriptorArray,
//   shaderSampledImageArrayNonUniformIndexing, descriptorBindingVariableDescriptorCount.
//
// Usage (G1-G3):
//   bindless.init(device, maxTextures);
//   uint32_t idx = bindless.registerTexture(device, view, sampler);
//   // bind bindless.handle() to set=1 with the bindless pipeline layout
//   // shader: texture(bindlessTextures[nonuniformEXT(mat.albedoTexIdx)], uv)
class BindlessTextureSet {
public:
    static constexpr uint32_t kDefaultIdx  = 0;
    static constexpr uint32_t kMaxTextures = 1024;

    BindlessTextureSet()  = default;
    ~BindlessTextureSet() { destroy(); }
    BindlessTextureSet(const BindlessTextureSet&)            = delete;
    BindlessTextureSet& operator=(const BindlessTextureSet&) = delete;

    // Create pool, layout, and descriptor set.
    // The first registerTexture() call must be the default (white) placeholder.
    bool     init(VkDevice device, uint32_t maxTextures = kMaxTextures);
    void     destroy();

    // Write a texture into the next free slot. Returns its bindless index.
    // Returns (maxTextures-1) and logs a warning when the array is full.
    // Not thread-safe; call from the main thread during scene setup.
    uint32_t registerTexture(VkDevice device, VkImageView view, VkSampler sampler);

    VkDescriptorSet       handle() const { return m_set; }
    VkDescriptorSetLayout layout() const { return m_layout; }
    uint32_t              count()  const { return m_count; }
    bool                  valid()  const { return m_set != VK_NULL_HANDLE; }

private:
    VkDevice              m_device      = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool        = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_layout      = VK_NULL_HANDLE;
    VkDescriptorSet       m_set         = VK_NULL_HANDLE;
    uint32_t              m_maxTextures = 0;
    uint32_t              m_count       = 0;
};

} // namespace vkt
