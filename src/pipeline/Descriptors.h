#pragma once

#include <volk.h>
#include <vector>
#include <mutex>

namespace vkt {

// -- DescriptorLayoutBuilder ----------------------------------------------------
// Fluent builder for VkDescriptorSetLayout.
// Usage:
//   auto layout = DescriptorLayoutBuilder{}
//       .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
//       .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
//       .build(device);
class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder& addBinding(uint32_t binding, VkDescriptorType type,
                                        VkShaderStageFlags stages, uint32_t count = 1);
    VkDescriptorSetLayout build(VkDevice device) const;

private:
    std::vector<VkDescriptorSetLayoutBinding> m_bindings;
};

// -- DescriptorPool ------------------------------------------------------------
// Manages a VkDescriptorPool using the reset-all strategy: individual sets
// cannot be freed  --  only resetPool() frees everything at once.
// This omits VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT intentionally;
// the per-frame uniform sets are never individually freed, only the whole pool
// is reset on shutdown or when doing a full scene reload.
// allocate() and resetPool() are internally synchronized for concurrent callers.
class DescriptorPool {
public:
    DescriptorPool() = default;
    ~DescriptorPool();
    DescriptorPool(const DescriptorPool&)            = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    // ratio: how many descriptors of this type to allocate per maxSets slot.
    struct PoolSizeRatio { VkDescriptorType type; float ratio; };

    void init(VkDevice device, uint32_t maxSets,
              const std::vector<PoolSizeRatio>& ratios);
    void destroy();

    // Allocate a descriptor set from this pool.
    VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    // Reset all allocations (reuse pool memory without reallocation)
    void resetPool();

private:
    VkDevice         m_device = VK_NULL_HANDLE;
    VkDescriptorPool m_pool   = VK_NULL_HANDLE;
    mutable std::mutex m_mutex;
};

// -- DescriptorWriter ----------------------------------------------------------
// Builds and submits vkUpdateDescriptorSets writes for a single descriptor set.
// Not thread-safe: construct, fill, and call update() on the same thread.
// Usage:
//   DescriptorWriter writer;
//   writer.writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, buffer, offset, range);
//   writer.writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, view, sampler, layout);
//   writer.update(device, set);
class DescriptorWriter {
public:
    DescriptorWriter& writeBuffer(uint32_t binding, VkDescriptorType type,
                                  VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);

    DescriptorWriter& writeImage(uint32_t binding, VkDescriptorType type,
                                 VkImageView view, VkSampler sampler,
                                 VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Array variant: writes descriptorCount image descriptors at one binding.
    DescriptorWriter& writeImages(uint32_t binding, VkDescriptorType type,
                                  const std::vector<VkImageView>& views, VkSampler sampler,
                                  VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    void update(VkDevice device, VkDescriptorSet set);
    void clear();

private:
    struct WriteRef {
        uint32_t bufferStart = UINT32_MAX;
        uint32_t bufferCount = 0;
        uint32_t imageStart  = UINT32_MAX;
        uint32_t imageCount  = 0;
    };

    // Stable storage for write info structs (pointers must remain valid until update())
    std::vector<VkDescriptorBufferInfo> m_bufferInfos;
    std::vector<VkDescriptorImageInfo>  m_imageInfos;
    std::vector<VkWriteDescriptorSet>   m_writes;
    std::vector<WriteRef>               m_refs;
};

} // namespace vkt
