#include "pipeline/BindlessTextureSet.h"

#include <cstdio>
#include <stdexcept>

namespace vkt {

bool BindlessTextureSet::init(VkDevice device, uint32_t maxTextures) {
    m_device      = device;
    m_maxTextures = maxTextures;

    // Binding flags: allow partial population, GPU updates, and variable count.
    VkDescriptorBindingFlags bindingFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT             |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT           |
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount  = 1;
    flagsInfo.pBindingFlags = &bindingFlags;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = maxTextures;
    binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext        = &flagsInfo;
    layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &binding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_layout) != VK_SUCCESS)
        throw std::runtime_error("[BindlessTextureSet] Failed to create descriptor set layout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = maxTextures;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_pool) != VK_SUCCESS)
        throw std::runtime_error("[BindlessTextureSet] Failed to create descriptor pool");

    // Allocate a variable-count descriptor set.
    VkDescriptorSetVariableDescriptorCountAllocateInfo varCountInfo{};
    varCountInfo.sType              =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    varCountInfo.descriptorSetCount = 1;
    varCountInfo.pDescriptorCounts  = &maxTextures;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pNext              = &varCountInfo;
    allocInfo.descriptorPool     = m_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_layout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &m_set) != VK_SUCCESS)
        throw std::runtime_error("[BindlessTextureSet] Failed to allocate descriptor set");

    std::printf("[BindlessTextureSet] Initialized; capacity=%u.\n", maxTextures);
    return true;
}

void BindlessTextureSet::destroy() {
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
        m_set  = VK_NULL_HANDLE;
    }
    if (m_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_layout, nullptr);
        m_layout = VK_NULL_HANDLE;
    }
    m_device      = VK_NULL_HANDLE;
    m_count       = 0;
    m_maxTextures = 0;
}

uint32_t BindlessTextureSet::registerTexture(VkDevice device, VkImageView view, VkSampler sampler) {
    if (m_count >= m_maxTextures) {
        std::printf("[BindlessTextureSet] Full (capacity=%u); clamping to last slot.\n",
                    m_maxTextures);
        return m_maxTextures - 1;
    }
    const uint32_t idx = m_count++;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView   = view;
    imageInfo.sampler     = sampler;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = m_set;
    write.dstBinding      = 0;
    write.dstArrayElement = idx;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    return idx;
}

} // namespace vkt
