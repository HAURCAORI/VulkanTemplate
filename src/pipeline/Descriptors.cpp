#include "Descriptors.h"

#include <stdexcept>
#include <cstdint>

namespace vkt {

// -- DescriptorLayoutBuilder ----------------------------------------------------

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addBinding(
    uint32_t binding, VkDescriptorType type, VkShaderStageFlags stages, uint32_t count)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding            = binding;
    b.descriptorType     = type;
    b.descriptorCount    = count;
    b.stageFlags         = stages;
    b.pImmutableSamplers = nullptr;
    m_bindings.push_back(b);
    return *this;
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device) const {
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = static_cast<uint32_t>(m_bindings.size());
    info.pBindings    = m_bindings.data();

    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("[Descriptors] Failed to create descriptor set layout");
    return layout;
}

// -- DescriptorPool ------------------------------------------------------------

void DescriptorPool::init(VkDevice device, uint32_t maxSets,
                          const std::vector<PoolSizeRatio>& ratios) {
    std::scoped_lock lock(m_mutex);
    m_device = device;

    std::vector<VkDescriptorPoolSize> sizes;
    sizes.reserve(ratios.size());
    for (auto& r : ratios) {
        sizes.push_back({r.type, static_cast<uint32_t>(r.ratio * static_cast<float>(maxSets))});
    }

    VkDescriptorPoolCreateInfo info{};
    info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets       = maxSets;
    info.poolSizeCount = static_cast<uint32_t>(sizes.size());
    info.pPoolSizes    = sizes.data();

    if (vkCreateDescriptorPool(device, &info, nullptr, &m_pool) != VK_SUCCESS)
        throw std::runtime_error("[Descriptors] Failed to create descriptor pool");
}

VkDescriptorSet DescriptorPool::allocate(VkDescriptorSetLayout layout) {
    std::scoped_lock lock(m_mutex);
    VkDescriptorSetAllocateInfo info{};
    info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool     = m_pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts        = &layout;

    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(m_device, &info, &set) != VK_SUCCESS)
        throw std::runtime_error("[Descriptors] Failed to allocate descriptor set");
    return set;
}

void DescriptorPool::resetPool() {
    std::scoped_lock lock(m_mutex);
    vkResetDescriptorPool(m_device, m_pool, 0);
}

void DescriptorPool::destroy() {
    std::scoped_lock lock(m_mutex);
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
}

DescriptorPool::~DescriptorPool() { destroy(); }

// -- DescriptorWriter ----------------------------------------------------------

DescriptorWriter& DescriptorWriter::writeBuffer(uint32_t binding, VkDescriptorType type,
                                                 VkBuffer buffer, VkDeviceSize offset,
                                                 VkDeviceSize range) {
    auto& bi = m_bufferInfos.emplace_back();
    bi.buffer       = buffer;
    bi.offset       = offset;
    bi.range        = range;

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstBinding      = binding;
    w.descriptorCount = 1;
    w.descriptorType  = type;
    m_writes.push_back(w);
    WriteRef ref{};
    ref.bufferStart = static_cast<uint32_t>(m_bufferInfos.size() - 1);
    ref.bufferCount = 1;
    m_refs.push_back(ref);
    return *this;
}

DescriptorWriter& DescriptorWriter::writeImage(uint32_t binding, VkDescriptorType type,
                                                VkImageView view, VkSampler sampler,
                                                VkImageLayout layout) {
    auto& ii = m_imageInfos.emplace_back();
    ii.imageView    = view;
    ii.sampler      = sampler;
    ii.imageLayout  = layout;

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstBinding      = binding;
    w.descriptorCount = 1;
    w.descriptorType  = type;
    m_writes.push_back(w);
    WriteRef ref{};
    ref.imageStart = static_cast<uint32_t>(m_imageInfos.size() - 1);
    ref.imageCount = 1;
    m_refs.push_back(ref);
    return *this;
}

DescriptorWriter& DescriptorWriter::writeImages(uint32_t binding, VkDescriptorType type,
                                                const std::vector<VkImageView>& views,
                                                VkSampler sampler, VkImageLayout layout) {
    if (views.empty()) return *this;

    const uint32_t start = static_cast<uint32_t>(m_imageInfos.size());
    m_imageInfos.reserve(m_imageInfos.size() + views.size());
    for (VkImageView view : views) {
        auto& ii = m_imageInfos.emplace_back();
        ii.imageView   = view;
        ii.sampler     = sampler;
        ii.imageLayout = layout;
    }

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstBinding      = binding;
    w.descriptorCount = static_cast<uint32_t>(views.size());
    w.descriptorType  = type;
    m_writes.push_back(w);

    WriteRef ref{};
    ref.imageStart = start;
    ref.imageCount = static_cast<uint32_t>(views.size());
    m_refs.push_back(ref);
    return *this;
}

void DescriptorWriter::update(VkDevice device, VkDescriptorSet set) {
    for (size_t i = 0; i < m_writes.size(); ++i) {
        auto& w = m_writes[i];
        w.dstSet = set;
        const auto& ref = m_refs[i];
        if (ref.bufferStart != UINT32_MAX)
            w.pBufferInfo = &m_bufferInfos[ref.bufferStart];
        if (ref.imageStart != UINT32_MAX)
            w.pImageInfo = &m_imageInfos[ref.imageStart];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(m_writes.size()),
                           m_writes.data(), 0, nullptr);
}

void DescriptorWriter::clear() {
    m_bufferInfos.clear();
    m_imageInfos.clear();
    m_writes.clear();
    m_refs.clear();
}

} // namespace vkt
