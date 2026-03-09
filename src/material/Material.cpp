#include "Material.h"

#include <stdexcept>
#include <cstring>

namespace vkt {

void MaterialManager::init(VmaAllocator allocator, uint32_t maxMaterials) {
    m_allocator = allocator;
    m_capacity  = maxMaterials > 0 ? maxMaterials : 1;

    const VkDeviceSize size = static_cast<VkDeviceSize>(m_capacity) * sizeof(GPUMaterial);

    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size  = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo,
                        &m_buffer, &m_allocation, &info) != VK_SUCCESS)
        throw std::runtime_error("[MaterialManager] Failed to create material SSBO");

    m_mapped = static_cast<GPUMaterial*>(info.pMappedData);
    if (!m_mapped)
        throw std::runtime_error("[MaterialManager] Material SSBO is not persistently mapped");

    // Zero-init the entire buffer to avoid garbage reads on first frame.
    std::memset(m_mapped, 0, static_cast<size_t>(size));

    // Register the default material at index 0.
    // All objects default to materialId=0 (white opaque, Blinn-Phong lit).
    m_count = 0;
    add("default", GPUMaterial{});
    flush(); // ensure GPU sees the default before the first frame
}

void MaterialManager::destroy() {
    if (m_allocator == nullptr) return;
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        m_buffer     = VK_NULL_HANDLE;
        m_allocation = nullptr;
        m_mapped     = nullptr;
    }
    m_nameIndex.clear();
    m_capacity   = 0;
    m_count      = 0;
    m_dirty      = false;
    m_allocator  = nullptr;
}

uint32_t MaterialManager::add(std::string name, const GPUMaterial& mat) {
    // Reject duplicate names; return the existing handle.
    if (auto it = m_nameIndex.find(name); it != m_nameIndex.end())
        return it->second;

    if (m_count >= m_capacity) {
        // Silently fall back to the default rather than corrupting memory.
        return kDefaultId;
    }

    const uint32_t id  = m_count++;
    m_mapped[id]       = mat;
    m_nameIndex[std::move(name)] = id;
    m_dirty = true;
    return id;
}

uint32_t MaterialManager::add(const GPUMaterial& mat) {
    if (m_count >= m_capacity) return kDefaultId;
    const uint32_t id  = m_count++;
    m_mapped[id]       = mat;
    m_dirty = true;
    return id;
}

void MaterialManager::update(uint32_t id, const GPUMaterial& mat) {
    if (id >= m_count) return;
    m_mapped[id] = mat;
    m_dirty = true;
}

GPUMaterial& MaterialManager::get(uint32_t id) {
    // Out-of-range access falls back to the default material (index 0).
    return m_mapped[id < m_count ? id : kDefaultId];
}

const GPUMaterial& MaterialManager::get(uint32_t id) const {
    return m_mapped[id < m_count ? id : kDefaultId];
}

uint32_t MaterialManager::find(std::string_view name) const {
    const auto it = m_nameIndex.find(std::string(name));
    return it != m_nameIndex.end() ? it->second : kDefaultId;
}

void MaterialManager::flush() {
    if (!m_dirty || m_allocation == nullptr) return;
    const VkDeviceSize size = static_cast<VkDeviceSize>(m_count) * sizeof(GPUMaterial);
    vmaFlushAllocation(m_allocator, m_allocation, 0, size);
    m_dirty = false;
}

} // namespace vkt
