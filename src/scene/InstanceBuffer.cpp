#include "InstanceBuffer.h"

#include <stdexcept>

namespace vkt {

void InstanceBuffer::init(VmaAllocator allocator, uint32_t maxInstances) {
    m_allocator = allocator;
    m_capacity  = maxInstances;

    const VkDeviceSize size = static_cast<VkDeviceSize>(maxInstances) * sizeof(InstanceData);

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo,
                        &m_buffer, &m_allocation, &info) != VK_SUCCESS)
        throw std::runtime_error("[InstanceBuffer] Failed to create SSBO");

    m_mapped = static_cast<InstanceData*>(info.pMappedData);
}

void InstanceBuffer::flush(uint32_t instanceCount) {
    if (m_allocation == VK_NULL_HANDLE || instanceCount == 0) return;
    const VkDeviceSize size = static_cast<VkDeviceSize>(instanceCount) * sizeof(InstanceData);
    // No-op on HOST_COHERENT memory; required on non-coherent devices (ARM/some iGPUs).
    vmaFlushAllocation(m_allocator, m_allocation, 0, size);
}

void InstanceBuffer::destroy() {
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        m_buffer     = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
        m_mapped     = nullptr;
    }
    m_capacity  = 0;
    m_allocator = VK_NULL_HANDLE;
}

} // namespace vkt
