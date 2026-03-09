#include "IndirectBuffer.h"

#include <stdexcept>

namespace vkt {

void IndirectBuffer::init(VmaAllocator allocator, uint32_t maxBatches) {
    m_allocator = allocator;
    m_capacity  = maxBatches;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = static_cast<VkDeviceSize>(maxBatches) * sizeof(VkDrawIndexedIndirectCommand);
    // STORAGE_BUFFER: compute shader reads / atomicAdd.
    // INDIRECT_BUFFER: vkCmdDrawIndexedIndirect.
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &m_buffer, &m_allocation, &info) != VK_SUCCESS)
        throw std::runtime_error("[IndirectBuffer] Failed to create indirect draw buffer");

    m_mapped = static_cast<VkDrawIndexedIndirectCommand*>(info.pMappedData);
}

void IndirectBuffer::flush(uint32_t batchCount) {
    if (m_allocation == VK_NULL_HANDLE || batchCount == 0) return;
    const VkDeviceSize size =
        static_cast<VkDeviceSize>(batchCount) * sizeof(VkDrawIndexedIndirectCommand);
    vmaFlushAllocation(m_allocator, m_allocation, 0, size);
}

void IndirectBuffer::destroy() {
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
