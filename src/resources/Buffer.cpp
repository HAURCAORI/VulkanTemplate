#include "Buffer.h"

#include <vk_mem_alloc.h>

#include <cstring>
#include <stdexcept>

namespace vkt {

// -- Move semantics -------------------------------------------------------------

Buffer::Buffer(Buffer&& o) noexcept
    : m_allocator(VK_NULL_HANDLE), m_buffer(VK_NULL_HANDLE),
      m_allocation(VK_NULL_HANDLE), m_size(0), m_mapped(nullptr)
{
    std::scoped_lock lock(m_mutex, o.m_mutex);
    m_allocator  = o.m_allocator;
    m_buffer     = o.m_buffer;
    m_allocation = o.m_allocation;
    m_size       = o.m_size;
    m_mapped     = o.m_mapped;
    o.m_buffer = VK_NULL_HANDLE;
    o.m_allocation = VK_NULL_HANDLE;
    o.m_mapped = nullptr;
    o.m_size = 0;
    o.m_allocator = VK_NULL_HANDLE;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        std::scoped_lock lock(m_mutex, o.m_mutex);
        if (m_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
            m_buffer     = VK_NULL_HANDLE;
            m_allocation = VK_NULL_HANDLE;
            m_mapped     = nullptr;
        }
        m_allocator  = o.m_allocator;
        m_buffer     = o.m_buffer;
        m_allocation = o.m_allocation;
        m_size       = o.m_size;
        m_mapped     = o.m_mapped;
        o.m_buffer     = VK_NULL_HANDLE;
        o.m_allocation = VK_NULL_HANDLE;
        o.m_mapped     = nullptr;
        o.m_size       = 0;
        o.m_allocator  = VK_NULL_HANDLE;
    }
    return *this;
}

// -- GPU-only -------------------------------------------------------------------

void Buffer::createGpuOnly(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage) {
    std::scoped_lock lock(m_mutex);
    m_allocator = allocator;
    m_size      = size;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = size;
    bufInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage         = VMA_MEMORY_USAGE_AUTO;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &m_buffer, &m_allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("[Buffer] Failed to create GPU-only buffer");
}

// -- CPU-visible ----------------------------------------------------------------

void Buffer::createCpuVisible(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage) {
    std::scoped_lock lock(m_mutex);
    m_allocator = allocator;
    m_size      = size;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = size;
    bufInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT; // persistent map

    VmaAllocationInfo info;
    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &m_buffer, &m_allocation, &info) != VK_SUCCESS)
        throw std::runtime_error("[Buffer] Failed to create CPU-visible buffer");

    m_mapped = info.pMappedData;
}

void Buffer::uploadData(const void* data, VkDeviceSize size) {
    std::scoped_lock lock(m_mutex);
    if (!m_mapped)
        throw std::runtime_error("[Buffer] uploadData called on non-mapped buffer");
    if (size > m_size)
        throw std::runtime_error("[Buffer] uploadData size exceeds allocation");
    std::memcpy(m_mapped, data, static_cast<size_t>(size));
    // vmaFlushAllocation is a no-op on HOST_COHERENT memory (typical on desktop)
    // but required on non-coherent devices (some ARM/integrated GPUs) to make
    // CPU writes visible to the GPU without an explicit cache flush instruction.
    vmaFlushAllocation(m_allocator, m_allocation, 0, size);
}

void Buffer::destroy() {
    std::scoped_lock lock(m_mutex);
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        m_buffer     = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
        m_mapped     = nullptr;
    }
}

Buffer::~Buffer() { destroy(); }

// -- Staging upload helper ------------------------------------------------------

void uploadToGpuBuffer(VmaAllocator allocator, VkCommandBuffer cmd,
                       Buffer& dst, const void* data, VkDeviceSize size,
                       Buffer& outStaging) {
    outStaging.createCpuVisible(allocator, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    outStaging.uploadData(data, size);

    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size      = size;
    vkCmdCopyBuffer(cmd, outStaging.handle(), dst.handle(), 1, &copy);
    // outStaging must remain alive until the caller submits cmd and the GPU finishes.
}

} // namespace vkt

