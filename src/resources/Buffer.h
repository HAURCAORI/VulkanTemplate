#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <mutex>

namespace vkt {

// RAII wrapper around a VkBuffer backed by a VmaAllocation.
// Move-only: the mutex makes copy semantics meaningless.
class Buffer {
public:
    Buffer() = default;
    ~Buffer();
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept;
    Buffer& operator=(Buffer&& o) noexcept;

    // -- Creation helpers ------------------------------------------------------

    // Device-local buffer (fastest for GPU reads). Data must be uploaded via
    // a staging buffer; the TRANSFER_DST usage flag is added automatically.
    void createGpuOnly(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage);

    // Persistently-mapped, host-writable buffer. Suitable for uniform buffers
    // and staging.  Uses HOST_ACCESS_SEQUENTIAL_WRITE for optimal placement.
    void createCpuVisible(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage);

    // memcpy data into the persistent mapping and flush for GPU visibility.
    // Only valid on CPU-visible buffers (asserts m_mapped != nullptr).
    void uploadData(const void* data, VkDeviceSize size);

    void destroy();

    VkBuffer      handle()     const noexcept { return m_buffer; }
    VkDeviceSize  size()       const noexcept { return m_size; }
    void*         mappedPtr()  const noexcept { return m_mapped; } // null if not persistently mapped

private:
    VmaAllocator  m_allocator  = VK_NULL_HANDLE;
    VkBuffer      m_buffer     = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkDeviceSize  m_size       = 0;
    void*         m_mapped     = nullptr;
    mutable std::mutex m_mutex;
};

// Records a vkCmdCopyBuffer from a freshly-created staging buffer into dst.
// outStaging is populated by this call; the caller must keep it alive until the
// command buffer has finished executing, then destroy it (or let it go out of scope).
void uploadToGpuBuffer(VmaAllocator allocator, VkCommandBuffer cmd,
                       Buffer& dst, const void* data, VkDeviceSize size,
                       Buffer& outStaging);

} // namespace vkt
