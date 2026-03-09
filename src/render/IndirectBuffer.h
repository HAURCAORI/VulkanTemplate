#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>

namespace vkt {

// Persistently-mapped per-frame buffer of VkDrawIndexedIndirectCommand entries.
//
// Buffer usage flags:
//   STORAGE_BUFFER  -- compute shader reads static fields; atomicAdd fills instanceCount.
//   INDIRECT_BUFFER -- vkCmdDrawIndexedIndirect reads the final draw parameters.
//
// Per-frame usage pattern:
//   1. CPU fills all fields for each batch (indexCount, firstIndex, vertexOffset,
//      firstInstance).  Sets instanceCount = 0.
//   2. flush() makes CPU writes visible to the GPU before the compute dispatch.
//   3. Compute shader atomicAdd-increments instanceCount for each visible object.
//   4. vkCmdDrawIndexedIndirect reads the completed commands from handle().
class IndirectBuffer {
public:
    IndirectBuffer() = default;
    ~IndirectBuffer() { destroy(); }
    IndirectBuffer(const IndirectBuffer&)            = delete;
    IndirectBuffer& operator=(const IndirectBuffer&) = delete;

    void init(VmaAllocator allocator, uint32_t maxBatches);

    // Flush CPU writes for the first batchCount entries.
    // Must be called after CPU fill and before the HOST -> COMPUTE pipeline barrier.
    void flush(uint32_t batchCount);

    void destroy();

    VkDrawIndexedIndirectCommand* data()     const { return m_mapped; }
    VkBuffer                      handle()   const { return m_buffer; }
    uint32_t                      capacity() const { return m_capacity; }

private:
    VmaAllocator                   m_allocator  = VK_NULL_HANDLE;
    VkBuffer                       m_buffer     = VK_NULL_HANDLE;
    VmaAllocation                  m_allocation = VK_NULL_HANDLE;
    VkDrawIndexedIndirectCommand*  m_mapped     = nullptr;
    uint32_t                       m_capacity   = 0;
};

} // namespace vkt
