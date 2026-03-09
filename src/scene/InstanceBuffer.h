#pragma once

#include "InstanceData.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>

namespace vkt {

// Persistently-mapped SSBO that holds per-instance data for one frame in flight.
// One InstanceBuffer per frame slot; the Application cycles between them to avoid
// writing to the buffer while the GPU is still reading the previous frame's data.
//
// Usage:
//   // Init (once, at startup):
//   ib.init(allocator, RenderLimits::kMaxInstances);
//
//   // Each frame:
//   InstanceData* dst = ib.data();
//   // ... write instance records into dst[0..n-1] ...
//   ib.flush(n); // make CPU writes visible to GPU before vkQueueSubmit
//
//   // Shutdown (before allocator is destroyed):
//   ib.destroy();
class InstanceBuffer {
public:
    InstanceBuffer() = default;
    ~InstanceBuffer() { destroy(); }
    InstanceBuffer(const InstanceBuffer&)            = delete;
    InstanceBuffer& operator=(const InstanceBuffer&) = delete;

    void init(VmaAllocator allocator, uint32_t maxInstances);

    // Flush CPU writes for the first instanceCount records.
    // Must be called after writing instance data and before vkQueueSubmit.
    // On HOST_COHERENT devices this is a no-op; on others it issues a cache flush.
    void flush(uint32_t instanceCount);

    void destroy();

    InstanceData* data()     const { return m_mapped; }
    VkBuffer      handle()   const { return m_buffer; }
    uint32_t      capacity() const { return m_capacity; }

private:
    VmaAllocator  m_allocator  = VK_NULL_HANDLE;
    VkBuffer      m_buffer     = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    InstanceData* m_mapped     = nullptr;
    uint32_t      m_capacity   = 0;
};

} // namespace vkt
