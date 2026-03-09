#pragma once

#include "scene/CullObject.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace vkt {

// GPU compute culling pass (cull.comp).
//
// Each frame:
//   1. CPU fills the per-frame CullObject SSBO (one entry per alive scene object).
//   2. CPU fills the per-frame IndirectBuffer (indexCount, firstIndex, vertexOffset,
//      firstInstance per batch; instanceCount = 0).
//   3. CPU flushes both buffers and inserts a HOST -> COMPUTE pipeline barrier.
//   4. dispatch() binds the compute pipeline and issues vkCmdDispatch.
//   5. The cull.comp shader:
//        - Tests each object's world-space bounding sphere against the 6 frustum planes.
//        - For visible objects: atomicAdd on indirectCmds[batchId].instanceCount to
//          claim a slot, then writes InstanceData to instanceSSBO[firstInstance + slot].
//   6. A COMPUTE_WRITE -> VERTEX_SHADER + DRAW_INDIRECT barrier before the geometry pass.
//   7. vkCmdDrawIndexedIndirect per batch uses GPU-computed instanceCount.
//
// Compute descriptor set (separate layout from the global graphics set):
//   set = 0, binding 0: CullObject SSBO    (per-object input, CPU-written each frame)
//   set = 0, binding 1: InstanceData SSBO  (per-instance output, GPU-written by compute)
//   set = 0, binding 2: Indirect SSBO      (VkDrawIndexedIndirectCommand, GPU atomicAdd)
//
// Push constants (112 bytes, compute stage only):
//   vec4  planes[6]   (96 B): world-space frustum planes (from Frustum::fromVP)
//   uint  objectCount  (4 B): number of valid CullObject entries this frame
//   uint  _pad[3]     (12 B): alignment
class CullPass {
public:
    CullPass() = default;
    ~CullPass() { destroy(); }
    CullPass(const CullPass&)            = delete;
    CullPass& operator=(const CullPass&) = delete;

    // maxObjects:     capacity of each per-frame CullObject buffer.
    // framesInFlight: number of frame slots.
    void init(VkDevice device, VmaAllocator allocator,
              const std::string& compSpvPath,
              uint32_t maxObjects, uint32_t framesInFlight);

    // Bind the InstanceData and Indirect SSBOs for a frame slot.
    // Must be called once per frame slot after init(), and again if buffer handles change.
    void updateDescriptors(uint32_t frameIndex,
                           VkBuffer instanceSSBO,  VkDeviceSize instanceSize,
                           VkBuffer indirectSSBO,  VkDeviceSize indirectSize);

    // Flush the frame's CullObject buffer and dispatch the compute cull shader.
    // frustumPlanes: 6 world-space planes from Frustum::fromVP().data().
    // objectCount:   number of valid entries written to objectData(frameIndex).
    void dispatch(VkCommandBuffer cmd, uint32_t frameIndex,
                  const glm::vec4* frustumPlanes, uint32_t objectCount);

    void destroy();

    // Per-frame CullObject buffer (CPU-writable).
    // Fill objectData(frame)[0 .. N-1] before dispatch().
    CullObject* objectData(uint32_t frameIndex) const;
    VkBuffer    objectBuffer(uint32_t frameIndex) const;
    uint32_t    objectCapacity() const { return m_objectCapacity; }

private:
    struct FrameObjects {
        VkBuffer      buffer     = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        CullObject*   mapped     = nullptr;
    };

    void createPipeline(VkDevice device, const std::string& compSpvPath);
    void createDescriptorPool(VkDevice device, uint32_t framesInFlight);
    void createObjectBuffers(uint32_t framesInFlight);

    VkDevice     m_device    = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    VkPipeline            m_pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool           = VK_NULL_HANDLE;

    std::vector<VkDescriptorSet> m_sets;        // one per frame slot
    std::vector<FrameObjects>    m_frameObjects; // one per frame slot

    uint32_t m_objectCapacity = 0;
    uint32_t m_framesInFlight = 0;
};

} // namespace vkt
