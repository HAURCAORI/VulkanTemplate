#include "CullPass.h"
#include "config/RenderLimits.h"
#include "pipeline/Shader.h"

#include <cassert>
#include <stdexcept>

namespace vkt {

// Push constant layout for cull.comp -- must exactly match the shader.
// Total: 6*16 + 4 + 12 = 112 bytes  (fits in the 128-byte Vulkan guarantee).
struct CullPushConstants {
    glm::vec4 planes[6];   // 96 bytes: world-space frustum planes
    uint32_t  objectCount; //  4 bytes: number of valid CullObject entries
    uint32_t  _pad[3];     // 12 bytes: alignment to 16-byte boundary
};
static_assert(sizeof(CullPushConstants) == 112,
    "CullPushConstants size mismatch -- update push_constant block in cull.comp");

// ---------------------------------------------------------------------------

void CullPass::init(VkDevice device, VmaAllocator allocator,
                    const std::string& compSpvPath,
                    uint32_t maxObjects, uint32_t framesInFlight) {
    if (m_pipeline != VK_NULL_HANDLE) return; // re-init guard: already initialized

    m_device         = device;
    m_allocator      = allocator;
    m_objectCapacity = maxObjects;
    m_framesInFlight = framesInFlight;

    createPipeline(device, compSpvPath);
    createDescriptorPool(device, framesInFlight);
    createObjectBuffers(framesInFlight);

    // Allocate one descriptor set per frame slot.
    m_sets.resize(framesInFlight);
    std::vector<VkDescriptorSetLayout> layouts(framesInFlight, m_setLayout);
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool     = m_pool;
    allocInfo.descriptorSetCount = framesInFlight;
    allocInfo.pSetLayouts        = layouts.data();
    if (vkAllocateDescriptorSets(device, &allocInfo, m_sets.data()) != VK_SUCCESS)
        throw std::runtime_error("[CullPass] Failed to allocate descriptor sets");

    // Bind the CullObject buffer (binding 0) immediately -- it never changes.
    // Bindings 1 (instance) and 2 (indirect) are set later via updateDescriptors().
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = m_frameObjects[i].buffer;
        bufInfo.offset = 0;
        bufInfo.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = m_sets[i];
        w.dstBinding      = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &bufInfo;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
}

// ---------------------------------------------------------------------------

void CullPass::createPipeline(VkDevice device, const std::string& compSpvPath) {
    // Descriptor set layout: 3 storage-buffer bindings (all compute-stage).
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding         = static_cast<uint32_t>(i);
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutCI.bindingCount = 3;
    layoutCI.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_setLayout) != VK_SUCCESS)
        throw std::runtime_error("[CullPass] Failed to create descriptor set layout");

    // Push constant range (compute stage only).
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(CullPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCI.setLayoutCount         = 1;
    pipelineLayoutCI.pSetLayouts            = &m_setLayout;
    pipelineLayoutCI.pushConstantRangeCount = 1;
    pipelineLayoutCI.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &m_pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("[CullPass] Failed to create pipeline layout");

    // Load compute shader and build compute pipeline.
    Shader comp;
    comp.load(device, compSpvPath);

    VkPipelineShaderStageCreateInfo stage = comp.stageInfo(VK_SHADER_STAGE_COMPUTE_BIT);

    VkComputePipelineCreateInfo pipelineCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineCI.stage  = stage;
    pipelineCI.layout = m_pipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_pipeline) != VK_SUCCESS)
        throw std::runtime_error("[CullPass] Failed to create compute pipeline");

    // comp destructor destroys the VkShaderModule (safe -- pipeline retains bytecode).
}

void CullPass::createDescriptorPool(VkDevice device, uint32_t framesInFlight) {
    // 3 STORAGE_BUFFER descriptors per frame slot (bindings 0, 1, 2).
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3 * framesInFlight;

    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets       = framesInFlight;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &m_pool) != VK_SUCCESS)
        throw std::runtime_error("[CullPass] Failed to create descriptor pool");
}

void CullPass::createObjectBuffers(uint32_t framesInFlight) {
    m_frameObjects.resize(framesInFlight);
    const VkDeviceSize size =
        static_cast<VkDeviceSize>(m_objectCapacity) * sizeof(CullObject);

    for (auto& fo : m_frameObjects) {
        VkBufferCreateInfo bufCI{};
        bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufCI.size  = size;
        bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO;
        allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo info{};
        if (vmaCreateBuffer(m_allocator, &bufCI, &allocCI,
                            &fo.buffer, &fo.allocation, &info) != VK_SUCCESS)
            throw std::runtime_error("[CullPass] Failed to create CullObject buffer");
        fo.mapped = static_cast<CullObject*>(info.pMappedData);
    }
}

// ---------------------------------------------------------------------------

void CullPass::updateDescriptors(uint32_t frameIndex,
                                  VkBuffer instanceSSBO,  VkDeviceSize instanceSize,
                                  VkBuffer indirectSSBO,  VkDeviceSize indirectSize) {
    assert(frameIndex < m_framesInFlight && "CullPass::updateDescriptors: frameIndex out of range");
    VkDescriptorBufferInfo instInfo{instanceSSBO, 0, instanceSize};
    VkDescriptorBufferInfo indrInfo{indirectSSBO, 0, indirectSize};

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = m_sets[frameIndex];
    writes[0].dstBinding      = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo     = &instInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = m_sets[frameIndex];
    writes[1].dstBinding      = 2;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo     = &indrInfo;

    vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);
}

// ---------------------------------------------------------------------------

void CullPass::dispatch(VkCommandBuffer cmd, uint32_t frameIndex,
                         const glm::vec4* frustumPlanes, uint32_t objectCount) {
    assert(frameIndex < m_framesInFlight && "CullPass::dispatch: frameIndex out of range");
    assert(objectCount <= m_objectCapacity  && "CullPass::dispatch: objectCount exceeds buffer capacity");
    if (objectCount == 0) return;

    // Flush the CullObject buffer so the GPU sees this frame's CPU writes.
    vmaFlushAllocation(m_allocator, m_frameObjects[frameIndex].allocation,
                       0, static_cast<VkDeviceSize>(objectCount) * sizeof(CullObject));

    CullPushConstants pc{};
    for (int i = 0; i < 6; ++i) pc.planes[i] = frustumPlanes[i];
    pc.objectCount = objectCount;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipelineLayout, 0, 1, &m_sets[frameIndex], 0, nullptr);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(CullPushConstants), &pc);

    // One thread per object; kCullWorkgroupSize threads per workgroup (matches local_size_x in cull.comp).
    const uint32_t wgSize = RenderLimits::kCullWorkgroupSize;
    const uint32_t groups = (objectCount + wgSize - 1u) / wgSize;
    vkCmdDispatch(cmd, groups, 1, 1);
}

// ---------------------------------------------------------------------------

void CullPass::destroy() {
    for (auto& fo : m_frameObjects) {
        if (fo.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator, fo.buffer, fo.allocation);
            fo.buffer     = VK_NULL_HANDLE;
            fo.allocation = VK_NULL_HANDLE;
            fo.mapped     = nullptr;
        }
    }
    m_frameObjects.clear();
    m_sets.clear();

    if (m_pipeline       != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_pool           != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    if (m_setLayout      != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_setLayout, nullptr);
        m_setLayout = VK_NULL_HANDLE;
    }

    m_device    = VK_NULL_HANDLE;
    m_allocator = VK_NULL_HANDLE;
}

CullObject* CullPass::objectData(uint32_t frameIndex) const {
    return frameIndex < m_frameObjects.size() ? m_frameObjects[frameIndex].mapped : nullptr;
}

VkBuffer CullPass::objectBuffer(uint32_t frameIndex) const {
    return frameIndex < m_frameObjects.size() ? m_frameObjects[frameIndex].buffer : VK_NULL_HANDLE;
}

} // namespace vkt
