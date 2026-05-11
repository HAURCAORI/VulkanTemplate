#include "SkyPass.h"

#include "pipeline/Descriptors.h"
#include "pipeline/PipelineBuilder.h"
#include "pipeline/Shader.h"

#include <stdexcept>
#include <cstring>

namespace vkt {

// Push constants written before each sky draw: 32 bytes.
struct SkyPushConstants {
    glm::vec4 skyColor;   // gradient top color (or unused for cubemap mode)
    int32_t   skyMode;    // 0=gradient, 1=cubemap
    int32_t   _pad0;
    int32_t   _pad1;
    int32_t   _pad2;
};
static_assert(sizeof(SkyPushConstants) == 32, "SkyPushConstants size mismatch");

// -- Helpers -------------------------------------------------------------------

// Transition a raw VkImage (created without the Image wrapper) from undefined
// to TRANSFER_DST and then to SHADER_READ_ONLY using a simple submit on the
// same command buffer that will be used for the placeholder cubemap upload.
// This is a self-contained helper that issues barriers + copy within one block.

static void transitionRaw(VkCommandBuffer cmd, VkImage image,
                           VkImageLayout oldLayout, VkImageLayout newLayout,
                           uint32_t layerCount) {
    VkImageMemoryBarrier barrier{};
    barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                   = oldLayout;
    barrier.newLayout                   = newLayout;
    barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                       = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = layerCount;

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
}

// -- SkyPass -------------------------------------------------------------------

void SkyPass::init(VkDevice device,
                   VmaAllocator allocator,
                   VkDescriptorSetLayout globalLayout,
                   VkDescriptorPool descriptorPool,
                   VkPipelineCache cache,
                   VkRenderPass hdrRenderPass,
                   VkSampleCountFlagBits samples,
                   const std::string& vertSpv,
                   const std::string& fragSpv,
                   uint32_t graphicsQueueFamily,
                   const Config& cfg) {
    if (m_device != VK_NULL_HANDLE) return;  // re-init guard

    m_device    = device;
    m_allocator = allocator;
    m_cfg       = cfg;

    createSkyLayout(device);
    createPlaceholderCubemap(device, allocator, descriptorPool, graphicsQueueFamily);
    createPipeline(device, hdrRenderPass, samples, cache, globalLayout, vertSpv, fragSpv);
}

void SkyPass::createSkyLayout(VkDevice device) {
    m_skyLayout = DescriptorLayoutBuilder{}
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
        .build(device);
}

void SkyPass::createPlaceholderCubemap(VkDevice device, VmaAllocator allocator,
                                        VkDescriptorPool descriptorPool,
                                        uint32_t graphicsQueueFamily) {
    // 1x1 black cubemap (RGBA8_UNORM, 6 faces).
    constexpr uint32_t kFaces = 6;
    constexpr uint32_t kBytesPerFace = 4;  // 1x1 RGBA8
    constexpr uint32_t kTotalBytes   = kFaces * kBytesPerFace;

    // Create the cubemap image.
    VkImageCreateInfo imgInfo{};
    imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imgInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgInfo.format        = VK_FORMAT_R8G8B8A8_UNORM;
    imgInfo.extent        = {1, 1, 1};
    imgInfo.mipLevels     = 1;
    imgInfo.arrayLayers   = kFaces;
    imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage         = VMA_MEMORY_USAGE_AUTO;
    allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateImage(allocator, &imgInfo, &allocCI,
                       &m_placeholderCube, &m_placeholderAlloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("[SkyPass] Failed to create placeholder cubemap");

    // Cube image view.
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = m_placeholderCube;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format                          = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = kFaces;

    if (vkCreateImageView(device, &viewInfo, nullptr, &m_placeholderCubeView) != VK_SUCCESS)
        throw std::runtime_error("[SkyPass] Failed to create placeholder cubemap view");

    // Sampler: linear, clamp to edge.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter    = VK_FILTER_LINEAR;
    samplerInfo.minFilter    = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod       = VK_LOD_CLAMP_NONE;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &m_cubeSampler) != VK_SUCCESS)
        throw std::runtime_error("[SkyPass] Failed to create cubemap sampler");

    // Upload 6 black pixels via staging buffer.  Use a temporary command pool
    // so we can submit without depending on the caller's command buffer state.
    VkBuffer      stagingBuf  = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size  = kTotalBytes;
        bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo bufAllocCI{};
        bufAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
        bufAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                         | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo bufInfo2{};
        if (vmaCreateBuffer(allocator, &bufInfo, &bufAllocCI,
                            &stagingBuf, &stagingAlloc, &bufInfo2) != VK_SUCCESS)
            throw std::runtime_error("[SkyPass] Failed to create staging buffer for cubemap");

        // Black pixels: RGBA = 0,0,0,255 for each face.
        uint8_t pixels[kTotalBytes];
        for (uint32_t i = 0; i < kFaces; ++i) {
            pixels[i * 4 + 0] = 0;
            pixels[i * 4 + 1] = 0;
            pixels[i * 4 + 2] = 0;
            pixels[i * 4 + 3] = 255;
        }
        std::memcpy(bufInfo2.pMappedData, pixels, kTotalBytes);
        vmaFlushAllocation(allocator, stagingAlloc, 0, kTotalBytes);
    }

    // Issue upload via a temporary command buffer on the graphics queue.
    VkCommandPool   tmpPool = VK_NULL_HANDLE;
    VkCommandBuffer tmpCmd  = VK_NULL_HANDLE;
    VkQueue         queue   = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &queue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamily;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &tmpPool) != VK_SUCCESS)
        throw std::runtime_error("[SkyPass] Failed to create temporary command pool");

    VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool        = tmpPool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &cbInfo, &tmpCmd) != VK_SUCCESS) {
        vkDestroyCommandPool(device, tmpPool, nullptr);
        throw std::runtime_error("[SkyPass] Failed to allocate temporary command buffer");
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(tmpCmd, &beginInfo);

    // Transition all 6 faces to TRANSFER_DST.
    transitionRaw(tmpCmd, m_placeholderCube,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kFaces);

    // Copy one pixel per face from the staging buffer.
    VkBufferImageCopy regions[kFaces]{};
    for (uint32_t i = 0; i < kFaces; ++i) {
        regions[i].bufferOffset                = i * kBytesPerFace;
        regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[i].imageSubresource.mipLevel   = 0;
        regions[i].imageSubresource.baseArrayLayer = i;
        regions[i].imageSubresource.layerCount     = 1;
        regions[i].imageExtent = {1, 1, 1};
    }
    vkCmdCopyBufferToImage(tmpCmd, stagingBuf, m_placeholderCube,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kFaces, regions);

    // Transition to SHADER_READ_ONLY.
    transitionRaw(tmpCmd, m_placeholderCube,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kFaces);

    vkEndCommandBuffer(tmpCmd);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &tmpCmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkDestroyCommandPool(device, tmpPool, nullptr);
    vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);

    // Write sky descriptor set (set=1, binding=0 = placeholder cube sampler).
    m_skySet = descriptorPool != VK_NULL_HANDLE
               ? [&]() -> VkDescriptorSet {
                   VkDescriptorSetAllocateInfo dsInfo{};
                   dsInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                   dsInfo.descriptorPool     = descriptorPool;
                   dsInfo.descriptorSetCount = 1;
                   dsInfo.pSetLayouts        = &m_skyLayout;
                   VkDescriptorSet ds = VK_NULL_HANDLE;
                   if (vkAllocateDescriptorSets(device, &dsInfo, &ds) != VK_SUCCESS)
                       throw std::runtime_error("[SkyPass] Failed to allocate sky descriptor set");
                   return ds;
               }()
               : VK_NULL_HANDLE;

    if (m_skySet != VK_NULL_HANDLE) {
        VkDescriptorImageInfo imgDescInfo{};
        imgDescInfo.sampler     = m_cubeSampler;
        imgDescInfo.imageView   = m_placeholderCubeView;
        imgDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = m_skySet;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo      = &imgDescInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void SkyPass::createPipeline(VkDevice device, VkRenderPass renderPass,
                              VkSampleCountFlagBits samples, VkPipelineCache cache,
                              VkDescriptorSetLayout globalLayout,
                              const std::string& vertSpv, const std::string& fragSpv) {
    // Pipeline layout: set=0 (globalLayout) + set=1 (m_skyLayout) + push constants.
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(SkyPushConstants);

    VkDescriptorSetLayout setLayouts[2] = {globalLayout, m_skyLayout};

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 2;
    layoutInfo.pSetLayouts            = setLayouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;

    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("[SkyPass] Failed to create pipeline layout");

    Shader vert, frag;
    vert.load(device, vertSpv);
    frag.load(device, fragSpv);

    // Sky pipeline: no vertex buffers, no depth write, depth test LESS_OR_EQUAL,
    // no back-face culling (cube viewed from inside).
    m_pipeline = PipelineBuilder::forFullscreenQuad()
        .addShaderStage(vert.stageInfo(VK_SHADER_STAGE_VERTEX_BIT))
        .addShaderStage(frag.stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT))
        .setDepthTest(true, false, VK_COMPARE_OP_LESS_OR_EQUAL)
        .setMultisampling(samples)
        .build(device, m_pipelineLayout, renderPass, 0, cache);

    if (m_pipeline == VK_NULL_HANDLE)
        throw std::runtime_error("[SkyPass] Failed to create sky pipeline");
}

void SkyPass::draw(VkCommandBuffer cmd, VkDescriptorSet globalSet) {
    if (m_cfg.mode == SkyMode::None || m_pipeline == VK_NULL_HANDLE) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    // Bind set=0 (global UBO with view+proj) and set=1 (sky cubemap).
    VkDescriptorSet sets[2] = {globalSet, m_skySet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipelineLayout, 0, 2, sets, 0, nullptr);

    SkyPushConstants pc{};
    pc.skyColor = glm::vec4(m_cfg.skyColor, 1.0f);
    pc.skyMode  = (m_cfg.mode == SkyMode::Skybox) ? 1
                 : (m_cfg.mode == SkyMode::SkySphere ? 2 : 0);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    // Draw 36 vertices (6 faces x 2 triangles x 3 verts), no vertex buffers needed.
    vkCmdDraw(cmd, 36, 1, 0, 0);
}

void SkyPass::setCubemap(VkImageView cubeView, VkSampler sampler) {
    if (m_skySet == VK_NULL_HANDLE || m_device == VK_NULL_HANDLE) return;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler     = sampler;
    imgInfo.imageView   = cubeView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = m_skySet;
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo      = &imgInfo;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    m_cfg.mode = SkyMode::Skybox;
}

void SkyPass::destroy() {
    if (m_device == VK_NULL_HANDLE) return;

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_placeholderCubeView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_placeholderCubeView, nullptr);
        m_placeholderCubeView = VK_NULL_HANDLE;
    }
    if (m_placeholderCube != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_placeholderCube, m_placeholderAlloc);
        m_placeholderCube  = VK_NULL_HANDLE;
        m_placeholderAlloc = VK_NULL_HANDLE;
    }
    if (m_cubeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_cubeSampler, nullptr);
        m_cubeSampler = VK_NULL_HANDLE;
    }
    if (m_skyLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_skyLayout, nullptr);
        m_skyLayout = VK_NULL_HANDLE;
    }
    // m_skySet is owned by the caller's descriptor pool (reset-all strategy).
    m_skySet = VK_NULL_HANDLE;

    m_device    = VK_NULL_HANDLE;
    m_allocator = VK_NULL_HANDLE;
}

} // namespace vkt
