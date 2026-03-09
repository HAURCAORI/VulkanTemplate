#include "PostProcessPass.h"

#include "pipeline/PipelineBuilder.h"
#include "pipeline/Shader.h"

#include <array>
#include <cassert>
#include <stdexcept>

namespace vkt {

// -- Init / destroy -------------------------------------------------------------

void PostProcessPass::init(VkDevice device, VmaAllocator allocator,
                           VkExtent2D extent,
                           VkRenderPass swapchainRenderPass,
                           VkSampleCountFlagBits swapchainSamples,
                           VkPipelineCache pipelineCache,
                           uint32_t frameCount,
                           const std::string& vertSpv,
                           const std::string& fragSpv,
                           const Config& cfg) {
    if (m_device != VK_NULL_HANDLE) return; // re-init guard: already initialized

    m_device     = device;
    m_extent     = extent;
    m_frameCount = frameCount;
    m_cfg        = cfg;

    try {
        createHdrRenderPass();
        createImages(allocator, extent);
        createFramebuffers();
        createSampler();
        createDescriptors();
        createPipeline(swapchainRenderPass, swapchainSamples, pipelineCache, vertSpv, fragSpv);
    } catch (...) {
        destroy(); // release any partially-created Vulkan objects
        throw;
    }
}

void PostProcessPass::destroy() {
    if (m_device == VK_NULL_HANDLE) return;

    if (m_pipeline   != VK_NULL_HANDLE) vkDestroyPipeline      (m_device, m_pipeline,   nullptr);
    if (m_pipeLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_pipeLayout, nullptr);

    m_postPool.destroy();
    if (m_postLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_postLayout, nullptr);
        m_postLayout = VK_NULL_HANDLE;
    }

    if (m_sampler != VK_NULL_HANDLE) vkDestroySampler(m_device, m_sampler, nullptr);
    destroyFramebuffers();
    destroyImages();

    if (m_hdrRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(m_device, m_hdrRenderPass, nullptr);
        m_hdrRenderPass = VK_NULL_HANDLE;
    }

    m_pipeline   = VK_NULL_HANDLE;
    m_pipeLayout = VK_NULL_HANDLE;
    m_sampler    = VK_NULL_HANDLE;
    m_device     = VK_NULL_HANDLE;
}

void PostProcessPass::rebuild(VmaAllocator allocator, VkExtent2D newExtent) {
    assert(m_device != VK_NULL_HANDLE && "PostProcessPass::rebuild() called before init()");
    m_extent = newExtent;
    destroyFramebuffers();
    destroyImages();
    createImages(allocator, newExtent);
    createFramebuffers();
    updateDescriptors();   // point descriptor sets at the new image views
}

// -- Private helpers -----------------------------------------------------------

void PostProcessPass::createHdrRenderPass() {
    // Color attachment: R16G16B16A16_SFLOAT, cleared each frame.
    // Final layout SHADER_READ_ONLY_OPTIMAL so the tonemap pass can sample it.
    VkAttachmentDescription color{};
    color.format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;            // discard previous
    color.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Depth attachment: D32_SFLOAT, cleared each frame, not needed after the pass.
    VkAttachmentDescription depth{};
    depth.format         = VK_FORMAT_D32_SFLOAT;
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    // Subpass dependencies ensure color image transitions before/after rendering.
    std::array<VkSubpassDependency, 2> deps{};
    // External -> 0: wait for the previous frame's tonemap read before writing color.
    deps[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass      = 0;
    deps[0].srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[0].dependencyFlags = 0; // No BY_REGION: tonemap reads the full image (non-local)
    // 0 -> External: wait for color writes before the tonemap pass reads.
    deps[1].srcSubpass      = 0;
    deps[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[1].dependencyFlags = 0; // No BY_REGION: tonemap reads the full image (non-local)

    std::array<VkAttachmentDescription, 2> attachments{color, depth};
    VkRenderPassCreateInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments    = attachments.data();
    rpInfo.subpassCount    = 1;
    rpInfo.pSubpasses      = &subpass;
    rpInfo.dependencyCount = static_cast<uint32_t>(deps.size());
    rpInfo.pDependencies   = deps.data();

    if (vkCreateRenderPass(m_device, &rpInfo, nullptr, &m_hdrRenderPass) != VK_SUCCESS)
        throw std::runtime_error("[PostProcessPass] Failed to create HDR render pass");
}

void PostProcessPass::createImages(VmaAllocator allocator, VkExtent2D extent) {
    m_hdrImages.resize(m_frameCount);
    for (auto& img : m_hdrImages) {
        img.create(allocator, m_device, {
            extent.width, extent.height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1  // mip levels
        });
    }
    // One shared depth image -- write + sample are sequential within one command buffer.
    m_depthImage.create(allocator, m_device, {
        extent.width, extent.height,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        1
    });
}

void PostProcessPass::createFramebuffers() {
    m_hdrFramebuffers.resize(m_frameCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < m_frameCount; ++i) {
        std::array<VkImageView, 2> views{m_hdrImages[i].view(), m_depthImage.view()};
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass      = m_hdrRenderPass;
        fb.attachmentCount = static_cast<uint32_t>(views.size());
        fb.pAttachments    = views.data();
        fb.width           = m_extent.width;
        fb.height          = m_extent.height;
        fb.layers          = 1;
        if (vkCreateFramebuffer(m_device, &fb, nullptr, &m_hdrFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("[PostProcessPass] Failed to create HDR framebuffer");
    }
}

void PostProcessPass::createSampler() {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter        = VK_FILTER_LINEAR;
    info.minFilter        = VK_FILTER_LINEAR;
    info.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.anisotropyEnable = VK_FALSE;
    info.maxAnisotropy    = 1.0f;
    info.minLod           = 0.0f;
    info.maxLod           = 0.0f;
    if (vkCreateSampler(m_device, &info, nullptr, &m_sampler) != VK_SUCCESS)
        throw std::runtime_error("[PostProcessPass] Failed to create HDR sampler");
}

void PostProcessPass::createDescriptors() {
    m_postLayout = DescriptorLayoutBuilder{}
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
        .build(m_device);

    // One set per frame-in-flight; each set points to its own HDR image.
    m_postPool.init(m_device, m_frameCount,
                    {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1.0f}});

    m_postSets.resize(m_frameCount);
    for (uint32_t i = 0; i < m_frameCount; ++i)
        m_postSets[i] = m_postPool.allocate(m_postLayout);

    updateDescriptors();
}

void PostProcessPass::updateDescriptors() {
    for (uint32_t i = 0; i < m_frameCount; ++i) {
        DescriptorWriter{}
            .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        m_hdrImages[i].view(), m_sampler,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .update(m_device, m_postSets[i]);
    }
}

void PostProcessPass::createPipeline(VkRenderPass swapchainRenderPass,
                                     VkSampleCountFlagBits swapchainSamples,
                                     VkPipelineCache cache,
                                     const std::string& vertSpv,
                                     const std::string& fragSpv) {
    // Push constant: exposure + gamma + tonemap mode (fragment stage only).
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(PostPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &m_postLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;
    if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipeLayout) != VK_SUCCESS)
        throw std::runtime_error("[PostProcessPass] Failed to create pipeline layout");

    Shader vert, frag;
    vert.load(m_device, vertSpv);
    frag.load(m_device, fragSpv);

    m_pipeline = PipelineBuilder::forFullscreenQuad()
        .addShaderStage(vert.stageInfo(VK_SHADER_STAGE_VERTEX_BIT))
        .addShaderStage(frag.stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT))
        .setMultisampling(swapchainSamples)
        .build(m_device, m_pipeLayout, swapchainRenderPass, 0, cache);
}

void PostProcessPass::destroyImages() {
    for (auto& img : m_hdrImages) img.destroy();
    m_hdrImages.clear();
    m_depthImage.destroy();
}

void PostProcessPass::destroyFramebuffers() {
    for (auto fb : m_hdrFramebuffers)
        if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(m_device, fb, nullptr);
    m_hdrFramebuffers.clear();
}

// -- Per-frame -----------------------------------------------------------------

void PostProcessPass::beginHdr(VkCommandBuffer cmd, uint32_t frameIndex) const {
    std::array<VkClearValue, 2> clears{};
    clears[0].color        = {0.0f, 0.0f, 0.0f, 1.0f};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    info.renderPass        = m_hdrRenderPass;
    info.framebuffer       = m_hdrFramebuffers[frameIndex];
    info.renderArea        = {{0, 0}, m_extent};
    info.clearValueCount   = 2;
    info.pClearValues      = clears.data();
    vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);

    // Set dynamic viewport + scissor to match the HDR target resolution.
    VkViewport vp{0.0f, 0.0f,
                  static_cast<float>(m_extent.width), static_cast<float>(m_extent.height),
                  0.0f, 1.0f};
    VkRect2D sc{{0, 0}, m_extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor (cmd, 0, 1, &sc);
}

void PostProcessPass::endHdr(VkCommandBuffer cmd) const {
    vkCmdEndRenderPass(cmd);
}

void PostProcessPass::blit(VkCommandBuffer cmd, uint32_t frameIndex) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeLayout, 0, 1, &m_postSets[frameIndex], 0, nullptr);

    PostPushConstants push;
    push.exposure    = m_cfg.exposure;
    push.gamma       = m_cfg.gamma;
    push.tonemapMode = m_cfg.tonemapMode;
    push._pad        = 0.0f;
    vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(push), &push);

    // Three vertices -- no vertex buffer required; fullscreen.vert uses gl_VertexIndex.
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace vkt
