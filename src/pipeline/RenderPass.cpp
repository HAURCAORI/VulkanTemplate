#include "RenderPass.h"
#include "core/Context.h"
#include "core/SwapChain.h"

#include <stdexcept>
#include <array>

namespace vkt {

void RenderPass::init(Context& ctx, SwapChain& swapchain) {
    m_ctx       = &ctx;
    m_swapchain = &swapchain;

    const bool     msaa    = swapchain.isMsaaEnabled();
    const VkSampleCountFlagBits samples = swapchain.msaaSamples();

    // -- Common subpass dependency ----------------------------------------------
    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                      | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                      | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.subpassCount    = 1;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies   = &dep;

    if (msaa) {
        // -- MSAA path: color(MSAA) + depth(MSAA) + resolve(1x) ---------------
        // Attachment 0: multisampled color ??rendered into, not stored
        VkAttachmentDescription msaaColor{};
        msaaColor.format         = swapchain.imageFormat();
        msaaColor.samples        = samples;
        msaaColor.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        msaaColor.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE; // resolved, not stored
        msaaColor.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        msaaColor.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        msaaColor.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        msaaColor.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // Attachment 1: multisampled depth
        VkAttachmentDescription depth{};
        depth.format         = swapchain.depthFormat();
        depth.samples        = samples;
        depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // Attachment 2: resolve target (single-sample swapchain image for present)
        VkAttachmentDescription resolve{};
        resolve.format         = swapchain.imageFormat();
        resolve.samples        = VK_SAMPLE_COUNT_1_BIT;
        resolve.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolve.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        resolve.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        resolve.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        resolve.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef  {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef  {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkAttachmentReference resolveRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;
        subpass.pResolveAttachments     = &resolveRef;

        std::array<VkAttachmentDescription, 3> attachments{msaaColor, depth, resolve};
        rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpInfo.pAttachments    = attachments.data();
        rpInfo.pSubpasses      = &subpass;

        if (vkCreateRenderPass(ctx.device(), &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS)
            throw std::runtime_error("[RenderPass] Failed to create MSAA render pass");

    } else {
        // -- No-MSAA path: color(1x) + depth(1x) ------------------------------
        VkAttachmentDescription color{};
        color.format         = swapchain.imageFormat();
        color.samples        = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depth{};
        depth.format         = swapchain.depthFormat();
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

        std::array<VkAttachmentDescription, 2> attachments{color, depth};
        rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpInfo.pAttachments    = attachments.data();
        rpInfo.pSubpasses      = &subpass;

        if (vkCreateRenderPass(ctx.device(), &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS)
            throw std::runtime_error("[RenderPass] Failed to create render pass");
    }

    createFramebuffers();

    // Register rebuild callback so SwapChain::rebuild() recreates our framebuffers
    swapchain.setRebuildCallback([this]() { rebuild(); });
}

void RenderPass::rebuild() {
    destroyFramebuffers();
    createFramebuffers();
}

void RenderPass::createFramebuffers() {
    const auto& swapViews = m_swapchain->imageViews();
    VkExtent2D  extent    = m_swapchain->extent();
    m_framebuffers.resize(swapViews.size());

    for (size_t i = 0; i < swapViews.size(); ++i) {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType     = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_renderPass;
        fbInfo.width      = extent.width;
        fbInfo.height     = extent.height;
        fbInfo.layers     = 1;

        if (m_swapchain->isMsaaEnabled()) {
            // MSAA: [msaaColor, depth, resolve(swapchainImage[i])]
            std::array<VkImageView, 3> attachments{
                m_swapchain->msaaImageView(),
                m_swapchain->depthImageView(),
                swapViews[i]
            };
            fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            fbInfo.pAttachments    = attachments.data();
        } else {
            // No MSAA: [swapchainImage[i], depth]
            std::array<VkImageView, 2> attachments{
                swapViews[i],
                m_swapchain->depthImageView()
            };
            fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            fbInfo.pAttachments    = attachments.data();
        }

        if (vkCreateFramebuffer(m_ctx->device(), &fbInfo, nullptr, &m_framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("[RenderPass] Failed to create framebuffer");
    }
}

void RenderPass::destroyFramebuffers() {
    for (auto fb : m_framebuffers)
        vkDestroyFramebuffer(m_ctx->device(), fb, nullptr);
    m_framebuffers.clear();
}

void RenderPass::begin(VkCommandBuffer cmd, uint32_t imageIndex,
                       VkClearColorValue clearColor) const {
    // Indices 0 (color) and 1 (depth) both use CLEAR loadOp in both MSAA and
    // non-MSAA paths. The resolve attachment at index 2 (MSAA only) uses
    // DONT_CARE so it never needs a clear value  --  count=2 covers both paths.
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = clearColor;
    clearValues[1].depthStencil = {1.0f, 0};

    VkExtent2D extent = m_swapchain->extent();

    VkRenderPassBeginInfo info{};
    info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass        = m_renderPass;
    info.framebuffer       = m_framebuffers[imageIndex];
    info.renderArea.offset = {0, 0};
    info.renderArea.extent = extent;
    // clearValueCount must cover every attachment up to and including the
    // highest-indexed one with VK_ATTACHMENT_LOAD_OP_CLEAR.  In both MSAA and
    // non-MSAA paths, index 1 (depth) is the last CLEAR attachment.
    // The resolve attachment at index 2 uses DONT_CARE, so count=2 is correct.
    info.clearValueCount   = 2u;
    info.pClearValues      = clearValues.data();

    vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);

    // Dynamic viewport and scissor (set in CMakeLists via dynamic state below)
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void RenderPass::end(VkCommandBuffer cmd) const {
    vkCmdEndRenderPass(cmd);
}

void RenderPass::destroy() {
    if (!m_ctx) return;
    destroyFramebuffers();
    if (m_renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(m_ctx->device(), m_renderPass, nullptr);
        m_renderPass = VK_NULL_HANDLE;
    }
    m_ctx = nullptr;
}

RenderPass::~RenderPass() { destroy(); }

} // namespace vkt
