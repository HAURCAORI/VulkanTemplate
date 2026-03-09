#include "ShadowPass.h"

#include "pipeline/PipelineBuilder.h"
#include "pipeline/Shader.h"
#include "scene/Mesh.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vkt {

namespace {

struct ShadowPushConstants {
    glm::mat4 lightSpaceMatrix;
};

std::array<glm::vec3, 8> buildFrustumCornersWS(const glm::mat4& invViewProj) {
    std::array<glm::vec3, 8> corners{};
    uint32_t idx = 0;
    for (int z = 0; z <= 1; ++z) {
        for (int y = -1; y <= 1; y += 2) {
            for (int x = -1; x <= 1; x += 2) {
                glm::vec4 p = invViewProj * glm::vec4(static_cast<float>(x), static_cast<float>(y),
                                                      static_cast<float>(z), 1.0f);
                corners[idx++] = glm::vec3(p) / p.w;
            }
        }
    }
    return corners;
}

} // namespace

void ShadowPass::init(VkDevice device, VmaAllocator allocator,
                      VkPhysicalDevice /*physDevice*/,
                      VkDescriptorSetLayout globalLayout,
                      VkPipelineCache pipelineCache,
                      const std::string& shadowVertSpv,
                      const Config& cfg) {
    if (m_device != VK_NULL_HANDLE) return; // re-init guard: already initialized

    m_device = device;
    m_cfg    = cfg;
    m_cascadeCount = std::clamp(cfg.cascadeCount, 1u, RenderLimits::kMaxShadowCascades);
    m_lightSpaceMatrices.fill(glm::mat4{1.0f});
    m_cascadeSplits.fill(std::numeric_limits<float>::max());

    try {
        m_cascades.resize(m_cascadeCount);
        for (auto& c : m_cascades) {
            c.image.create(allocator, device, {
                cfg.resolution, cfg.resolution,
                VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                1
            });
        }

        createRenderPass();
        createFramebuffers();
        createShadowSampler();
        createPipeline(globalLayout, pipelineCache, shadowVertSpv);
    } catch (...) {
        destroy(); // release any partially-created Vulkan objects
        throw;
    }
}

void ShadowPass::createRenderPass() {
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format         = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 0;
    depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 0;
    subpass.pDepthStencilAttachment = &depthRef;

    std::array<VkSubpassDependency, 2> deps{};
    deps[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass      = 0;
    deps[0].srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask    = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[0].dependencyFlags = 0;

    deps[1].srcSubpass      = 0;
    deps[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask    = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[1].dependencyFlags = 0;

    VkRenderPassCreateInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments    = &depthAttachment;
    rpInfo.subpassCount    = 1;
    rpInfo.pSubpasses      = &subpass;
    rpInfo.dependencyCount = static_cast<uint32_t>(deps.size());
    rpInfo.pDependencies   = deps.data();

    if (vkCreateRenderPass(m_device, &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create render pass");
}

void ShadowPass::createFramebuffers() {
    for (auto& c : m_cascades) {
        VkImageView view = c.image.view();
        VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbInfo.renderPass      = m_renderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments    = &view;
        fbInfo.width           = m_cfg.resolution;
        fbInfo.height          = m_cfg.resolution;
        fbInfo.layers          = 1;

        if (vkCreateFramebuffer(m_device, &fbInfo, nullptr, &c.framebuffer) != VK_SUCCESS)
            throw std::runtime_error("[ShadowPass] Failed to create framebuffer");
    }
}

void ShadowPass::createShadowSampler() {
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter        = VK_FILTER_LINEAR;
    samplerInfo.minFilter        = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.compareEnable    = VK_FALSE;
    samplerInfo.compareOp        = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod           = 0.0f;
    samplerInfo.maxLod           = 0.0f;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy    = 1.0f;

    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_shadowSampler) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create shadow sampler");
}

void ShadowPass::createPipeline(VkDescriptorSetLayout globalLayout,
                                VkPipelineCache pipelineCache,
                                const std::string& shadowVertSpv) {
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(ShadowPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &globalLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipeLayout) != VK_SUCCESS)
        throw std::runtime_error("[ShadowPass] Failed to create pipeline layout");

    Shader vert;
    vert.load(m_device, shadowVertSpv);

    auto binding = Vertex::bindingDesc();
    VkVertexInputAttributeDescription posAttr{};
    posAttr.location = 0;
    posAttr.binding  = 0;
    posAttr.format   = VK_FORMAT_R32G32B32_SFLOAT;
    posAttr.offset   = offsetof(Vertex, position);
    std::vector<VkVertexInputBindingDescription>   bindings{binding};
    std::vector<VkVertexInputAttributeDescription> attrs{posAttr};

    m_pipeline = PipelineBuilder{}
        .addShaderStage(vert.stageInfo(VK_SHADER_STAGE_VERTEX_BIT))
        .setVertexInput(bindings, attrs)
        .setDepthTest(true, true)
        .setCullMode(VK_CULL_MODE_NONE)
        .setDepthBias(true)
        .setColorAttachments(0)
        .build(m_device, m_pipeLayout, m_renderPass, 0, pipelineCache);
}

void ShadowPass::updateCascades(const glm::vec3& lightDir,
                                const glm::mat4& cameraView,
                                const glm::mat4& cameraProj,
                                float cameraNear,
                                float cameraFar) {
    const float nearPlane = std::max(0.01f, cameraNear);
    const float farPlane  = std::max(nearPlane + 0.1f, cameraFar);
    const float clipRange = farPlane - nearPlane;
    const float lambda = std::clamp(m_cfg.splitLambda, 0.0f, 1.0f);

    std::array<float, RenderLimits::kMaxShadowCascades + 1> splits{};
    splits[0] = nearPlane;
    for (uint32_t i = 1; i <= m_cascadeCount; ++i) {
        const float p    = static_cast<float>(i) / static_cast<float>(m_cascadeCount);
        const float logD = nearPlane * std::pow(farPlane / nearPlane, p);
        const float uniD = nearPlane + clipRange * p;
        splits[i] = uniD + lambda * (logD - uniD);
    }
    for (uint32_t i = m_cascadeCount; i < RenderLimits::kMaxShadowCascades; ++i)
        m_cascadeSplits[i] = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < m_cascadeCount; ++i)
        m_cascadeSplits[i] = splits[i + 1];

    const glm::mat4 invViewProj = glm::inverse(cameraProj * cameraView);
    const auto frustumCorners = buildFrustumCornersWS(invViewProj);
    const glm::vec3 lightDirN = glm::normalize(lightDir);

    for (uint32_t c = 0; c < m_cascadeCount; ++c) {
        const float tNear = (splits[c]     - nearPlane) / clipRange;
        const float tFar  = (splits[c + 1] - nearPlane) / clipRange;

        std::array<glm::vec3, 8> cascadeCorners{};
        for (uint32_t i = 0; i < 4; ++i) {
            const glm::vec3 nearCorner = frustumCorners[i];
            const glm::vec3 farCorner  = frustumCorners[i + 4];
            const glm::vec3 ray = farCorner - nearCorner;
            cascadeCorners[i]     = nearCorner + ray * tNear;
            cascadeCorners[i + 4] = nearCorner + ray * tFar;
        }

        glm::vec3 center{0.0f};
        for (const auto& p : cascadeCorners) center += p;
        center /= 8.0f;

        glm::vec3 up = (std::abs(lightDirN.y) > 0.99f) ? glm::vec3{0.0f, 0.0f, 1.0f}
                                                       : glm::vec3{0.0f, 1.0f, 0.0f};

        // The light eye must be far enough upstream (against lightDirN) that all cascade
        // corners have negative z (in front) in the resulting view.
        // 200 is stable for typical close cameras; when corners extend farther upstream
        // (camera far away, frustum fans wide) the eye must be pushed back accordingly.
        // Measure actual upstream extent via a probe view, then take the larger of 200
        // and (upstreamExtent + padding) — close cameras keep stable behaviour, far
        // cameras get the extra room without flipping corners behind the eye.
        const glm::mat4 lightProbe = glm::lookAt(center, center + lightDirN, up);
        float maxUpstreamZ = 0.0f;
        for (const auto& p : cascadeCorners) {
            // Positive probe z = corner is upstream (against light direction) of center.
            const float z = (lightProbe * glm::vec4(p, 1.0f)).z;
            maxUpstreamZ = std::max(maxUpstreamZ, z);
        }
        const float eyeDist = std::max(m_cfg.eyeDistBaseline, maxUpstreamZ + m_cfg.depthPadding);
        const glm::mat4 lightView = glm::lookAt(center - lightDirN * eyeDist, center, up);

        glm::vec3 mins{ std::numeric_limits<float>::max() };
        glm::vec3 maxs{ std::numeric_limits<float>::lowest() };
        for (const auto& p : cascadeCorners) {
            const glm::vec3 ls = glm::vec3(lightView * glm::vec4(p, 1.0f));
            mins = glm::min(mins, ls);
            maxs = glm::max(maxs, ls);
        }
        mins.z -= m_cfg.depthPadding;
        maxs.z += m_cfg.depthPadding;

        if (m_cfg.stabilize) {
            const float extentX = maxs.x - mins.x;
            const float extentY = maxs.y - mins.y;
            const float texelX = extentX / static_cast<float>(m_cfg.resolution);
            const float texelY = extentY / static_cast<float>(m_cfg.resolution);
            glm::vec3 centerLS = 0.5f * (mins + maxs);
            centerLS.x = std::floor(centerLS.x / texelX) * texelX;
            centerLS.y = std::floor(centerLS.y / texelY) * texelY;
            mins.x = centerLS.x - 0.5f * extentX;
            maxs.x = centerLS.x + 0.5f * extentX;
            mins.y = centerLS.y - 0.5f * extentY;
            maxs.y = centerLS.y + 0.5f * extentY;
        }

        // glm::lookAt() produces RH view space where in-front points have
        // negative z. glm::ortho() expects positive near/far distances, so
        // convert from view-space z extents: near=-maxZ, far=-minZ.
        float nearDist = std::max(0.01f, -maxs.z);
        float farDist  = std::max(nearDist + 0.1f, -mins.z);
        glm::mat4 lightProj = glm::ortho(mins.x, maxs.x, mins.y, maxs.y, nearDist, farDist);
        lightProj[1][1] *= -1.0f;
        m_lightSpaceMatrices[c] = lightProj * lightView;
    }
}

void ShadowPass::render(VkCommandBuffer cmd,
                        VkDescriptorSet globalSet,
                        const std::vector<DrawBatch>& batches) {
    if (!m_enabled) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeLayout, 0, 1, &globalSet, 0, nullptr);

    VkViewport vp{};
    vp.width    = static_cast<float>(m_cfg.resolution);
    vp.height   = static_cast<float>(m_cfg.resolution);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    VkRect2D sc{{0, 0}, {m_cfg.resolution, m_cfg.resolution}};

    for (uint32_t c = 0; c < m_cascadeCount; ++c) {
        VkClearValue clearValue{};
        clearValue.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rpBegin.renderPass        = m_renderPass;
        rpBegin.framebuffer       = m_cascades[c].framebuffer;
        rpBegin.renderArea.offset = {0, 0};
        rpBegin.renderArea.extent = {m_cfg.resolution, m_cfg.resolution};
        rpBegin.clearValueCount   = 1;
        rpBegin.pClearValues      = &clearValue;

        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdSetDepthBias(cmd, m_cfg.biasConstant, 0.0f, m_cfg.biasSlope);

        ShadowPushConstants push{m_lightSpaceMatrices[c]};
        vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);

        for (const auto& b : batches)
            b.mesh->drawInstanced(cmd, b.count, b.firstInstance);

        vkCmdEndRenderPass(cmd);
    }
}

std::vector<VkImageView> ShadowPass::depthViews() const {
    std::vector<VkImageView> views;
    views.reserve(m_cascadeCount);
    for (const auto& c : m_cascades) views.push_back(c.image.view());
    return views;
}

void ShadowPass::destroy() {
    if (m_device == VK_NULL_HANDLE) return;

    for (auto& c : m_cascades) {
        if (c.framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(m_device, c.framebuffer, nullptr);
            c.framebuffer = VK_NULL_HANDLE;
        }
        c.image.destroy();
    }
    m_cascades.clear();

    if (m_pipeline      != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_pipeline, nullptr);
    if (m_pipeLayout    != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_pipeLayout, nullptr);
    if (m_renderPass    != VK_NULL_HANDLE) vkDestroyRenderPass(m_device, m_renderPass, nullptr);
    if (m_shadowSampler != VK_NULL_HANDLE) vkDestroySampler(m_device, m_shadowSampler, nullptr);

    m_pipeline      = VK_NULL_HANDLE;
    m_pipeLayout    = VK_NULL_HANDLE;
    m_renderPass    = VK_NULL_HANDLE;
    m_shadowSampler = VK_NULL_HANDLE;
    m_device        = VK_NULL_HANDLE;
}

} // namespace vkt
