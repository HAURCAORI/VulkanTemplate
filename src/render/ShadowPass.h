#pragma once

#include "resources/Image.h"
#include "scene/RenderWorld.h"
#include "config/RenderLimits.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <array>
#include <string>
#include <vector>

namespace vkt {

// Renders a single directional shadow map (D32_SFLOAT depth image).
// Owned by Application; init() before the main pipeline, destroy() before the allocator.
//
// Each frame:
//   1. Call render() -- records shadow depth pass + updates lightMatrix()
//   2. lightMatrix() is uploaded to GlobalUBO.lightSpaceMatrix
//   3. Main pass samples the shadow map via the global descriptor set (set=0, binding=2)
class ShadowPass {
public:
    struct Config {
        uint32_t cascadeCount = 4;       // active cascades [1..kMaxShadowCascades]
        uint32_t resolution   = 2048;    // per-cascade shadow map resolution
        float    splitLambda  = 0.65f;   // 0=uniform, 1=logarithmic cascade splits
        float    biasConstant = 0.4f;    // constant depth bias (vkCmdSetDepthBias)
        float    biasSlope    = 1.2f;    // slope-scaled depth bias
        float    depthPadding    = 10.0f;  // light-space z padding to avoid clipping
        float    eyeDistBaseline = 200.0f; // minimum eye distance from cascade center; prevents
                                           // coordinate-system collapse for close cameras
        bool     stabilize    = true;    // snap projection to texel grid to reduce shimmer
    };

    ShadowPass() = default;
    ~ShadowPass() { destroy(); }
    ShadowPass(const ShadowPass&)            = delete;
    ShadowPass& operator=(const ShadowPass&) = delete;

    // globalLayout: the global descriptor set layout (set=0; shadow shader uses binding=1 SSBO).
    // Call after createDescriptors() so the globalLayout is ready.
    // shadowVertSpv: absolute path to shadow.vert.spv (use Application::resolveShaderPath).
    void init(VkDevice              device,
              VmaAllocator          allocator,
              VkPhysicalDevice      physDevice,
              VkDescriptorSetLayout globalLayout,
              VkPipelineCache       pipelineCache,
              const std::string&    shadowVertSpv,
              const Config&         cfg = {});

    void destroy();

    // Update cascade matrices/splits from the current camera view.
    // Call before writing GlobalUBO for this frame.
    void updateCascades(const glm::vec3& lightDir,
                        const glm::mat4& cameraView,
                        const glm::mat4& cameraProj,
                        float cameraNear,
                        float cameraFar);

    // Record all cascade depth passes into cmd (call BEFORE the main render pass).
    // globalSet: set=0 descriptor (must contain the instance SSBO at binding=1).
    void render(VkCommandBuffer               cmd,
                VkDescriptorSet               globalSet,
                const std::vector<DrawBatch>& batches);

    uint32_t cascadeCount() const { return m_cascadeCount; }
    const std::array<glm::mat4, RenderLimits::kMaxShadowCascades>& cascadeMatrices() const {
        return m_lightSpaceMatrices;
    }
    const std::array<float, RenderLimits::kMaxShadowCascades>& cascadeSplits() const {
        return m_cascadeSplits;
    }

    // Shadow depth image views and shared sampler -- write to global binding=2 array.
    std::vector<VkImageView> depthViews() const;
    VkSampler   sampler() const { return m_shadowSampler; }

    bool enabled()        const { return m_enabled; }
    void setEnabled(bool e)     { m_enabled = e; }

private:
    void createRenderPass();
    void createFramebuffers();
    void createShadowSampler();
    void createPipeline(VkDescriptorSetLayout globalLayout,
                        VkPipelineCache       pipelineCache,
                        const std::string&    shadowVertSpv);

    struct CascadeResources {
        Image         image;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
    };
    std::vector<CascadeResources> m_cascades;
    VkSampler        m_shadowSampler = VK_NULL_HANDLE;
    VkRenderPass     m_renderPass    = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeLayout    = VK_NULL_HANDLE;
    VkPipeline       m_pipeline      = VK_NULL_HANDLE;
    VkDevice         m_device        = VK_NULL_HANDLE;
    uint32_t         m_cascadeCount = 1;
    std::array<glm::mat4, RenderLimits::kMaxShadowCascades> m_lightSpaceMatrices{};
    std::array<float, RenderLimits::kMaxShadowCascades>     m_cascadeSplits{};
    Config           m_cfg;
    bool             m_enabled = true;
};

} // namespace vkt
