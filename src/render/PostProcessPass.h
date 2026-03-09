#pragma once

#include "pipeline/Descriptors.h"
#include "resources/Image.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>
#include <vector>

namespace vkt {

// Push constant block for the fullscreen tonemap shader.
// Sent each blit() call so exposure/gamma/mode can change per-frame.
struct PostPushConstants {
    float exposure    = 1.0f;  // linear exposure multiplier (applied before tonemap)
    float gamma       = 2.2f;  // output gamma (applied after tonemap)
    int   tonemapMode = 0;     // 0=Reinhard, 1=ACES approx, 2=clamp only
    float _pad        = 0.0f;
};
static_assert(sizeof(PostPushConstants) == 16, "PostPushConstants size mismatch");

// HDR offscreen color pass + fullscreen tonemap blit.
//
// The geometry pass renders into an R16G16B16A16_SFLOAT image (no MSAA) owned
// by this class. After the geometry pass finishes, blit() draws a fullscreen
// triangle inside the swapchain render pass that samples the HDR image and
// outputs a tonemapped + gamma-corrected LDR color.
//
// Usage pattern (per frame):
//   m_postProcess.beginHdr(cmd, currentFrame);
//     // ... geometry pass (vkCmdBindPipeline / vkCmdDraw / ...) ...
//   m_postProcess.endHdr(cmd);
//   m_renderPass.begin(cmd, imageIndex);          // swapchain pass
//     m_postProcess.blit(cmd, currentFrame);      // tonemap
//     // ... ImGui ...
//   m_renderPass.end(cmd);
//
// Resize: call rebuild() after the swapchain is rebuilt.
// Pipeline and render passes are stable; only images/framebuffers are recreated.
class PostProcessPass {
public:
    struct Config {
        float exposure    = 1.0f;
        float gamma       = 2.2f;
        int   tonemapMode = 0;   // 0=Reinhard, 1=ACES, 2=Clamp
    };

    PostProcessPass() = default;
    ~PostProcessPass() { destroy(); }
    PostProcessPass(const PostProcessPass&)            = delete;
    PostProcessPass& operator=(const PostProcessPass&) = delete;

    // swapchainRenderPass: render pass the blit pipeline will target (compatibility match).
    // swapchainSamples:    MSAA sample count of the swapchain render pass.
    // frameCount:          number of frames in flight (e.g. MAX_FRAMES_IN_FLIGHT).
    // vertSpv/fragSpv:     absolute paths to compiled SPIR-V (use Application::resolveShaderPath).
    void init(VkDevice              device,
              VmaAllocator          allocator,
              VkExtent2D            extent,
              VkRenderPass          swapchainRenderPass,
              VkSampleCountFlagBits swapchainSamples,
              VkPipelineCache       pipelineCache,
              uint32_t              frameCount,
              const std::string&    vertSpv,
              const std::string&    fragSpv,
              const Config&         cfg = {});

    void destroy();

    // Recreate HDR images and framebuffers for a new swapchain extent.
    // The render passes and pipeline are stable; only per-frame images are rebuilt.
    // Call after vkDeviceWaitIdle() and the swapchain rebuild in handleResize().
    void rebuild(VmaAllocator allocator, VkExtent2D newExtent);

    // HDR offscreen render pass (color + depth, 1x MSAA).
    // The geometry pipeline must be compiled against this render pass.
    VkRenderPass hdrRenderPass() const { return m_hdrRenderPass; }

    // Begin/end the HDR offscreen render pass (1x MSAA, R16G16B16A16_SFLOAT color).
    // beginHdr also sets dynamic viewport + scissor to the HDR extent.
    void beginHdr(VkCommandBuffer cmd, uint32_t frameIndex) const;
    void endHdr  (VkCommandBuffer cmd) const;

    // Record fullscreen tonemap blit inside the active swapchain render pass.
    // Dynamic viewport + scissor must already be set (RenderPass::begin() handles this).
    void blit(VkCommandBuffer cmd, uint32_t frameIndex);

    Config&       config()       { return m_cfg; }
    const Config& config() const { return m_cfg; }

private:
    void createHdrRenderPass();
    void createImages(VmaAllocator allocator, VkExtent2D extent);
    void createFramebuffers();
    void createSampler();
    void createDescriptors();
    void updateDescriptors();
    void createPipeline(VkRenderPass          swapchainRenderPass,
                        VkSampleCountFlagBits  swapchainSamples,
                        VkPipelineCache        cache,
                        const std::string&     vertSpv,
                        const std::string&     fragSpv);
    void destroyImages();
    void destroyFramebuffers();

    VkDevice   m_device     = VK_NULL_HANDLE;
    VkExtent2D m_extent     {};
    uint32_t   m_frameCount = 0;

    // Per-frame HDR color images (one per frame-in-flight to avoid GPU aliasing).
    std::vector<Image>         m_hdrImages;
    // Shared depth image (write + read are sequential within one command buffer).
    Image                      m_depthImage;
    std::vector<VkFramebuffer> m_hdrFramebuffers;

    VkRenderPass m_hdrRenderPass = VK_NULL_HANDLE;
    VkSampler    m_sampler       = VK_NULL_HANDLE;

    // Descriptor resources for the tonemap blit.
    VkDescriptorSetLayout        m_postLayout = VK_NULL_HANDLE;
    DescriptorPool               m_postPool;
    std::vector<VkDescriptorSet> m_postSets;

    // Fullscreen tonemap pipeline.
    VkPipelineLayout m_pipeLayout = VK_NULL_HANDLE;
    VkPipeline       m_pipeline   = VK_NULL_HANDLE;

    Config m_cfg;
};

} // namespace vkt
