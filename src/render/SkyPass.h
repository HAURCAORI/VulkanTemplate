#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <string>

namespace vkt {

enum class SkyMode {
    None,       // no sky rendered
    Color,      // simple gradient/solid color sky
    Skybox,     // cubemap skybox
    SkySphere,  // procedural sky sphere distinct from the simple color gradient
};

// SkyPass renders a background sky before the geometry pass inside an active
// HDR render pass.  It draws a unit cube at infinity (depth forced to far plane)
// and either samples a cubemap (SkyMode::Skybox) or generates a gradient.
//
// Usage:
//   skyPass.init(device, allocator, globalLayout, descriptorPool,
//                pipelineCache, hdrRenderPass, samples,
//                "skybox.vert.spv", "skybox.frag.spv", cfg);
//   // Inside HDR render pass:
//   skyPass.draw(cmd, globalSet);
class SkyPass {
public:
    struct Config {
        SkyMode    mode     = SkyMode::Color;
        glm::vec3  skyColor = {0.05f, 0.15f, 0.3f};  // top color for gradient mode
    };

    SkyPass()  = default;
    ~SkyPass() { destroy(); }
    SkyPass(const SkyPass&)            = delete;
    SkyPass& operator=(const SkyPass&) = delete;

    // Initialize the sky pipeline and placeholder cubemap.
    // globalLayout:        set=0 layout containing the GlobalUBO (binding 0).
    // descriptorPool:      pool from which the internal sky set (set=1) is allocated.
    // hdrRenderPass:       the render pass the sky will draw into (must be active during draw()).
    // graphicsQueueFamily: queue family index used for the placeholder upload command pool.
    void init(VkDevice device,
              VmaAllocator allocator,
              VkDescriptorSetLayout globalLayout,
              VkDescriptorPool descriptorPool,
              VkPipelineCache cache,
              VkRenderPass hdrRenderPass,
              VkSampleCountFlagBits samples,
              const std::string& vertSpv,
              const std::string& fragSpv,
              uint32_t graphicsQueueFamily = 0,
              const Config& cfg = {});

    void destroy();

    // Record sky draw commands.  Must be called inside the HDR render pass,
    // before geometry draws so the sky is behind all opaque objects.
    void draw(VkCommandBuffer cmd, VkDescriptorSet globalSet);

    // -- Runtime configuration -------------------------------------------------
    void    setMode(SkyMode mode)       { m_cfg.mode = mode; }
    void    setSkyColor(glm::vec3 c)    { m_cfg.skyColor = c; }
    SkyMode mode()              const   { return m_cfg.mode; }
    const glm::vec3& skyColor() const   { return m_cfg.skyColor; }

    // Bind an external cubemap to the sky descriptor and switch to Skybox mode.
    // cubeView must be a VK_IMAGE_VIEW_TYPE_CUBE view in SHADER_READ_ONLY_OPTIMAL.
    // The caller retains ownership of the image and sampler.
    void setCubemap(VkImageView cubeView, VkSampler sampler);

    // -- Probe recapture hooks (A4) --------------------------------------------
    // Call requestRecapture() when the sky changes (color, texture, etc.)
    // to signal that any cached IBL probe is stale and should be re-captured.
    void requestRecapture()     { m_capturePending = true; }
    bool isCaptureStale() const { return m_capturePending; }
    void markCaptureComplete()  { m_capturePending = false; }

private:
    void createSkyLayout(VkDevice device);
    void createPlaceholderCubemap(VkDevice device, VmaAllocator allocator,
                                  VkDescriptorPool descriptorPool,
                                  uint32_t graphicsQueueFamily);
    void createPipeline(VkDevice device, VkRenderPass renderPass,
                        VkSampleCountFlagBits samples, VkPipelineCache cache,
                        VkDescriptorSetLayout globalLayout,
                        const std::string& vertSpv, const std::string& fragSpv);

    VkDevice     m_device    = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    VkPipeline       m_pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;

    // set=1 layout: binding 0 = samplerCube (fragment stage)
    VkDescriptorSetLayout m_skyLayout = VK_NULL_HANDLE;
    VkDescriptorSet       m_skySet    = VK_NULL_HANDLE;

    // Placeholder 1x1 black cubemap used in Color mode or when no cubemap is loaded.
    VkImage       m_placeholderCube     = VK_NULL_HANDLE;
    VkImageView   m_placeholderCubeView = VK_NULL_HANDLE;
    VmaAllocation m_placeholderAlloc    = VK_NULL_HANDLE;
    VkSampler     m_cubeSampler         = VK_NULL_HANDLE;

    Config m_cfg;
    bool   m_capturePending = false;
};

} // namespace vkt
