#pragma once
#include "resources/Image.h"
#include "resources/Buffer.h"
#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>

namespace vkt {

// One-shot GPU baker: converts an equirectangular HDR image into the three
// IBL textures used by the PBR shading path.
//
// Usage:
//   IblBaker baker;
//   baker.init(device, physDevice, allocator, graphicsQueue, queueFamily,
//              equirectSpv, irradianceSpv, prefilterSpv);
//   auto result = baker.bake(equirectPath);
//   baker.destroy();
//
// The returned images are GPU-only, in SHADER_READ_ONLY_OPTIMAL layout,
// ready to be bound as samplerCube / sampler2D.
class IblBaker {
public:
    struct Config {
        uint32_t envCubeSize      = 512;   // resolution of intermediate env cubemap
        uint32_t irrSize          = 32;    // irradiance cubemap resolution
        uint32_t prefilterSize    = 128;   // prefiltered env base mip resolution
        uint32_t prefilterMips    = 5;     // number of roughness mip levels
        uint32_t prefilterSamples = 1024;  // GGX samples per prefilter texel
        uint32_t lutSize          = 64;    // BRDF LUT resolution (NdotV x roughness)
        uint32_t lutSamples       = 1024;  // Hammersley samples per LUT texel (CPU)
    };

    struct Result {
        Image irradiance;  // irrSize x irrSize x 6, RGBA16F, CUBE view, SHADER_READ_ONLY
        Image prefilter;   // prefilterSize x prefilterSize x 6 + mips, RGBA16F, CUBE view
        Image brdfLut;     // lutSize x lutSize, R16G16_SFLOAT, 2D, SHADER_READ_ONLY
        Image envCubemap;  // envCubeSize x envCubeSize x 6, RGBA16F -- use for sky display
    };

    IblBaker()  = default;
    ~IblBaker() { destroy(); }
    IblBaker(const IblBaker&)            = delete;
    IblBaker& operator=(const IblBaker&) = delete;

    void init(VkDevice          device,
              VkPhysicalDevice  physDevice,
              VmaAllocator      allocator,
              VkQueue           graphicsQueue,
              uint32_t          graphicsQueueFamily,
              const std::string& equirectToSpv,
              const std::string& irradianceSpv,
              const std::string& prefilterSpv,
              const Config&      cfg = {});

    void destroy();

    // Load the .hdr file and run all GPU bake passes.
    // Blocks until all GPU work is complete.
    // Throws std::runtime_error on failure.
    Result bake(const std::string& equirectPath);

private:
    // Shared descriptor set layout: binding 0 = COMBINED_IMAGE_SAMPLER,
    //                               binding 1 = STORAGE_IMAGE (COMPUTE stage).
    VkDescriptorSetLayout m_bakeLayout = VK_NULL_HANDLE;

    // Pipeline layouts differ only in push constant range size.
    // A: faceSize only (4 bytes) -- equirect and irradiance passes.
    // C: faceSize + roughness + sampleCount (12 bytes) -- prefilter pass.
    VkPipelineLayout m_pipelineLayoutA = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayoutC = VK_NULL_HANDLE;

    VkPipeline m_equirectPipeline   = VK_NULL_HANDLE;
    VkPipeline m_irradiancePipeline = VK_NULL_HANDLE;
    VkPipeline m_prefilterPipeline  = VK_NULL_HANDLE;

    VkCommandPool m_cmdPool         = VK_NULL_HANDLE;
    VkSampler     m_equirectSampler = VK_NULL_HANDLE; // linear + clamp for HDR input
    VkSampler     m_cubeSampler     = VK_NULL_HANDLE; // linear + clamp for env cube sampling

    VkDevice         m_device      = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice  = VK_NULL_HANDLE;
    VmaAllocator     m_allocator   = VK_NULL_HANDLE;
    VkQueue          m_queue       = VK_NULL_HANDLE;
    uint32_t         m_queueFamily = 0;
    Config           m_cfg;

    VkShaderModule  loadSpv(const std::string& path) const;
    VkPipeline      createComputePipeline(VkShaderModule shader,
                                          VkPipelineLayout layout) const;
    VkCommandBuffer beginOneShot() const;
    void            endAndSubmit(VkCommandBuffer cmd) const;
};

} // namespace vkt
