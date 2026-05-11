#include "IblBaker.h"

#include <stb_image.h>   // STB_IMAGE_IMPLEMENTATION is in Texture.cpp; just include here

#include <glm/glm.hpp>

#include <stdexcept>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <array>
#include <cassert>

namespace vkt {

// -- CPU math helpers ----------------------------------------------------------

static float radicalInverseCpu(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

static glm::vec2 hammersleyCpu(uint32_t i, uint32_t N) {
    return {static_cast<float>(i) / static_cast<float>(N), radicalInverseCpu(i)};
}

static glm::vec3 importanceSampleGGXCpu(glm::vec2 xi, float roughness) {
    constexpr float PI = 3.14159265359f;
    float a        = roughness * roughness;
    float phi      = 2.0f * PI * xi.x;
    float cosTheta = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
    float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    return {std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};
}

// Integrate the GGX split-sum BRDF for a given NdotV and roughness.
// Returns (scale, bias) that encode the F0-independent part of the reflectance.
static glm::vec2 integrateBRDF(float NdotV, float roughness, uint32_t samples) {
    // V lies in the tangent hemisphere; N = (0,0,1) in tangent space.
    glm::vec3 V{std::sqrt(1.0f - NdotV * NdotV), 0.0f, NdotV};
    float A = 0.0f, B = 0.0f;

    for (uint32_t i = 0; i < samples; ++i) {
        glm::vec2 xi = hammersleyCpu(i, samples);
        glm::vec3 H  = importanceSampleGGXCpu(xi, roughness);
        // In tangent space N=(0,0,1), so the transform is identity.
        glm::vec3 L  = glm::normalize(2.0f * glm::dot(V, H) * H - V);

        float NdotL = glm::clamp(L.z,            0.0f, 1.0f);
        float NdotH = glm::clamp(H.z,            0.0f, 1.0f);
        float VdotH = glm::clamp(glm::dot(V, H), 0.0f, 1.0f);

        if (NdotL > 0.0f) {
            // Smith GGX geometry in IBL form: k = roughness^2 / 2
            float k    = (roughness * roughness) * 0.5f;
            float gV   = NdotV / (NdotV * (1.0f - k) + k);
            float gL   = NdotL / (NdotL * (1.0f - k) + k);
            float G    = gV * gL;
            float Gvis = (G * VdotH) / (NdotH * NdotV + 1e-7f);
            float Fc   = std::pow(1.0f - VdotH, 5.0f);
            A += (1.0f - Fc) * Gvis;
            B += Fc * Gvis;
        }
    }

    float fSamples = static_cast<float>(samples);
    return {A / fSamples, B / fSamples};
}

// IEEE 754 float -> half-float (round to nearest, no denormal rounding to zero).
static uint16_t floatToHalf(float v) {
    uint32_t f;
    std::memcpy(&f, &v, 4);
    uint32_t sign = (f >> 31u) & 0x1u;
    uint32_t exp  = (f >> 23u) & 0xFFu;
    uint32_t mant =  f         & 0x7FFFFFu;

    if (exp == 255u) // NaN or Inf
        return static_cast<uint16_t>((sign << 15u) | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp == 0u)   // zero or denormal -- treat as zero
        return static_cast<uint16_t>(sign << 15u);

    int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31) // overflow -> half Inf
        return static_cast<uint16_t>((sign << 15u) | 0x7C00u);
    if (e <= 0) { // underflow -> half denormal or zero
        if (e < -10)
            return static_cast<uint16_t>(sign << 15u);
        mant = (mant | 0x800000u) >> static_cast<uint32_t>(1 - e);
        return static_cast<uint16_t>((sign << 15u) | (mant >> 13u));
    }
    return static_cast<uint16_t>((sign << 15u) | (static_cast<uint32_t>(e) << 10u) | (mant >> 13u));
}

// -- IblBaker::init ------------------------------------------------------------

void IblBaker::init(VkDevice         device,
                    VkPhysicalDevice  physDevice,
                    VmaAllocator      allocator,
                    VkQueue           graphicsQueue,
                    uint32_t          graphicsQueueFamily,
                    const std::string& equirectToSpv,
                    const std::string& irradianceSpv,
                    const std::string& prefilterSpv,
                    const Config&      cfg)
{
    m_device      = device;
    m_physDevice  = physDevice;
    m_allocator   = allocator;
    m_queue       = graphicsQueue;
    m_queueFamily = graphicsQueueFamily;
    m_cfg         = cfg;

    // -- Shared descriptor set layout ------------------------------------------
    // binding 0: COMBINED_IMAGE_SAMPLER (equirect or env cube input)
    // binding 1: STORAGE_IMAGE          (output image array written by compute)
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings    = bindings.data();
        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_bakeLayout) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create descriptor set layout");
    }

    // -- Pipeline layout A: push constant = { uint faceSize } (4 bytes) ---------
    {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(uint32_t); // faceSize

        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount         = 1;
        plInfo.pSetLayouts            = &m_bakeLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges    = &pcRange;
        if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_pipelineLayoutA) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create pipeline layout A");
    }

    // -- Pipeline layout C: push constant = { uint faceSize; float roughness; uint sampleCount }
    //    Total 12 bytes, used for the prefilter pass.
    {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(uint32_t) + sizeof(float) + sizeof(uint32_t); // 12 bytes

        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount         = 1;
        plInfo.pSetLayouts            = &m_bakeLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges    = &pcRange;
        if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_pipelineLayoutC) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create pipeline layout C");
    }

    // -- Compute pipelines ------------------------------------------------------
    {
        VkShaderModule eqMod  = loadSpv(equirectToSpv);
        VkShaderModule irrMod = loadSpv(irradianceSpv);
        VkShaderModule pfMod  = loadSpv(prefilterSpv);

        m_equirectPipeline   = createComputePipeline(eqMod,  m_pipelineLayoutA);
        m_irradiancePipeline = createComputePipeline(irrMod, m_pipelineLayoutA);
        m_prefilterPipeline  = createComputePipeline(pfMod,  m_pipelineLayoutC);

        vkDestroyShaderModule(m_device, eqMod,  nullptr);
        vkDestroyShaderModule(m_device, irrMod, nullptr);
        vkDestroyShaderModule(m_device, pfMod,  nullptr);
    }

    // -- Transient command pool -------------------------------------------------
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_queueFamily;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create command pool");
    }

    // -- Sampler for the equirectangular HDR input (2D, linear, clamp) ----------
    {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.anisotropyEnable = VK_FALSE;
        si.maxAnisotropy    = 1.0f;
        si.minLod           = 0.0f;
        si.maxLod           = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(m_device, &si, nullptr, &m_equirectSampler) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create equirect sampler");
    }

    // -- Sampler for the intermediate env cubemap (cube, linear, clamp) ---------
    {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.anisotropyEnable = VK_FALSE;
        si.maxAnisotropy    = 1.0f;
        si.minLod           = 0.0f;
        si.maxLod           = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(m_device, &si, nullptr, &m_cubeSampler) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create cube sampler");
    }
}

// -- IblBaker::destroy ---------------------------------------------------------

void IblBaker::destroy() {
    if (m_device == VK_NULL_HANDLE) return;

    if (m_equirectSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_equirectSampler, nullptr);
        m_equirectSampler = VK_NULL_HANDLE;
    }
    if (m_cubeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_cubeSampler, nullptr);
        m_cubeSampler = VK_NULL_HANDLE;
    }
    if (m_cmdPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
        m_cmdPool = VK_NULL_HANDLE;
    }
    if (m_prefilterPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_prefilterPipeline, nullptr);
        m_prefilterPipeline = VK_NULL_HANDLE;
    }
    if (m_irradiancePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_irradiancePipeline, nullptr);
        m_irradiancePipeline = VK_NULL_HANDLE;
    }
    if (m_equirectPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_equirectPipeline, nullptr);
        m_equirectPipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayoutC != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayoutC, nullptr);
        m_pipelineLayoutC = VK_NULL_HANDLE;
    }
    if (m_pipelineLayoutA != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayoutA, nullptr);
        m_pipelineLayoutA = VK_NULL_HANDLE;
    }
    if (m_bakeLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_bakeLayout, nullptr);
        m_bakeLayout = VK_NULL_HANDLE;
    }

    m_device = VK_NULL_HANDLE;
}

// -- IblBaker::bake ------------------------------------------------------------

IblBaker::Result IblBaker::bake(const std::string& equirectPath) {
    assert(m_device != VK_NULL_HANDLE && "[IblBaker] bake() called before init()");

    // -------------------------------------------------------------------------
    // Step 1: Load equirectangular HDR image from disk (linear, float RGBA).
    // stbi_loadf decodes Radiance .hdr RGBE -> linear float; do NOT gamma-correct.
    // -------------------------------------------------------------------------
    int w = 0, h = 0, srcChannels = 0;
    float* hdrData = stbi_loadf(equirectPath.c_str(), &w, &h, &srcChannels, 4);
    if (!hdrData)
        throw std::runtime_error("[IblBaker] Failed to load HDR: " + equirectPath
                                 + " -- " + stbi_failure_reason());

    const VkDeviceSize equirectBytes = static_cast<VkDeviceSize>(w) * h * 4 * sizeof(float);

    // -------------------------------------------------------------------------
    // Step 2: Upload equirect to a GPU image (R32G32B32A32_SFLOAT).
    // -------------------------------------------------------------------------
    Buffer equirectStaging;
    equirectStaging.createCpuVisible(m_allocator, equirectBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    equirectStaging.uploadData(hdrData, equirectBytes);
    stbi_image_free(hdrData);
    hdrData = nullptr;

    Image equirectImg;
    equirectImg.create(m_allocator, m_device, {
        static_cast<uint32_t>(w),
        static_cast<uint32_t>(h),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        1   // mipLevels
    });

    // -------------------------------------------------------------------------
    // Step 3: Create intermediate env cubemap and output IBL images.
    // -------------------------------------------------------------------------
    Image envCube;
    envCube.create(m_allocator, m_device, {
        m_cfg.envCubeSize, m_cfg.envCubeSize,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        1,  // mipLevels
        VK_SAMPLE_COUNT_1_BIT,
        6,  // 6 cubemap faces
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
    });

    Image irrImg;
    irrImg.create(m_allocator, m_device, {
        m_cfg.irrSize, m_cfg.irrSize,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        6,
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
    });

    Image pfImg;
    pfImg.create(m_allocator, m_device, {
        m_cfg.prefilterSize, m_cfg.prefilterSize,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        m_cfg.prefilterMips,
        VK_SAMPLE_COUNT_1_BIT,
        6,
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
    });

    // -------------------------------------------------------------------------
    // Step 4: Create 2D_ARRAY image views for storage writes.
    // The Image class creates a CUBE view; compute shaders need 2D_ARRAY views.
    // -------------------------------------------------------------------------
    auto makeArrayView = [&](VkImage image, VkFormat fmt, uint32_t baseMip,
                              uint32_t mipCount, uint32_t layerCount) -> VkImageView {
        VkImageViewCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image                           = image;
        vi.viewType                        = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vi.format                          = fmt;
        vi.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel   = baseMip;
        vi.subresourceRange.levelCount     = mipCount;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount     = layerCount;
        VkImageView v = VK_NULL_HANDLE;
        if (vkCreateImageView(m_device, &vi, nullptr, &v) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create 2D_ARRAY image view");
        return v;
    };

    VkImageView envCubeArrayView = makeArrayView(envCube.handle(),
        VK_FORMAT_R16G16B16A16_SFLOAT, 0, 1, 6);
    VkImageView irrArrayView = makeArrayView(irrImg.handle(),
        VK_FORMAT_R16G16B16A16_SFLOAT, 0, 1, 6);

    // One per-mip 2D_ARRAY view for prefilter output.
    std::vector<VkImageView> pfMipViews(m_cfg.prefilterMips);
    for (uint32_t mip = 0; mip < m_cfg.prefilterMips; ++mip) {
        pfMipViews[mip] = makeArrayView(pfImg.handle(),
            VK_FORMAT_R16G16B16A16_SFLOAT, mip, 1, 6);
    }

    // -------------------------------------------------------------------------
    // Step 5: Create a private descriptor pool for this bake session.
    // -------------------------------------------------------------------------
    VkDescriptorPool bakePool = VK_NULL_HANDLE;
    {
        std::array<VkDescriptorPoolSize, 2> sizes{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = 10;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = 10 + m_cfg.prefilterMips;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets       = 20 + m_cfg.prefilterMips;
        poolInfo.poolSizeCount = static_cast<uint32_t>(sizes.size());
        poolInfo.pPoolSizes    = sizes.data();
        if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &bakePool) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to create descriptor pool");
    }

    // Helper: allocate one descriptor set from the bake pool.
    auto allocSet = [&]() -> VkDescriptorSet {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = bakePool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &m_bakeLayout;
        VkDescriptorSet s = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(m_device, &ai, &s) != VK_SUCCESS)
            throw std::runtime_error("[IblBaker] Failed to allocate descriptor set");
        return s;
    };

    // Helper: write a COMBINED_IMAGE_SAMPLER at binding 0.
    auto writeSampledImage = [&](VkDescriptorSet set, VkImageView view,
                                  VkSampler sampler, VkImageLayout layout) {
        VkDescriptorImageInfo imgInfo{};
        imgInfo.sampler     = sampler;
        imgInfo.imageView   = view;
        imgInfo.imageLayout = layout;

        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = set;
        w.dstBinding      = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo      = &imgInfo;
        vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    };

    // Helper: write a STORAGE_IMAGE at binding 1.
    auto writeStorageImage = [&](VkDescriptorSet set, VkImageView view) {
        VkDescriptorImageInfo imgInfo{};
        imgInfo.sampler     = VK_NULL_HANDLE; // no sampler for storage image
        imgInfo.imageView   = view;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = set;
        w.dstBinding      = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.descriptorCount = 1;
        w.pImageInfo      = &imgInfo;
        vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    };

    // -------------------------------------------------------------------------
    // Step 6: Allocate and write descriptor sets for all passes.
    // -------------------------------------------------------------------------

    // Equirect -> env cube pass.
    VkDescriptorSet equirectSet = allocSet();
    writeSampledImage(equirectSet, equirectImg.view(), m_equirectSampler,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeStorageImage(equirectSet, envCubeArrayView);

    // Irradiance convolution pass.
    VkDescriptorSet irrSet = allocSet();
    writeSampledImage(irrSet, envCube.view(), m_cubeSampler,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeStorageImage(irrSet, irrArrayView);

    // Prefilter pass: one descriptor set per mip level.
    std::vector<VkDescriptorSet> pfSets(m_cfg.prefilterMips);
    for (uint32_t mip = 0; mip < m_cfg.prefilterMips; ++mip) {
        pfSets[mip] = allocSet();
        writeSampledImage(pfSets[mip], envCube.view(), m_cubeSampler,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        writeStorageImage(pfSets[mip], pfMipViews[mip]);
    }

    // -------------------------------------------------------------------------
    // Step 7: Record and submit the compute command buffer.
    // -------------------------------------------------------------------------
    VkCommandBuffer cmd = beginOneShot();

    // Helper: generic compute -> compute memory barrier on a single image.
    auto computeBarrier = [&](VkImage image,
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               uint32_t mipCount, uint32_t layerCount,
                               uint32_t baseMip = 0) {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout           = oldLayout;
        b.newLayout           = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = image;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, layerCount};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // -- Upload equirect from staging to GPU image -----------------------------
    Image::transitionLayout(cmd, equirectImg.handle(),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    Image::copyFromBuffer(cmd, equirectStaging.handle(), equirectImg.handle(),
                          static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    Image::transitionLayout(cmd, equirectImg.handle(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        1, VK_IMAGE_ASPECT_COLOR_BIT, 1);

    // -- Pass 1: equirect -> env cube (RGBA16F, GENERAL for storage write) -----
    Image::transitionLayout(cmd, envCube.handle(),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        1, VK_IMAGE_ASPECT_COLOR_BIT, 6);

    {
        uint32_t pushData = m_cfg.envCubeSize;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_equirectPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipelineLayoutA, 0, 1, &equirectSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayoutA, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(uint32_t), &pushData);
        uint32_t gx = (m_cfg.envCubeSize + 15u) / 16u;
        uint32_t gy = (m_cfg.envCubeSize + 15u) / 16u;
        vkCmdDispatch(cmd, gx, gy, 6);
    }

    // Barrier: wait for env cube storage writes before transitioning for sampling.
    computeBarrier(envCube.handle(),
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   1, 6);

    // -- Pass 2: irradiance convolution ----------------------------------------
    Image::transitionLayout(cmd, irrImg.handle(),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        1, VK_IMAGE_ASPECT_COLOR_BIT, 6);

    {
        uint32_t pushData = m_cfg.irrSize;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_irradiancePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipelineLayoutA, 0, 1, &irrSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayoutA, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(uint32_t), &pushData);
        uint32_t gx = (m_cfg.irrSize + 7u) / 8u;
        uint32_t gy = (m_cfg.irrSize + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 6);
    }

    // Barrier + transition irradiance to shader-read for the final image.
    computeBarrier(irrImg.handle(),
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   1, 6);

    // -- Pass 3: prefiltered env map (one dispatch per roughness mip) ----------
    for (uint32_t mip = 0; mip < m_cfg.prefilterMips; ++mip) {
        // Transition this mip from UNDEFINED to GENERAL.
        {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask       = 0;
            b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = pfImg.handle();
            b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6};
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
        }

        uint32_t mipSize   = std::max(1u, m_cfg.prefilterSize >> mip);
        float    roughness = (m_cfg.prefilterMips > 1)
                             ? static_cast<float>(mip) / static_cast<float>(m_cfg.prefilterMips - 1)
                             : 0.0f;

        // Push constants for prefilter pass: { uint faceSize; float roughness; uint sampleCount }
        struct PrefilterPush {
            uint32_t faceSize;
            float    roughness;
            uint32_t sampleCount;
        } push{mipSize, roughness, m_cfg.prefilterSamples};

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prefilterPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipelineLayoutC, 0, 1, &pfSets[mip], 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayoutC, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PrefilterPush), &push);

        uint32_t gx = (mipSize + 7u) / 8u;
        uint32_t gy = (mipSize + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 6);

        // Barrier after each mip dispatch before reading/writing the next mip.
        {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            b.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL; // keep GENERAL until final transition
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = pfImg.handle();
            b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6};
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
        }
    }

    // Transition all prefilter mips from GENERAL to SHADER_READ_ONLY_OPTIMAL.
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = pfImg.handle();
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, m_cfg.prefilterMips, 0, 6};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
    }

    endAndSubmit(cmd);

    // -------------------------------------------------------------------------
    // Step 8: CPU compute BRDF LUT (R16G16_SFLOAT, NdotV x roughness).
    // -------------------------------------------------------------------------
    const uint32_t N = m_cfg.lutSize;
    std::vector<uint16_t> lutData(static_cast<size_t>(N) * N * 2); // R16G16: 2 channels per texel

    for (uint32_t y = 0; y < N; ++y) {
        // y-axis = roughness (0 = smooth, 1 = rough)
        float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
        roughness = std::max(roughness, 0.01f);
        for (uint32_t x = 0; x < N; ++x) {
            // x-axis = NdotV (0 = grazing, 1 = head-on)
            float NdotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
            NdotV = std::max(NdotV, 0.001f);

            glm::vec2 ab = integrateBRDF(NdotV, roughness, m_cfg.lutSamples);
            uint32_t idx = (y * N + x) * 2;
            lutData[idx + 0] = floatToHalf(glm::clamp(ab.x, 0.0f, 1.0f));
            lutData[idx + 1] = floatToHalf(glm::clamp(ab.y, 0.0f, 1.0f));
        }
    }

    // Upload BRDF LUT to GPU.
    const VkDeviceSize lutBytes = static_cast<VkDeviceSize>(N) * N * 2 * sizeof(uint16_t);
    Buffer lutStaging;
    lutStaging.createCpuVisible(m_allocator, lutBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    lutStaging.uploadData(lutData.data(), lutBytes);

    Image brdfLutImg;
    brdfLutImg.create(m_allocator, m_device, {
        N, N,
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        1
    });

    {
        VkCommandBuffer uploadCmd = beginOneShot();
        Image::transitionLayout(uploadCmd, brdfLutImg.handle(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        Image::copyFromBuffer(uploadCmd, lutStaging.handle(), brdfLutImg.handle(), N, N);
        Image::transitionLayout(uploadCmd, brdfLutImg.handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        endAndSubmit(uploadCmd);
    }

    // -------------------------------------------------------------------------
    // Step 9: Cleanup transient resources.
    // -------------------------------------------------------------------------
    vkDestroyDescriptorPool(m_device, bakePool, nullptr);

    for (VkImageView v : pfMipViews)
        vkDestroyImageView(m_device, v, nullptr);
    vkDestroyImageView(m_device, irrArrayView,     nullptr);
    vkDestroyImageView(m_device, envCubeArrayView, nullptr);

    // equirectImg is temporary; envCube is returned as the sky display texture.
    equirectImg.destroy();
    equirectStaging.destroy();
    lutStaging.destroy();

    // -------------------------------------------------------------------------
    // Step 10: Return baked IBL images.
    // -------------------------------------------------------------------------
    Result result;
    result.irradiance  = std::move(irrImg);
    result.prefilter   = std::move(pfImg);
    result.brdfLut     = std::move(brdfLutImg);
    result.envCubemap  = std::move(envCube);
    return result;
}

// -- IblBaker::loadSpv ---------------------------------------------------------

VkShaderModule IblBaker::loadSpv(const std::string& path) const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("[IblBaker] Cannot open shader: " + path);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> code(static_cast<size_t>(size + 3) / 4, 0);
    if (!file.read(reinterpret_cast<char*>(code.data()), size))
        throw std::runtime_error("[IblBaker] Failed to read shader: " + path);

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = static_cast<size_t>(size);
    info.pCode    = code.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &info, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("[IblBaker] Failed to create shader module: " + path);
    return mod;
}

// -- IblBaker::createComputePipeline -------------------------------------------

VkPipeline IblBaker::createComputePipeline(VkShaderModule shader,
                                             VkPipelineLayout layout) const {
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage  = stageInfo;
    pipelineInfo.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline)
            != VK_SUCCESS)
        throw std::runtime_error("[IblBaker] Failed to create compute pipeline");
    return pipeline;
}

// -- IblBaker::beginOneShot / endAndSubmit -------------------------------------

VkCommandBuffer IblBaker::beginOneShot() const {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = m_cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &ai, &cmd) != VK_SUCCESS)
        throw std::runtime_error("[IblBaker] Failed to allocate command buffer");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

void IblBaker::endAndSubmit(VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(m_device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
        throw std::runtime_error("[IblBaker] Failed to create fence");

    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(m_queue, 1, &submit, fence);
    vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(m_device, fence, nullptr);
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &cmd);
}

} // namespace vkt
