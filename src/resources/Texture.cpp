// STB_IMAGE_IMPLEMENTATION must be defined in exactly one translation unit.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Texture.h"

#include <stdexcept>
#include <cstdio>

namespace vkt {

void Texture::loadFromFile(VmaAllocator allocator, VkDevice device, VkPhysicalDevice physDevice,
                            VkCommandBuffer cmd, Buffer& outStaging,
                            const std::string& path) {
    std::scoped_lock lock(m_mutex);
    // STBI_rgb_alpha forces 4-channel output regardless of source format, so
    // the GPU format is always VK_FORMAT_R8G8B8A8_SRGB  --  no per-format branching.
    int w, h, srcChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &srcChannels, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error("[Texture] Failed to load '" + path + "': "
                                 + stbi_failure_reason());

    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(w) * h * 4;
    const VkFormat texFormat = VK_FORMAT_R8G8B8A8_SRGB;
    VkFormatProperties formatProps{};
    vkGetPhysicalDeviceFormatProperties(physDevice, texFormat, &formatProps);
    const bool supportsLinearBlit =
        (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    const uint32_t mipLevels = supportsLinearBlit
        ? Image::calcMipLevels(static_cast<uint32_t>(w), static_cast<uint32_t>(h))
        : 1u;
    if (!supportsLinearBlit)
        std::printf("[Texture] '%s' linear blit unsupported; using a single mip level.\n", path.c_str());

    // Upload pixel data into the staging buffer
    outStaging.createCpuVisible(allocator, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    outStaging.uploadData(pixels, imageSize);
    stbi_image_free(pixels);

    std::printf("[Texture] Loaded '%s' %dx%d, %u mip levels\n",
                path.c_str(), w, h, mipLevels);

    // Create the GPU image
    m_image.create(allocator, device, {
        static_cast<uint32_t>(w),
        static_cast<uint32_t>(h),
        texFormat,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |  // source for blit-based mip generation
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |  // destination for the staging copy
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        mipLevels
    });

    // Record upload + mip generation into the caller-provided command buffer.
    // generateMipmaps transitions all levels to SHADER_READ_ONLY_OPTIMAL.
    Image::transitionLayout(cmd, m_image.handle(),
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    Image::copyFromBuffer(cmd, outStaging.handle(), m_image.handle(),
                          static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    if (mipLevels > 1) {
        Image::generateMipmaps(cmd, m_image.handle(),
                               static_cast<uint32_t>(w), static_cast<uint32_t>(h), mipLevels);
    } else {
        Image::transitionLayout(cmd, m_image.handle(),
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    }

    // Linear + trilinear mipmaps + anisotropic filtering
    m_sampler.create(device, physDevice,
                     Sampler::linearConfig(static_cast<float>(mipLevels)));
}

void Texture::destroy() {
    std::scoped_lock lock(m_mutex);
    m_sampler.destroy();
    m_image.destroy();
}

Texture::~Texture() { destroy(); }

} // namespace vkt
