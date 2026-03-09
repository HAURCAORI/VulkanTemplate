#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>
#include <mutex>

namespace vkt {

// RAII wrapper around a VkImage + VkImageView backed by a VmaAllocation.
// Always device-local (GPU-only); CPU upload goes through a staging Buffer.
// Move-only (mutex semantics).
class Image {
public:
    Image() = default;
    ~Image();
    Image(const Image&)            = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& o) noexcept;
    Image& operator=(Image&& o) noexcept;

    struct CreateInfo {
        uint32_t             width;
        uint32_t             height;
        VkFormat             format;
        VkImageUsageFlags    usage;
        VkImageAspectFlags   aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
        uint32_t             mipLevels  = 1;
        VkSampleCountFlagBits samples   = VK_SAMPLE_COUNT_1_BIT;
    };

    void create(VmaAllocator allocator, VkDevice device, const CreateInfo& info);
    void destroy();

    VkImage     handle()    const noexcept { return m_image; }
    VkImageView view()      const noexcept { return m_view; }
    VkFormat    format()    const noexcept { return m_format; }
    uint32_t    width()     const noexcept { return m_width; }
    uint32_t    height()    const noexcept { return m_height; }
    uint32_t    mipLevels() const noexcept { return m_mipLevels; }

    // Inserts a VkImageMemoryBarrier for the full mip chain.
    // aspectMask: COLOR (default) or DEPTH_BIT / DEPTH_BIT|STENCIL_BIT for depth images.
    // Unknown layout pairs fall back to a conservative ALL_COMMANDS barrier.
    static void transitionLayout(VkCommandBuffer cmd, VkImage image,
                                 VkImageLayout oldLayout, VkImageLayout newLayout,
                                 uint32_t mipLevels = 1,
                                 VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);

    // Copy a buffer's contents into this image (image must be in TRANSFER_DST_OPTIMAL)
    static void copyFromBuffer(VkCommandBuffer cmd, VkBuffer src, VkImage dst,
                               uint32_t width, uint32_t height);

    // Generate the full mip chain for an image using blit operations.
    // The image must have been created with VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT and mipLevels > 1.
    // After this call the image is in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    static void generateMipmaps(VkCommandBuffer cmd, VkImage image,
                                 uint32_t width, uint32_t height, uint32_t mipLevels);

    // Number of mip levels needed to fully mip an image of the given dimensions.
    static uint32_t calcMipLevels(uint32_t width, uint32_t height);

private:
    VmaAllocator  m_allocator  = VK_NULL_HANDLE;
    VkDevice      m_device     = VK_NULL_HANDLE;
    VkImage       m_image      = VK_NULL_HANDLE;
    VkImageView   m_view       = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkFormat      m_format     = VK_FORMAT_UNDEFINED;
    uint32_t      m_width      = 0;
    uint32_t      m_height     = 0;
    uint32_t      m_mipLevels  = 1;
    mutable std::mutex m_mutex;
};

} // namespace vkt
