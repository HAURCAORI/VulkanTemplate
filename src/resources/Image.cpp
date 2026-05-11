#include "Image.h"

#include <stdexcept>
#include <vector>

namespace vkt {

// -- Move semantics -------------------------------------------------------------

Image::Image(Image&& o) noexcept
    : m_allocator(VK_NULL_HANDLE), m_device(VK_NULL_HANDLE), m_image(VK_NULL_HANDLE),
      m_view(VK_NULL_HANDLE), m_allocation(VK_NULL_HANDLE), m_format(VK_FORMAT_UNDEFINED),
      m_width(0), m_height(0), m_mipLevels(1), m_arrayLayers(1)
{
    std::scoped_lock lock(m_mutex, o.m_mutex);
    m_allocator   = o.m_allocator;
    m_device      = o.m_device;
    m_image       = o.m_image;
    m_view        = o.m_view;
    m_allocation  = o.m_allocation;
    m_format      = o.m_format;
    m_width       = o.m_width;
    m_height      = o.m_height;
    m_mipLevels   = o.m_mipLevels;
    m_arrayLayers = o.m_arrayLayers;
    o.m_image       = VK_NULL_HANDLE;
    o.m_view        = VK_NULL_HANDLE;
    o.m_allocation  = VK_NULL_HANDLE;
    o.m_allocator   = VK_NULL_HANDLE;
    o.m_device      = VK_NULL_HANDLE;
    o.m_width       = 0;
    o.m_height      = 0;
    o.m_mipLevels   = 1;
    o.m_arrayLayers = 1;
}

Image& Image::operator=(Image&& o) noexcept {
    if (this != &o) {
        std::scoped_lock lock(m_mutex, o.m_mutex);
        if (m_view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_view, nullptr);
            m_view = VK_NULL_HANDLE;
        }
        if (m_image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_allocator, m_image, m_allocation);
            m_image      = VK_NULL_HANDLE;
            m_allocation = VK_NULL_HANDLE;
        }
        m_allocator   = o.m_allocator;
        m_device      = o.m_device;
        m_image       = o.m_image;
        m_view        = o.m_view;
        m_allocation  = o.m_allocation;
        m_format      = o.m_format;
        m_width       = o.m_width;
        m_height      = o.m_height;
        m_mipLevels   = o.m_mipLevels;
        m_arrayLayers = o.m_arrayLayers;
        o.m_image       = VK_NULL_HANDLE;
        o.m_view        = VK_NULL_HANDLE;
        o.m_allocation  = VK_NULL_HANDLE;
        o.m_allocator   = VK_NULL_HANDLE;
        o.m_device      = VK_NULL_HANDLE;
        o.m_width       = 0;
        o.m_height      = 0;
        o.m_mipLevels   = 1;
        o.m_arrayLayers = 1;
    }
    return *this;
}

// -- Create ---------------------------------------------------------------------

void Image::create(VmaAllocator allocator, VkDevice device, const CreateInfo& info) {
    std::scoped_lock lock(m_mutex);
    m_allocator   = allocator;
    m_device      = device;
    m_format      = info.format;
    m_width       = info.width;
    m_height      = info.height;
    m_mipLevels   = info.mipLevels;
    m_arrayLayers = info.arrayLayers;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags         = info.createFlags;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = info.format;
    imageInfo.extent        = {info.width, info.height, 1};
    imageInfo.mipLevels     = info.mipLevels;
    imageInfo.arrayLayers   = info.arrayLayers;
    imageInfo.samples       = info.samples;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = info.usage;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage         = VMA_MEMORY_USAGE_AUTO;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_image, &m_allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("[Image] Failed to create image");

    // Choose view type: cubemap if 6 layers with CUBE_COMPATIBLE flag, otherwise 2D.
    const bool isCube = (info.arrayLayers == 6) &&
                        ((info.createFlags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = m_image;
    viewInfo.viewType                        = isCube ? VK_IMAGE_VIEW_TYPE_CUBE
                                                      : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = info.format;
    viewInfo.subresourceRange.aspectMask     = info.aspect;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = info.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = info.arrayLayers;

    if (vkCreateImageView(device, &viewInfo, nullptr, &m_view) != VK_SUCCESS)
        throw std::runtime_error("[Image] Failed to create image view");
}

void Image::destroy() {
    std::scoped_lock lock(m_mutex);
    if (m_view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_view, nullptr);
        m_view = VK_NULL_HANDLE;
    }
    if (m_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
        m_image      = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
    }
}

Image::~Image() { destroy(); }

// -- Static helpers -------------------------------------------------------------

void Image::transitionLayout(VkCommandBuffer cmd, VkImage image,
                              VkImageLayout oldLayout, VkImageLayout newLayout,
                              uint32_t mipLevels, VkImageAspectFlags aspectMask,
                              uint32_t layerCount) {
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = {aspectMask, 0, mipLevels, 0, layerCount};

    VkPipelineStageFlags srcStage{};
    VkPipelineStageFlags dstStage{};

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    // -- Shadow map transitions ------------------------------------------------
    // UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    //   Used when first creating a shadow map depth image each frame.
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
               newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                              | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    //   Used after the shadow pass to let the main pass sample the depth image.
    } else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    // SHADER_READ_ONLY_OPTIMAL -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    //   Used at the start of the next frame to reclaim the shadow map for writing.
    } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                              | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

    } else {
        // Conservative fallback: stalls the full pipeline.  Correct for any
        // transition pair but slower than a targeted barrier  --  add a specific
        // case above if this shows up as a bottleneck in a GPU profiler.
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
}

void Image::copyFromBuffer(VkCommandBuffer cmd, VkBuffer src, VkImage dst,
                            uint32_t width, uint32_t height) {
    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset       = {0, 0, 0};
    region.imageExtent       = {width, height, 1};

    vkCmdCopyBufferToImage(cmd, src, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void Image::copyFromBufferLayers(VkCommandBuffer cmd, VkBuffer src, VkImage dst,
                                  uint32_t width, uint32_t height, uint32_t layerCount,
                                  VkDeviceSize bytesPerLayer) {
    // Default bytes per layer: tightly-packed RGBA8 pixels.
    if (bytesPerLayer == 0)
        bytesPerLayer = static_cast<VkDeviceSize>(width) * height * 4;

    std::vector<VkBufferImageCopy> regions;
    regions.reserve(layerCount);
    for (uint32_t i = 0; i < layerCount; ++i) {
        VkBufferImageCopy region{};
        region.bufferOffset      = i * bytesPerLayer;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, i, 1};
        region.imageOffset       = {0, 0, 0};
        region.imageExtent       = {width, height, 1};
        regions.push_back(region);
    }
    vkCmdCopyBufferToImage(cmd, src, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(regions.size()), regions.data());
}

uint32_t Image::calcMipLevels(uint32_t width, uint32_t height) {
    uint32_t levels = 1;
    while ((width > 1) || (height > 1)) {
        width  = (width  > 1) ? (width  >> 1) : 1;
        height = (height > 1) ? (height >> 1) : 1;
        ++levels;
    }
    return levels;
}

void Image::generateMipmaps(VkCommandBuffer cmd, VkImage image,
                              uint32_t width, uint32_t height, uint32_t mipLevels) {
    // Blit each mip level from the previous level.
    // The image must start in TRANSFER_DST_OPTIMAL (from the initial upload).
    // Each level is transitioned: DST -> SRC (for blitting), then -> SHADER_READ_ONLY at the end.
    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image                           = image;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.subresourceRange.levelCount     = 1;

    int32_t mipW = static_cast<int32_t>(width);
    int32_t mipH = static_cast<int32_t>(height);

    for (uint32_t i = 1; i < mipLevels; ++i) {
        // Transition level i-1 from DST to SRC so we can blit from it
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr, 0, nullptr, 1, &barrier);

        // Blit level i-1 -> level i (half the dimensions, linear filter)
        int32_t nextW = (mipW > 1) ? mipW / 2 : 1;
        int32_t nextH = (mipH > 1) ? mipH / 2 : 1;

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[0]  = {0, 0, 0};
        blit.srcOffsets[1]  = {mipW, mipH, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[0]  = {0, 0, 0};
        blit.dstOffsets[1]  = {nextW, nextH, 1};
        vkCmdBlitImage(cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // Transition level i-1 to its final SHADER_READ_ONLY layout
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr, 0, nullptr, 1, &barrier);

        mipW = nextW;
        mipH = nextH;
    }

    // Transition the final mip level (still in TRANSFER_DST) to SHADER_READ_ONLY
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace vkt
