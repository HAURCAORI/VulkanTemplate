#include "SwapChain.h"
#include "Context.h"

#include <stdexcept>
#include <array>

namespace vkt {

VkFormat SwapChain::findDepthFormat() const {
    const std::array<VkFormat, 3> candidates{
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };

    for (VkFormat format : candidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(m_ctx->physicalDevice(), format, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return format;
    }
    throw std::runtime_error("[SwapChain] No supported depth format found");
}

void SwapChain::init(Context& ctx, VmaAllocator allocator, uint32_t width, uint32_t height,
                     VkSampleCountFlagBits samples, VkPresentModeKHR presentMode) {
    m_ctx                  = &ctx;
    m_allocator            = allocator;
    m_msaaSamples          = samples;
    m_preferredPresentMode = presentMode;
    m_depthFormat          = findDepthFormat(); // query once; result can't change after init
    rebuild(width, height);
    m_initialized = true;
}

void SwapChain::rebuild(uint32_t width, uint32_t height) {
    // If rebuilding an existing swapchain, destroy transient resources first.
    // Image views must be destroyed before the swapchain images they reference
    // are retired (i.e. before set_old_swapchain hands them to the driver).
    if (m_swapchain.swapchain != VK_NULL_HANDLE) {
        // Destroy our image views before handing the old swapchain to the driver
        // via set_old_swapchain.  The driver retires the old images only after the
        // new swapchain is created, so the views must not outlive this point.
        destroyMsaaResources();
        destroyDepthResources();
        for (auto view : m_imageViews)
            vkDestroyImageView(m_ctx->device(), view, nullptr);
        m_imageViews.clear();
    }

    vkb::SwapchainBuilder builder{
        m_ctx->physicalDevice(),
        m_ctx->device(),
        m_ctx->surface(),
        m_ctx->graphicsQueueIndex(),
        m_ctx->presentQueueIndex()
    };

    auto result = builder
        // B8G8R8A8_SRGB is the most common sRGB format on desktop; vk-bootstrap
        // falls back to the first available format if unsupported.
        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        // Preferred mode set by caller; always fall back to FIFO (guaranteed by spec).
        .set_desired_present_mode(m_preferredPresentMode)
        .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .set_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        // Passing the old handle lets the driver reuse its memory during rebuild.
        .set_old_swapchain(m_swapchain.swapchain)
        .build();

    if (!result)
        throw std::runtime_error("[SwapChain] Build failed: " + result.error().message());

    // Destroy old vkb swapchain bookkeeping (actual VkSwapchainKHR passed to new one above)
    vkb::destroy_swapchain(m_swapchain);
    m_swapchain  = result.value();
    m_images     = m_swapchain.get_images().value();
    m_imageViews = m_swapchain.get_image_views().value();

    createDepthResources();
    if (isMsaaEnabled())
        createMsaaResources();

    // Notify RenderPass so it can rebuild framebuffers
    if (m_rebuildCallback)
        m_rebuildCallback();
}

void SwapChain::createDepthResources() {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = m_depthFormat;
    imageInfo.extent        = {m_swapchain.extent.width, m_swapchain.extent.height, 1};
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = m_msaaSamples; // depth matches MSAA sample count
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage         = VMA_MEMORY_USAGE_AUTO;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &m_depthImage, &m_depthAllocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("[SwapChain] Failed to create depth image");

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = m_depthImage;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = m_depthFormat;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(m_ctx->device(), &viewInfo, nullptr, &m_depthImageView) != VK_SUCCESS) {
        // Clean up the already-allocated image so the caller doesn't need to track
        // partial state (destroy() may not run if m_initialized is still false).
        vmaDestroyImage(m_allocator, m_depthImage, m_depthAllocation);
        m_depthImage      = VK_NULL_HANDLE;
        m_depthAllocation = VK_NULL_HANDLE;
        throw std::runtime_error("[SwapChain] Failed to create depth image view");
    }
}

void SwapChain::destroyDepthResources() {
    if (m_depthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_ctx->device(), m_depthImageView, nullptr);
        m_depthImageView = VK_NULL_HANDLE;
    }
    if (m_depthImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_depthImage, m_depthAllocation);
        m_depthImage      = VK_NULL_HANDLE;
        m_depthAllocation = VK_NULL_HANDLE;
    }
}

VkResult SwapChain::acquireNextImage(VkSemaphore imageAvailable, uint32_t& outImageIndex) const {
    return vkAcquireNextImageKHR(
        m_ctx->device(),
        m_swapchain.swapchain,
        UINT64_MAX,
        imageAvailable,
        VK_NULL_HANDLE,
        &outImageIndex
    );
}

void SwapChain::createMsaaResources() {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = m_swapchain.image_format;
    imageInfo.extent        = {m_swapchain.extent.width, m_swapchain.extent.height, 1};
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = m_msaaSamples;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage         = VMA_MEMORY_USAGE_AUTO;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateImage(m_allocator, &imageInfo, &allocInfo,
                       &m_msaaImage, &m_msaaAllocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("[SwapChain] Failed to create MSAA color image");

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = m_msaaImage;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = m_swapchain.image_format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(m_ctx->device(), &viewInfo, nullptr, &m_msaaImageView) != VK_SUCCESS) {
        vmaDestroyImage(m_allocator, m_msaaImage, m_msaaAllocation);
        m_msaaImage      = VK_NULL_HANDLE;
        m_msaaAllocation = VK_NULL_HANDLE;
        throw std::runtime_error("[SwapChain] Failed to create MSAA color image view");
    }
}

void SwapChain::destroyMsaaResources() {
    if (m_msaaImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_ctx->device(), m_msaaImageView, nullptr);
        m_msaaImageView = VK_NULL_HANDLE;
    }
    if (m_msaaImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_msaaImage, m_msaaAllocation);
        m_msaaImage      = VK_NULL_HANDLE;
        m_msaaAllocation = VK_NULL_HANDLE;
    }
}

void SwapChain::destroy() {
    if (!m_initialized) return;
    destroyMsaaResources();
    destroyDepthResources();
    for (auto view : m_imageViews)
        vkDestroyImageView(m_ctx->device(), view, nullptr);
    m_imageViews.clear();
    vkb::destroy_swapchain(m_swapchain);
    m_initialized = false;
}

SwapChain::~SwapChain() { destroy(); }

} // namespace vkt
