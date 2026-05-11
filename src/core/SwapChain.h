#pragma once

#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <functional>

namespace vkt {

class Context;

// Owns the vkb swapchain, its per-image color views, the shared depth image,
// and (when MSAA is active) the multisampled color resolve image.
// rebuild() reuses the old VkSwapchainKHR handle via set_old_swapchain so the
// driver can recycle memory without a full teardown.
class SwapChain {
public:
    SwapChain() = default;
    ~SwapChain();
    SwapChain(const SwapChain&)            = delete;
    SwapChain& operator=(const SwapChain&) = delete;

    // samples: VK_SAMPLE_COUNT_1_BIT disables MSAA.
    // Application clamps the requested count to the device maximum before calling.
    // presentMode: VK_PRESENT_MODE_FIFO_KHR (vsync) or MAILBOX_KHR (uncapped low-latency).
    // Falls back to FIFO if the requested mode is unsupported.
    void init(Context& ctx, VmaAllocator allocator, uint32_t width, uint32_t height,
              VkSampleCountFlagBits samples    = VK_SAMPLE_COUNT_1_BIT,
              VkPresentModeKHR      presentMode = VK_PRESENT_MODE_MAILBOX_KHR);
    void rebuild(uint32_t width, uint32_t height);
    void destroy();

    // -- Swapchain properties --------------------------------------------------
    VkSwapchainKHR           handle()      const { return m_swapchain.swapchain; }
    VkFormat                 imageFormat() const { return m_swapchain.image_format; }
    VkFormat                 depthFormat() const { return m_depthFormat; }
    VkExtent2D               extent()      const { return m_swapchain.extent; }
    uint32_t                 imageCount()  const { return static_cast<uint32_t>(m_imageViews.size()); }

    const std::vector<VkImage>&     images()     const { return m_images; }
    const std::vector<VkImageView>& imageViews() const { return m_imageViews; }

    // Depth resources (single shared depth buffer)
    VkImage     depthImage()     const { return m_depthImage; }
    VkImageView depthImageView() const { return m_depthImageView; }

    // MSAA color resolve image (VK_NULL_HANDLE when MSAA is disabled)
    VkImage               msaaImage()     const { return m_msaaImage; }
    VkImageView           msaaImageView() const { return m_msaaImageView; }
    VkSampleCountFlagBits msaaSamples()   const { return m_msaaSamples; }
    bool                  isMsaaEnabled() const { return m_msaaSamples > VK_SAMPLE_COUNT_1_BIT; }

    // -- Frame acquisition -----------------------------------------------------
    // Blocks with no timeout (UINT64_MAX)  --  acceptable for a desktop render loop.
    // Returns VK_ERROR_OUT_OF_DATE_KHR or VK_SUBOPTIMAL_KHR on resize.
    VkResult acquireNextImage(VkSemaphore imageAvailable, uint32_t& outImageIndex) const;

    // -- Resize callback -------------------------------------------------------
    // RenderPass registers this so SwapChain::rebuild() triggers framebuffer
    // recreation automatically without the caller needing to coordinate them.
    void setRebuildCallback(std::function<void()> cb) { m_rebuildCallback = std::move(cb); }

private:
    VkFormat findDepthFormat() const;
    void createDepthResources();
    void destroyDepthResources();
    void createMsaaResources();
    void destroyMsaaResources();

    Context*     m_ctx       = nullptr;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    vkb::Swapchain            m_swapchain{};
    std::vector<VkImage>      m_images;
    std::vector<VkImageView>  m_imageViews;

    VkImage       m_depthImage      = VK_NULL_HANDLE;
    VkImageView   m_depthImageView  = VK_NULL_HANDLE;
    VmaAllocation m_depthAllocation = VK_NULL_HANDLE;
    VkFormat      m_depthFormat     = VK_FORMAT_D32_SFLOAT;

    // MSAA color attachment (only valid when m_msaaSamples > VK_SAMPLE_COUNT_1_BIT)
    VkSampleCountFlagBits m_msaaSamples    = VK_SAMPLE_COUNT_1_BIT;
    VkImage               m_msaaImage      = VK_NULL_HANDLE;
    VkImageView           m_msaaImageView  = VK_NULL_HANDLE;
    VmaAllocation         m_msaaAllocation = VK_NULL_HANDLE;

    std::function<void()> m_rebuildCallback;
    VkPresentModeKHR      m_preferredPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    bool                  m_initialized = false;
};

} // namespace vkt
