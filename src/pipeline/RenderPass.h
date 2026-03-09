#pragma once

#include <volk.h>
#include <vector>

namespace vkt {

class Context;
class SwapChain;

// Owns the VkRenderPass and the per-swapchain-image VkFramebuffers.
// Supports two configurations selected at init() time:
//   * No MSAA: [color(1x), depth(1x)]
//   * MSAA:    [msaaColor(Nx), depth(Nx), resolve(1x)]
// Registers a rebuild callback on SwapChain so framebuffers are automatically
// recreated whenever the swapchain is rebuilt (e.g. on window resize).
class RenderPass {
public:
    RenderPass() = default;
    ~RenderPass();
    RenderPass(const RenderPass&)            = delete;
    RenderPass& operator=(const RenderPass&) = delete;

    void init(Context& ctx, SwapChain& swapchain);
    void rebuild(); // recreates framebuffers for the current swapchain extent
    void destroy();

    VkRenderPass handle() const { return m_renderPass; }

    // Records vkCmdBeginRenderPass and sets dynamic viewport + scissor.
    // clearColor applies to the color attachment (index 0).
    void begin(VkCommandBuffer cmd, uint32_t imageIndex,
               VkClearColorValue clearColor = {0.01f, 0.01f, 0.01f, 1.0f}) const;

    void end(VkCommandBuffer cmd) const;

private:
    void createFramebuffers();
    void destroyFramebuffers();

    Context*   m_ctx       = nullptr;
    SwapChain* m_swapchain = nullptr;

    VkRenderPass              m_renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;
};

} // namespace vkt
