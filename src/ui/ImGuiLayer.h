#pragma once

#include <volk.h>
#include <GLFW/glfw3.h>

namespace vkt {

class Context;
class RenderPass;

// Wraps ImGui initialization, per-frame recording, and teardown for the
// Vulkan + GLFW backend.  render() must be called inside an active render pass.
// The internal descriptor pool uses FREE_DESCRIPTOR_SET_BIT because ImGui
// manages its own font/image descriptors independently.
class ImGuiLayer {
public:
    ImGuiLayer() = default;
    ~ImGuiLayer();
    ImGuiLayer(const ImGuiLayer&)            = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    void init(GLFWwindow* window, Context& ctx, RenderPass& renderPass,
              uint32_t imageCount,
              VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT);
    void destroy();

    void newFrame();
    void render(VkCommandBuffer cmd);

    // Call after swapchain rebuild to update ImGui's internal image count
    void onSwapchainRebuilt(uint32_t imageCount);

    // Call when the window moves to a monitor with a different DPI scale.
    // scale is the content scale reported by glfwGetWindowContentScale().
    // No-ops if scale hasn't changed.
    void onDpiChange(float scale);

private:
    // Reset style to defaults and apply DPI-aware scaling.
    // Safe to call at any time; no GPU operations required (v1.92 dynamic fonts).
    void applyScale(float scale);

    VkDevice         m_device      = VK_NULL_HANDLE;
    VkDescriptorPool m_imguiPool   = VK_NULL_HANDLE;
    float            m_dpiScale    = 1.0f;
    bool             m_initialized = false;
};

} // namespace vkt
