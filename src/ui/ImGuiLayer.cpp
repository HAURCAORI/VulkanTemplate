#include "ImGuiLayer.h"
#include "core/Context.h"
#include "pipeline/RenderPass.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <stdexcept>
#include <array>

namespace vkt {

void ImGuiLayer::init(GLFWwindow* window, Context& ctx, RenderPass& renderPass,
                      uint32_t imageCount,
                      VkSampleCountFlagBits msaaSamples) {
    m_device = ctx.device();

    // ImGui needs its own descriptor pool for its internal resources
    std::array<VkDescriptorPoolSize, 1> poolSizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
    }};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets       = 16;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_imguiPool) != VK_SUCCESS)
        throw std::runtime_error("[ImGuiLayer] Failed to create descriptor pool");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Query the content scale for the window's current monitor and apply it.
    // On Windows this reflects the per-monitor magnification (e.g. 2.5 at 250%).
    // applyScale() sets style.FontScaleDpi (v1.92 dynamic font scaling) and
    // ScaleAllSizes() so widget padding/borders match the display DPI.
    glfwGetWindowContentScale(window, &m_dpiScale, nullptr);
    applyScale(m_dpiScale);

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion     = VK_API_VERSION_1_2;
    initInfo.Instance       = ctx.instance();
    initInfo.PhysicalDevice = ctx.physicalDevice();
    initInfo.Device         = ctx.device();
    initInfo.QueueFamily    = ctx.graphicsQueueIndex();
    initInfo.Queue          = ctx.graphicsQueue();
    initInfo.DescriptorPool = m_imguiPool;
    // MinImageCount=2 is the lowest ImGui accepts; clamp from below in
    // onSwapchainRebuilt() if the swapchain ever returns fewer than 2 images.
    initInfo.MinImageCount  = 2;
    initInfo.ImageCount     = imageCount;
    // PipelineInfoMain was introduced in ImGui v1.92 to decouple pipeline
    // creation from the ImGui_ImplVulkan_InitInfo call.
    initInfo.PipelineInfoMain.RenderPass  = renderPass.handle();
    initInfo.PipelineInfoMain.MSAASamples = msaaSamples;

    ImGui_ImplVulkan_Init(&initInfo);
    // Font texture is uploaded automatically by the backend (no manual call needed)

    m_initialized = true;
}

void ImGuiLayer::newFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render(VkCommandBuffer cmd) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void ImGuiLayer::onSwapchainRebuilt(uint32_t imageCount) {
    ImGui_ImplVulkan_SetMinImageCount(imageCount < 2 ? 2 : imageCount);
}

void ImGuiLayer::applyScale(float scale) {
    // Reset to base style so ScaleAllSizes() always multiplies from the
    // unscaled defaults, not from a previously scaled state.
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle{};
    ImGui::StyleColorsDark();

    // ScaleAllSizes multiplies padding, borders, grab sizes, etc.
    style.ScaleAllSizes(scale);

    // FontScaleDpi is applied on top of each font's base size at draw time
    // (v1.92+ feature). No font atlas rebuild needed.
    style.FontScaleDpi = scale;
}

void ImGuiLayer::onDpiChange(float scale) {
    if (scale == m_dpiScale) return;
    m_dpiScale = scale;
    applyScale(scale);
}

void ImGuiLayer::destroy() {
    if (!m_initialized) return;
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (m_imguiPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_imguiPool, nullptr);
        m_imguiPool = VK_NULL_HANDLE;
    }
    m_initialized = false;
}

ImGuiLayer::~ImGuiLayer() { destroy(); }

} // namespace vkt
