#include "Application.h"
#include "scene/Frustum.h"
#include "scene/SceneLoader.h"
#include "utils/Platform.h"

#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <array>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#if defined(_WIN32)
#  include <timeapi.h>  // timeBeginPeriod / timeEndPeriod (winmm)
#endif

namespace vkt {
namespace fs = std::filesystem;

namespace {
constexpr glm::vec3 kShadowLightDirFallback{-0.5f, -1.0f, -0.5f};

uint16_t floatToHalfBits(float v) {
    uint32_t f = 0;
    std::memcpy(&f, &v, sizeof(f));
    const uint32_t sign = (f >> 31u) & 0x1u;
    const uint32_t exp  = (f >> 23u) & 0xFFu;
    uint32_t mant       =  f         & 0x7FFFFFu;

    if (exp == 255u)
        return static_cast<uint16_t>((sign << 15u) | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp == 0u)
        return static_cast<uint16_t>(sign << 15u);

    const int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31)
        return static_cast<uint16_t>((sign << 15u) | 0x7C00u);
    if (e <= 0) {
        if (e < -10)
            return static_cast<uint16_t>(sign << 15u);
        mant = (mant | 0x800000u) >> static_cast<uint32_t>(1 - e);
        return static_cast<uint16_t>((sign << 15u) | (mant >> 13u));
    }
    return static_cast<uint16_t>((sign << 15u) | (static_cast<uint32_t>(e) << 10u) | (mant >> 13u));
}
} // namespace

// -- Lifecycle ------------------------------------------------------------------

void Application::init(const Config& cfg) {
    m_cfg = cfg;
    m_shadowReceiverBias = std::max(0.0f, cfg.shadowReceiverBias);
    m_shaderSpecs.lightCap = static_cast<int32_t>(
        std::clamp(cfg.shaderLightCap, 1u, static_cast<uint32_t>(MAX_LIGHTS)));
    createWindow(cfg);
    initVulkan(cfg);
    createFrameData();
    createRenderFinishedSemaphores();
    createGpuTimestamps();
    createDescriptors();
    createBindlessResources();
    createPipelineCache();
    // PostProcess creates the HDR render pass before createPipeline() needs it.
    createPostProcess(cfg);
    createPipeline();
    // Shadow pass init requires globalLayout (created in createDescriptors) and
    // pipelineCache (created in createPipelineCache). Init before loadScene so
    // the shadow image is ready for descriptor writes.
    {
        ShadowPass::Config shadowCfg{};
        shadowCfg.cascadeCount = cfg.shadowCascades;
        shadowCfg.resolution   = cfg.shadowResolution;
        shadowCfg.splitLambda  = cfg.shadowSplitLambda;
        m_shadowPass.init(m_ctx.device(), m_allocator.handle(), m_ctx.physicalDevice(),
                          m_globalLayout, m_pipelineCache,
                          resolveShaderPath("shadow.vert.spv"),
                          shadowCfg);
    }
    // Write shadow sampler into global descriptor sets (binding=2).
    // Must be done after ShadowPass::init() creates the image and sampler.
    updateShadowDescriptors();
    // CullPass init: needs instance buffer handles (created in createDescriptors)
    // and the compiled cull.comp SPIR-V.
    createCullPass();
    // Sky background pass: draws before geometry inside the HDR render pass.
    // Must be initialized before createIblResources() so that setCubemap()
    // finds m_skyPass already initialized and can update its descriptor.
    createSkyPass(cfg);
    // IBL placeholder textures: writes bindings 4, 5, 6 into global sets.
    // If iblPath is set, bakes the env cubemap and calls m_skyPass.setCubemap().
    createIblResources();
    loadScene();
    // Render graph: built after all passes and the scene are initialized so that
    // execute lambdas can reference fully constructed members.
    createRenderGraph();
    // Async streaming: init after m_defaultImage/Sampler (placeholders) exist.
    createStreamLoader();
    // Shader hot-reload watcher (Track F): start last so the exe dir and
    // output dir are fully established.  Non-fatal if source dir not found.
    createShaderWatcher();
    m_running = true;
}

void Application::createWindow(const Config& cfg) {
    if (!glfwInit())
        throw std::runtime_error("[App] glfwInit failed");

    m_mouseSensitivity = cfg.mouseSensitivity;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWmonitor* monitor = nullptr;
    int w = static_cast<int>(cfg.width);
    int h = static_cast<int>(cfg.height);
    if (cfg.fullscreen) {
        monitor = glfwGetPrimaryMonitor();
        if (!monitor)
            throw std::runtime_error("[App] Failed to get primary monitor for fullscreen");
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (!mode)
            throw std::runtime_error("[App] Failed to get monitor video mode for fullscreen");
        w = mode->width;
        h = mode->height;
        // Borderless fullscreen: match the desktop video mode so the driver
        // avoids a mode switch and alt-tab stays fast.
        glfwWindowHint(GLFW_RED_BITS,     mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS,   mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS,    mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    }

    m_window = glfwCreateWindow(w, h, cfg.title.c_str(), monitor, nullptr);
    if (!m_window)
        throw std::runtime_error("[App] Failed to create GLFW window");

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, onFramebufferResize);
    glfwSetWindowContentScaleCallback(m_window, onWindowScale);
    glfwSetCursorPosCallback(m_window, onMouseMove);
}

void Application::initVulkan(const Config& cfg) {
    // cfg.validation enables layers at runtime (requires Vulkan SDK).
    // VKT_VALIDATION compile flag also forces them on (e.g. CI / debug builds).
#if defined(VKT_VALIDATION)
    const bool validation = true;
#else
    const bool validation = cfg.validation;
#endif
    m_ctx.init(m_window, {cfg.title, validation});
    m_allocator.init(m_ctx);

    // Clamp MSAA to what the physical device actually supports
    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(m_ctx.physicalDevice(), &devProps);
    // Save for E1 GPU timestamp conversion: nanoseconds per GPU tick.
    m_timestampPeriod = devProps.limits.timestampPeriod;

    // H1: query heap count once for the Memory Budget Dashboard.
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(m_ctx.physicalDevice(), &memProps);
    m_heapCount = memProps.memoryHeapCount;
    VkSampleCountFlags supported = devProps.limits.framebufferColorSampleCounts
                                 & devProps.limits.framebufferDepthSampleCounts;
    VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_1_BIT;
    for (auto c : {cfg.preferredMsaa,
                   VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_2_BIT}) {
        if (supported & c) { msaa = c; break; }
    }
    std::printf("[App] MSAA: %dx\n", static_cast<int>(msaa));

    int w, h;
    glfwGetFramebufferSize(m_window, &w, &h);
    const VkPresentModeKHR presentMode = cfg.vsync ? VK_PRESENT_MODE_FIFO_KHR
                                                   : VK_PRESENT_MODE_MAILBOX_KHR;
    m_swapchain.init(m_ctx, m_allocator.handle(),
                     static_cast<uint32_t>(w), static_cast<uint32_t>(h), msaa, presentMode);
    m_renderPass.init(m_ctx, m_swapchain);
    m_imgui.init(m_window, m_ctx, m_renderPass, m_swapchain.imageCount(),
                 m_swapchain.msaaSamples());

    // Camera positioned to see the full multi-object scene
    m_camera.setPerspective(cfg.cameraFov,
        static_cast<float>(w) / static_cast<float>(h), cfg.cameraNear, cfg.cameraFar);
    m_camera.setPosition({1.0f, 2.5f, 6.0f});
    m_camera.setLookAt({1.0f, 0.5f, 0.0f});
}

void Application::createFrameData() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_ctx.graphicsQueueIndex();
    // RESET_COMMAND_BUFFER_BIT allows vkResetCommandBuffer per-frame without
    // resetting the whole pool.  Each frame slot gets its own pool so recording
    // on one slot never blocks another.
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < m_frames.size(); ++i) {
        auto& f = m_frames[i];

        if (vkCreateCommandPool(m_ctx.device(), &poolInfo, nullptr, &f.commandPool) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to create command pool");

        VkCommandBufferAllocateInfo cbInfo{};
        cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbInfo.commandPool        = f.commandPool;
        cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_ctx.device(), &cbInfo, &f.commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to allocate command buffer");

        if (vkCreateSemaphore(m_ctx.device(), &semInfo, nullptr, &f.imageAvailableSemaphore) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to create semaphores");

        if (vkCreateFence(m_ctx.device(), &fenceInfo, nullptr, &f.inFlightFence) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to create fence");

        // Name objects for RenderDoc / NSight visibility
        char name[64];
        std::snprintf(name, sizeof(name), "Frame[%zu] Command Buffer", i);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_COMMAND_BUFFER,
                             f.commandBuffer, name);
        std::snprintf(name, sizeof(name), "Frame[%zu] imageAvailable", i);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_SEMAPHORE,
                             f.imageAvailableSemaphore, name);
        std::snprintf(name, sizeof(name), "Frame[%zu] inFlightFence", i);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_FENCE,
                             f.inFlightFence, name);
    }
}

void Application::createGpuTimestamps() {
    // Init GPU timestamp query pools for E1 per-pass timing.
    // Failure is non-fatal: GpuTimestamps degrades gracefully when unsupported.
    m_gpuTimestamps.init(m_ctx.device(), m_ctx.physicalDevice(),
                         m_ctx.graphicsQueueIndex(), MAX_FRAMES_IN_FLIGHT);
}

void Application::createBindlessResources() {
    // G1: Detect descriptor indexing support (core in Vulkan 1.2).
    VkPhysicalDeviceDescriptorIndexingFeatures indexing{};
    indexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    VkPhysicalDeviceFeatures2 feat2{};
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.pNext = &indexing;
    vkGetPhysicalDeviceFeatures2(m_ctx.physicalDevice(), &feat2);

    m_bindlessSupported =
        indexing.descriptorBindingPartiallyBound               &&
        indexing.runtimeDescriptorArray                        &&
        indexing.shaderSampledImageArrayNonUniformIndexing     &&
        indexing.descriptorBindingVariableDescriptorCount      &&
        indexing.descriptorBindingSampledImageUpdateAfterBind;

    std::printf("[App] Bindless textures: %s\n",
                m_bindlessSupported ? "supported" : "not supported (per-bind fallback)");

    if (!m_bindlessSupported) {
        m_useBindless = false;
        return;
    }

    m_bindlessSet.init(m_ctx.device(), RenderLimits::kMaxBindlessTextures);
    m_useBindless = true;
}

void Application::createRenderFinishedSemaphores() {
    const uint32_t count = m_swapchain.imageCount();
    m_renderFinishedSemaphores.resize(count, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < count; ++i) {
        if (vkCreateSemaphore(m_ctx.device(), &semInfo, nullptr,
                              &m_renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to create renderFinished semaphore");
        char name[64];
        std::snprintf(name, sizeof(name), "Image[%u] renderFinished", i);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_SEMAPHORE,
                             m_renderFinishedSemaphores[i], name);
    }
}

void Application::destroyRenderFinishedSemaphores() {
    for (auto sem : m_renderFinishedSemaphores)
        if (sem != VK_NULL_HANDLE)
            vkDestroySemaphore(m_ctx.device(), sem, nullptr);
    m_renderFinishedSemaphores.clear();
}

void Application::createDescriptors() {
    // Set 0: global UBO (binding 0) + instance SSBO (binding 1) + shadow map (binding 2)
    //        + material SSBO (binding 3) + IBL irradiance (4) + IBL prefilter (5) + IBL LUT (6)
    //        -- one per frame in flight.
    m_globalLayout = DescriptorLayoutBuilder{}
        .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT)
        .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    RenderLimits::kMaxShadowCascades)
        .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
        .addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)  // IBL irradiance cubemap
        .addBinding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)  // IBL prefiltered env cubemap
        .addBinding(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)  // IBL BRDF LUT
        .build(m_ctx.device());

    // Set 1: per-material albedo (binding 0) + normal map (binding 1).
    m_materialLayout = DescriptorLayoutBuilder{}
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
        .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
        .build(m_ctx.device());

    // Pool capacity is runtime-configurable for scene scalability.
    // IBL adds 3 more samplers per global set (bindings 4,5,6); normal map adds 1 per material set.
    // Raise COMBINED_IMAGE_SAMPLER ratio from 6 to 9 to accommodate IBL bindings.
    const uint32_t maxSets = std::max(64u, m_cfg.maxDescriptorSets);
    m_descriptorPool.init(m_ctx.device(), maxSets, {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1.0f},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1.0f},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9.0f},
    });

    // Note: m_defaultMaterialSet is allocated in loadScene() after the image is created.

    // Material SSBO: shared across all frames (materials change infrequently).
    // Initialized before global set writes so binding=3 can be written immediately.
    m_materials.init(m_allocator.handle(), RenderLimits::kMaxMaterials);

    // Uniform buffers + instance SSBOs + descriptor sets  --  one per frame
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_uniformBuffers[i].createCpuVisible(
            m_allocator.handle(), sizeof(GlobalUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        m_instanceBuffers[i].init(m_allocator.handle(), RenderLimits::kMaxInstances);

        char name[64];
        std::snprintf(name, sizeof(name), "Global UBO [%u]", i);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_BUFFER,
                             m_uniformBuffers[i].handle(), name);
        std::snprintf(name, sizeof(name), "Instance SSBO [%u]", i);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_BUFFER,
                             m_instanceBuffers[i].handle(), name);

        const VkDeviceSize ssboSize =
            static_cast<VkDeviceSize>(RenderLimits::kMaxInstances) * sizeof(InstanceData);

        m_globalSets[i] = m_descriptorPool.allocate(m_globalLayout);
        DescriptorWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         m_uniformBuffers[i].handle(), 0, sizeof(GlobalUBO))
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         m_instanceBuffers[i].handle(), 0, ssboSize)
            .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         m_materials.buffer(), 0, m_materials.bufferSize())
            .update(m_ctx.device(), m_globalSets[i]);
    }
}

void Application::updateShadowDescriptors() {
    auto views = m_shadowPass.depthViews();
    if (views.empty()) return;
    while (views.size() < RenderLimits::kMaxShadowCascades)
        views.push_back(views.back());
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        DescriptorWriter{}
            .writeImages(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        views,
                        m_shadowPass.sampler(),
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            .update(m_ctx.device(), m_globalSets[i]);
    }
}

void Application::createPostProcess(const Config& cfg) {
    int w, h;
    glfwGetFramebufferSize(m_window, &w, &h);

    PostProcessPass::Config ppCfg;
    ppCfg.exposure    = cfg.exposure;
    ppCfg.gamma       = cfg.gamma;
    ppCfg.tonemapMode = cfg.tonemapMode;

    m_postProcess.init(
        m_ctx.device(),
        m_allocator.handle(),
        VkExtent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)},
        m_renderPass.handle(),       // blit pipeline targets the swapchain render pass
        m_swapchain.msaaSamples(),   // must match the swapchain render pass MSAA
        m_pipelineCache,
        MAX_FRAMES_IN_FLIGHT,
        resolveShaderPath("fullscreen.vert.spv"),
        resolveShaderPath("post_basic.frag.spv"),
        ppCfg);
}

// -- Cull pass (Phase 6) -------------------------------------------------------

void Application::createCullPass() {
    // Per-frame indirect draw buffers (one VkDrawIndexedIndirectCommand per batch).
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_indirectBuffers[i].init(m_allocator.handle(), RenderLimits::kMaxBatches);
    }

    // Compute pipeline for GPU frustum culling.
    m_cullPass.init(
        m_ctx.device(),
        m_allocator.handle(),
        resolveShaderPath("cull.comp.spv"),
        RenderLimits::kMaxInstances,
        MAX_FRAMES_IN_FLIGHT);

    // Bind per-frame instance SSBO (binding 1) and indirect buffer (binding 2)
    // into the compute descriptor sets.  The instance SSBO is the same buffer
    // used by the graphics pass at set=0, binding=1; compute writes it first,
    // then the geometry pass reads it after a compute->vertex barrier.
    const VkDeviceSize instSize =
        static_cast<VkDeviceSize>(RenderLimits::kMaxInstances) * sizeof(InstanceData);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        const VkDeviceSize indrSize =
            static_cast<VkDeviceSize>(RenderLimits::kMaxBatches) * sizeof(VkDrawIndexedIndirectCommand);
        m_cullPass.updateDescriptors(
            i,
            m_instanceBuffers[i].handle(), instSize,
            m_indirectBuffers[i].handle(), indrSize);
    }
}

VkCommandBuffer Application::beginImmediateCommands(VkCommandPool& pool) const {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_ctx.graphicsQueueIndex();
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (vkCreateCommandPool(m_ctx.device(), &poolInfo, nullptr, &pool) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to create transient command pool");

    VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool        = pool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_ctx.device(), &cbInfo, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(m_ctx.device(), pool, nullptr);
        pool = VK_NULL_HANDLE;
        throw std::runtime_error("[App] Failed to allocate transient command buffer");
    }

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
        vkDestroyCommandPool(m_ctx.device(), pool, nullptr);
        pool = VK_NULL_HANDLE;
        throw std::runtime_error("[App] Failed to begin transient command buffer");
    }
    return cmd;
}

void Application::endImmediateCommands(VkCommandPool pool, VkCommandBuffer cmd) const {
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(m_ctx.device(), pool, nullptr);
        throw std::runtime_error("[App] Failed to end transient command buffer");
    }

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    if (vkQueueSubmit(m_ctx.graphicsQueue(), 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkDestroyCommandPool(m_ctx.device(), pool, nullptr);
        throw std::runtime_error("[App] Failed to submit transient command buffer");
    }
    vkQueueWaitIdle(m_ctx.graphicsQueue());
    vkDestroyCommandPool(m_ctx.device(), pool, nullptr);
}

void Application::updateSolidCubemap(Image& image, const glm::vec4& color) {
    if (image.handle() == VK_NULL_HANDLE) return;

    const uint32_t width     = std::max(1u, image.width());
    const uint32_t height    = std::max(1u, image.height());
    const uint32_t mipLevels = std::max(1u, image.mipLevels());
    const uint32_t layers    = std::max(1u, image.arrayLayers());
    const bool isHalf        = image.format() == VK_FORMAT_R16G16B16A16_SFLOAT;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = beginImmediateCommands(pool);
    std::vector<Buffer> staging;
    staging.reserve(mipLevels);

    Image::transitionLayout(cmd, image.handle(),
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            mipLevels, VK_IMAGE_ASPECT_COLOR_BIT, layers);

    for (uint32_t mip = 0; mip < mipLevels; ++mip) {
        const uint32_t mipWidth  = std::max(1u, width >> mip);
        const uint32_t mipHeight = std::max(1u, height >> mip);
        const VkDeviceSize bytesPerPixel = isHalf ? 8 : 4;
        const VkDeviceSize bytesPerLayer =
            static_cast<VkDeviceSize>(mipWidth) * mipHeight * bytesPerPixel;
        const VkDeviceSize totalBytes = bytesPerLayer * layers;

        auto& stg = staging.emplace_back();
        stg.createCpuVisible(m_allocator.handle(), totalBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        if (isHalf) {
            std::vector<uint16_t> pixels(static_cast<size_t>(mipWidth) * mipHeight * layers * 4u);
            const std::array<uint16_t, 4> halfColor{
                floatToHalfBits(color.r),
                floatToHalfBits(color.g),
                floatToHalfBits(color.b),
                floatToHalfBits(color.a),
            };
            for (size_t i = 0; i < pixels.size(); i += 4) {
                pixels[i + 0] = halfColor[0];
                pixels[i + 1] = halfColor[1];
                pixels[i + 2] = halfColor[2];
                pixels[i + 3] = halfColor[3];
            }
            stg.uploadData(pixels.data(), totalBytes);
        } else {
            const std::array<uint8_t, 4> rgba8{
                static_cast<uint8_t>(std::clamp(color.r, 0.0f, 1.0f) * 255.0f),
                static_cast<uint8_t>(std::clamp(color.g, 0.0f, 1.0f) * 255.0f),
                static_cast<uint8_t>(std::clamp(color.b, 0.0f, 1.0f) * 255.0f),
                static_cast<uint8_t>(std::clamp(color.a, 0.0f, 1.0f) * 255.0f),
            };
            std::vector<uint8_t> pixels(static_cast<size_t>(totalBytes));
            for (size_t i = 0; i < pixels.size(); i += 4) {
                pixels[i + 0] = rgba8[0];
                pixels[i + 1] = rgba8[1];
                pixels[i + 2] = rgba8[2];
                pixels[i + 3] = rgba8[3];
            }
            stg.uploadData(pixels.data(), totalBytes);
        }

        std::vector<VkBufferImageCopy> regions;
        regions.reserve(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            VkBufferImageCopy region{};
            region.bufferOffset      = bytesPerLayer * layer;
            region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
            region.imageExtent       = {mipWidth, mipHeight, 1};
            regions.push_back(region);
        }
        vkCmdCopyBufferToImage(cmd, stg.handle(), image.handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<uint32_t>(regions.size()), regions.data());
    }

    Image::transitionLayout(cmd, image.handle(),
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            mipLevels, VK_IMAGE_ASPECT_COLOR_BIT, layers);
    endImmediateCommands(pool, cmd);
}

void Application::updateSolidImage2D(Image& image, const glm::vec4& color) {
    if (image.handle() == VK_NULL_HANDLE) return;

    const VkDeviceSize bytesPerPixel =
        image.format() == VK_FORMAT_R16G16_SFLOAT ? 4 : 4;
    const VkDeviceSize totalBytes =
        static_cast<VkDeviceSize>(std::max(1u, image.width())) * std::max(1u, image.height()) * bytesPerPixel;

    Buffer staging;
    staging.createCpuVisible(m_allocator.handle(), totalBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    if (image.format() == VK_FORMAT_R16G16_SFLOAT) {
        std::vector<uint16_t> pixels(static_cast<size_t>(image.width()) * image.height() * 2u);
        const uint16_t x = floatToHalfBits(color.r);
        const uint16_t y = floatToHalfBits(color.g);
        for (size_t i = 0; i < pixels.size(); i += 2) {
            pixels[i + 0] = x;
            pixels[i + 1] = y;
        }
        staging.uploadData(pixels.data(), totalBytes);
    } else {
        std::vector<uint8_t> pixels(static_cast<size_t>(totalBytes));
        const uint8_t x = static_cast<uint8_t>(std::clamp(color.r, 0.0f, 1.0f) * 255.0f);
        const uint8_t y = static_cast<uint8_t>(std::clamp(color.g, 0.0f, 1.0f) * 255.0f);
        for (size_t i = 0; i < pixels.size(); i += 2) {
            pixels[i + 0] = x;
            pixels[i + 1] = y;
        }
        staging.uploadData(pixels.data(), totalBytes);
    }

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = beginImmediateCommands(pool);
    Image::transitionLayout(cmd, image.handle(),
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    Image::copyFromBuffer(cmd, staging.handle(), image.handle(),
                          std::max(1u, image.width()), std::max(1u, image.height()));
    Image::transitionLayout(cmd, image.handle(),
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    endImmediateCommands(pool, cmd);
    staging.destroy();
}

// -- IBL resources (placeholder textures) -------------------------------------

void Application::createIblResources() {
    // Samplers are always created; they are shared by both the baked and placeholder paths.
    m_iblCubeSampler.create(m_ctx.device(), m_ctx.physicalDevice(), Sampler::clampConfig());
    m_iblLutSampler.create(m_ctx.device(), m_ctx.physicalDevice(), Sampler::clampConfig());

    if (!m_cfg.iblPath.empty()) {
        // Full IBL bake from equirectangular HDR panorama.
        IblBaker::Config bakerCfg;
        bakerCfg.prefilterMips = static_cast<uint32_t>(m_cfg.iblMaxPrefilterLod + 1);

        m_iblBaker.init(
            m_ctx.device(),
            m_ctx.physicalDevice(),
            m_allocator.handle(),
            m_ctx.graphicsQueue(),
            m_ctx.graphicsQueueIndex(),
            resolveShaderPath("equirect_to_cube.comp.spv"),
            resolveShaderPath("irradiance_conv.comp.spv"),
            resolveShaderPath("prefilter_env.comp.spv"),
            bakerCfg);

        auto result      = m_iblBaker.bake(m_cfg.iblPath);
        m_iblIrradiance  = std::move(result.irradiance);
        m_iblPrefilter   = std::move(result.prefilter);
        m_iblBrdfLut     = std::move(result.brdfLut);
        m_iblEnvCubemap  = std::move(result.envCubemap);
        m_iblBaker.destroy(); // pipelines no longer needed after baking

        m_cfg.iblEnabled = true; // auto-enable when a path is provided
        m_iblBakedMaxPrefilterLod = static_cast<int>(m_iblPrefilter.mipLevels()) - 1;

        // Give the sky pass the baked env cubemap so it renders the environment
        // instead of the 1x1 black placeholder. setCubemap() also switches the
        // mode to SkyMode::Skybox automatically.
        m_skyPass.setCubemap(m_iblEnvCubemap.view(), m_iblCubeSampler.handle());

        updateIblDescriptors();
        return;
    }

    // Placeholder path: 1x1 black cubemaps + flat BRDF LUT.
    // When iblEnabled is false these are bound but not meaningfully sampled
    // (the shader branches on ubo.iblParams.x).

    VkCommandBuffer uploadCmd = VK_NULL_HANDLE;
    VkCommandPool   tmpPool   = VK_NULL_HANDLE;
    VkQueue         queue     = VK_NULL_HANDLE;
    vkGetDeviceQueue(m_ctx.device(), m_ctx.graphicsQueueIndex(), 0, &queue);

    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_ctx.graphicsQueueIndex();
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(m_ctx.device(), &poolInfo, nullptr, &tmpPool) != VK_SUCCESS)
            throw std::runtime_error("[App] IBL: Failed to create temporary command pool");

        VkCommandBufferAllocateInfo cbInfo{};
        cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbInfo.commandPool        = tmpPool;
        cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_ctx.device(), &cbInfo, &uploadCmd) != VK_SUCCESS) {
            vkDestroyCommandPool(m_ctx.device(), tmpPool, nullptr);
            throw std::runtime_error("[App] IBL: Failed to allocate upload command buffer");
        }
    }

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(uploadCmd, &begin);

    // All staging buffers held alive until after queue submission.
    std::vector<Buffer> staging;

    // Helper: create a 1x1 black RGBA8 cubemap (6 faces, CUBE_COMPATIBLE).
    auto makePlaceholderCube = [&](Image& img) {
        img.create(m_allocator.handle(), m_ctx.device(), {
            1, 1, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, 1,
            VK_SAMPLE_COUNT_1_BIT,
            6,  // arrayLayers
            VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
        });

        constexpr uint32_t kFaces     = 6;
        constexpr uint32_t kBytesEach = 4;  // 1x1 RGBA8
        uint8_t pixels[kFaces * kBytesEach] = {};
        for (uint32_t i = 0; i < kFaces; ++i) pixels[i * 4 + 3] = 255; // alpha=255

        auto& stg = staging.emplace_back();
        stg.createCpuVisible(m_allocator.handle(), kFaces * kBytesEach,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        stg.uploadData(pixels, kFaces * kBytesEach);

        Image::transitionLayout(uploadCmd, img.handle(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, VK_IMAGE_ASPECT_COLOR_BIT, 6);
        Image::copyFromBufferLayers(uploadCmd, stg.handle(), img.handle(),
                                    1, 1, kFaces, kBytesEach);
        Image::transitionLayout(uploadCmd, img.handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1, VK_IMAGE_ASPECT_COLOR_BIT, 6);
    };

    // Irradiance cubemap placeholder (black = no IBL diffuse).
    makePlaceholderCube(m_iblIrradiance);

    // Prefiltered env cubemap placeholder (black = no IBL specular).
    makePlaceholderCube(m_iblPrefilter);

    // Environment cubemap placeholder used when a runtime probe refresh updates the sky-driven IBL.
    makePlaceholderCube(m_iblEnvCubemap);

    // BRDF LUT placeholder: R8G8_UNORM 64x64, value (128,128) = (0.5, 0.5).
    // This approximates a mid-range split-sum result for all roughness/NdotV values.
    {
        constexpr uint32_t kSize  = 64;
        constexpr uint32_t kBytes = kSize * kSize * 2;  // R8G8
        m_iblBrdfLut.create(m_allocator.handle(), m_ctx.device(), {
            kSize, kSize, VK_FORMAT_R8G8_UNORM,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, 1
        });

        std::vector<uint8_t> lutPixels(kBytes, 128);  // (0.5, 0.5) for all texels

        auto& stg = staging.emplace_back();
        stg.createCpuVisible(m_allocator.handle(), kBytes,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        stg.uploadData(lutPixels.data(), kBytes);

        Image::transitionLayout(uploadCmd, m_iblBrdfLut.handle(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        Image::copyFromBuffer(uploadCmd, stg.handle(), m_iblBrdfLut.handle(), kSize, kSize);
        Image::transitionLayout(uploadCmd, m_iblBrdfLut.handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    vkEndCommandBuffer(uploadCmd);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &uploadCmd;
    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkDestroyCommandPool(m_ctx.device(), tmpPool, nullptr);
    staging.clear();

    updateIblDescriptors();
    m_iblBakedMaxPrefilterLod = std::max(0, static_cast<int>(m_iblPrefilter.mipLevels()) - 1);
}

void Application::updateIblDescriptors() {
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        DescriptorWriter{}
            .writeImage(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        m_iblIrradiance.view(), m_iblCubeSampler.handle(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .writeImage(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        m_iblPrefilter.view(), m_iblCubeSampler.handle(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .writeImage(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        m_iblBrdfLut.view(), m_iblLutSampler.handle(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .update(m_ctx.device(), m_globalSets[i]);
    }
}

void Application::rebuildSkyIblProbe() {
    if (!m_cfg.iblPath.empty() && m_skyPass.mode() == SkyMode::Skybox) {
        m_skyPass.markCaptureComplete();
        m_lastSkyProbeUpdateTime = m_animTime;
        return;
    }

    glm::vec4 probeColor{0.0f, 0.0f, 0.0f, 1.0f};
    if (m_skyPass.mode() == SkyMode::Color || m_skyPass.mode() == SkyMode::SkySphere)
        probeColor = glm::vec4(m_skyPass.skyColor(), 1.0f);

    updateSolidCubemap(m_iblIrradiance, probeColor);
    updateSolidCubemap(m_iblPrefilter, probeColor);
    if (m_iblEnvCubemap.handle() != VK_NULL_HANDLE)
        updateSolidCubemap(m_iblEnvCubemap, probeColor);
    updateSolidImage2D(m_iblBrdfLut, {0.5f, 0.5f, 0.0f, 1.0f});
    updateIblDescriptors();

    m_lastSkyProbeUpdateTime = m_animTime;
    m_skyPass.markCaptureComplete();
}

// -- Sky pass ------------------------------------------------------------------

void Application::createSkyPass(const Config& cfg) {
    m_skyPass.init(
        m_ctx.device(),
        m_allocator.handle(),
        m_globalLayout,
        m_descriptorPool.handle(),
        m_pipelineCache,
        m_postProcess.hdrRenderPass(),
        VK_SAMPLE_COUNT_1_BIT,   // HDR offscreen pass is always 1x MSAA
        resolveShaderPath("skybox.vert.spv"),
        resolveShaderPath("skybox.frag.spv"),
        m_ctx.graphicsQueueIndex(),
        cfg.sky);
}

// -- Pipeline cache -------------------------------------------------------------

void Application::createPipelineCache() {
    VkPipelineCacheCreateInfo cacheInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};

    // Attempt to seed the cache with data saved from a previous run.
    // The spec requires drivers to silently discard incompatible cache data,
    // but some drivers on new hardware are buggy here.  Validate the header
    // ourselves and skip the data if the GPU UUID does not match.
    std::vector<uint8_t> cacheData;
    if (std::ifstream f("pipeline_cache.bin", std::ios::binary | std::ios::ate); f.good()) {
        const auto size = static_cast<size_t>(f.tellg());
        // The Vulkan pipeline cache header (VkPipelineCacheHeaderVersionOne) is
        // exactly 32 bytes: headerSize(4) + version(4) + vendorID(4) +
        // deviceID(4) + pipelineCacheUUID(16).
        constexpr size_t kHeaderSize = 32;
        if (size >= kHeaderSize) {
            f.seekg(0);
            cacheData.resize(size);
            f.read(reinterpret_cast<char*>(cacheData.data()),
                   static_cast<std::streamsize>(size));

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(m_ctx.physicalDevice(), &props);

            // Read the relevant fields from the on-disk header.
            uint32_t diskVendor = 0, diskDevice = 0;
            std::memcpy(&diskVendor, cacheData.data() + 8,  4);
            std::memcpy(&diskDevice, cacheData.data() + 12, 4);
            const uint8_t* diskUUID = cacheData.data() + 16;

            const bool uuidMatch =
                diskVendor == props.vendorID &&
                diskDevice == props.deviceID &&
                std::memcmp(diskUUID, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;

            if (uuidMatch) {
                cacheInfo.initialDataSize = cacheData.size();
                cacheInfo.pInitialData    = cacheData.data();
                std::printf("[App] Pipeline cache loaded (%zu bytes)\n", size);
            } else {
                cacheData.clear();
                std::printf("[App] Pipeline cache skipped (GPU mismatch -- will rebuild)\n");
            }
        }
    }

    if (vkCreatePipelineCache(m_ctx.device(), &cacheInfo, nullptr, &m_pipelineCache) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to create pipeline cache");
}

void Application::savePipelineCache() {
    if (m_pipelineCache == VK_NULL_HANDLE) return;

    size_t size = 0;
    vkGetPipelineCacheData(m_ctx.device(), m_pipelineCache, &size, nullptr);
    std::vector<uint8_t> data(size);
    vkGetPipelineCacheData(m_ctx.device(), m_pipelineCache, &size, data.data());

    if (std::ofstream f("pipeline_cache.bin", std::ios::binary); f.good()) {
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(size));
        std::printf("[App] Pipeline cache saved (%zu bytes)\n", size);
    }
}

// -- Pipeline -------------------------------------------------------------------

std::string Application::resolveShaderPath(const char* fileName) const {
    std::vector<fs::path> searchRoots;
    searchRoots.emplace_back(platform::getExeDir() / "shaders");
    const std::string shaderDir = platform::readEnvVar("VKT_SHADER_DIR");
    if (!shaderDir.empty())
        searchRoots.emplace_back(shaderDir);

    for (const auto& root : searchRoots) {
        const fs::path candidate = root / fileName;
        if (fs::exists(candidate))
            return candidate.string();
    }

    throw std::runtime_error(
        std::string("[App] Shader not found: ") + fileName +
        ". Set VKT_SHADER_DIR or run from a build directory with compiled shaders.");
}

void Application::createPipeline() {
    std::array<VkDescriptorSetLayout, 2> layouts{m_globalLayout, m_materialLayout};

    // No push constants: per-instance data (model, normalMatrix) is now in the
    // SSBO at set=0, binding=1, indexed by gl_InstanceIndex.
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = static_cast<uint32_t>(layouts.size());
    layoutInfo.pSetLayouts            = layouts.data();
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pPushConstantRanges    = nullptr;

    if (vkCreatePipelineLayout(m_ctx.device(), &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to create pipeline layout");

    Shader vert, frag;
    vert.load(m_ctx.device(), resolveShaderPath("default.vert.spv"));
    frag.load(m_ctx.device(), resolveShaderPath("default.frag.spv"));

    VkSpecializationMapEntry lightCapEntry{};
    lightCapEntry.constantID = 0;
    lightCapEntry.offset     = offsetof(ShaderSpecializationConstants, lightCap);
    lightCapEntry.size       = sizeof(ShaderSpecializationConstants::lightCap);

    VkSpecializationInfo fragSpecs{};
    fragSpecs.mapEntryCount = 1;
    fragSpecs.pMapEntries   = &lightCapEntry;
    fragSpecs.dataSize      = sizeof(ShaderSpecializationConstants);
    fragSpecs.pData         = &m_shaderSpecs;

    auto vertStage = vert.stageInfo(VK_SHADER_STAGE_VERTEX_BIT);
    auto fragStage = frag.stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT);
    fragStage.pSpecializationInfo = &fragSpecs;

    auto binding    = Vertex::bindingDesc();
    auto attributes = Vertex::attributeDescs();
    std::vector<VkVertexInputBindingDescription>   bindings{binding};
    std::vector<VkVertexInputAttributeDescription> attrs{attributes.begin(), attributes.end()};

    // Geometry renders into the HDR offscreen target (1x MSAA, R16G16B16A16_SFLOAT).
    // The tonemap blit later copies it into the swapchain with exposure + gamma.
    m_pipeline = PipelineBuilder{}
        .addShaderStage(vertStage)
        .addShaderStage(fragStage)
        .setVertexInput(bindings, attrs)
        .setDepthTest(true, true)
        .setCullMode(VK_CULL_MODE_BACK_BIT)
        .setMultisampling(VK_SAMPLE_COUNT_1_BIT)  // HDR offscreen is always 1x
        .build(m_ctx.device(), m_pipelineLayout, m_postProcess.hdrRenderPass(), 0, m_pipelineCache);

    debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_PIPELINE,
                         m_pipeline, "Main Pipeline");
    debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                         m_pipelineLayout, "Main Pipeline Layout");

    // G2: Create bindless pipeline variant (default_bindless.frag + bindless layout).
    // Layout: set=0 global, set=1 bindless array (replaces per-material set=1).
    if (m_useBindless) {
        std::array<VkDescriptorSetLayout, 2> bindlessLayouts{
            m_globalLayout, m_bindlessSet.layout()
        };
        VkPipelineLayoutCreateInfo bindlessLayoutInfo{};
        bindlessLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        bindlessLayoutInfo.setLayoutCount = static_cast<uint32_t>(bindlessLayouts.size());
        bindlessLayoutInfo.pSetLayouts    = bindlessLayouts.data();

        if (vkCreatePipelineLayout(m_ctx.device(), &bindlessLayoutInfo, nullptr,
                                   &m_bindlessPipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to create bindless pipeline layout");

        Shader bindlessFrag;
        bindlessFrag.load(m_ctx.device(), resolveShaderPath("default_bindless.frag.spv"));
        auto bindlessFragStage = bindlessFrag.stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT);
        bindlessFragStage.pSpecializationInfo = &fragSpecs;

        m_bindlessPipeline = PipelineBuilder{}
            .addShaderStage(vertStage)
            .addShaderStage(bindlessFragStage)
            .setVertexInput(bindings, attrs)
            .setDepthTest(true, true)
            .setCullMode(VK_CULL_MODE_BACK_BIT)
            .setMultisampling(VK_SAMPLE_COUNT_1_BIT)
            .build(m_ctx.device(), m_bindlessPipelineLayout,
                   m_postProcess.hdrRenderPass(), 0, m_pipelineCache);

        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_PIPELINE,
                             m_bindlessPipeline, "Main Pipeline (Bindless)");
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                             m_bindlessPipelineLayout, "Bindless Pipeline Layout");
    }
}

// -- Async streaming ----------------------------------------------------------

// -- Track F: Pipeline Hot Reload ---------------------------------------------

void Application::createShaderWatcher() {
    const std::string exeDir   = platform::getExeDir().string();
    const std::string sourceDir = ShaderWatcher::findSourceDir(exeDir);
    // Output SPV files live in <exe>/shaders/ (where resolveShaderPath searches).
    const std::string outputDir =
        (fs::path(platform::getExeDir()) / "shaders").string();
    const std::string glslcPath = ShaderWatcher::findGlslc();

    // Only the main geometry shaders are hot-reloadable via this watcher.
    // Shadow/post/cull pipelines are more complex to rebuild and are excluded.
    // G2: include default_bindless.frag so the bindless variant is also reloaded.
    m_shaderWatcher.init(glslcPath, sourceDir, outputDir,
                         {"default.vert", "default.frag", "default_bindless.frag"});
}

bool Application::tryHotReloadPipeline() {
    if (!m_shaderWatcher.consumeReady()) return false;

    // Build the new pipeline(s). Use the updated SPV files that the watcher compiled.
    VkPipeline newPipeline         = VK_NULL_HANDLE;
    VkPipeline newBindlessPipeline = VK_NULL_HANDLE;
    try {
        Shader vert, frag;
        vert.load(m_ctx.device(), resolveShaderPath("default.vert.spv"));
        frag.load(m_ctx.device(), resolveShaderPath("default.frag.spv"));

        VkSpecializationMapEntry lightCapEntry{};
        lightCapEntry.constantID = 0;
        lightCapEntry.offset     = offsetof(ShaderSpecializationConstants, lightCap);
        lightCapEntry.size       = sizeof(ShaderSpecializationConstants::lightCap);

        VkSpecializationInfo fragSpecs{};
        fragSpecs.mapEntryCount = 1;
        fragSpecs.pMapEntries   = &lightCapEntry;
        fragSpecs.dataSize      = sizeof(ShaderSpecializationConstants);
        fragSpecs.pData         = &m_shaderSpecs;

        auto vertStage = vert.stageInfo(VK_SHADER_STAGE_VERTEX_BIT);
        auto fragStage = frag.stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT);
        fragStage.pSpecializationInfo = &fragSpecs;

        auto binding    = Vertex::bindingDesc();
        auto attributes = Vertex::attributeDescs();
        std::vector<VkVertexInputBindingDescription>   bindings{binding};
        std::vector<VkVertexInputAttributeDescription> attrs{attributes.begin(), attributes.end()};

        newPipeline = PipelineBuilder{}
            .addShaderStage(vertStage)
            .addShaderStage(fragStage)
            .setVertexInput(bindings, attrs)
            .setDepthTest(true, true)
            .setCullMode(VK_CULL_MODE_BACK_BIT)
            .setMultisampling(VK_SAMPLE_COUNT_1_BIT)
            .build(m_ctx.device(), m_pipelineLayout,
                   m_postProcess.hdrRenderPass(), 0, m_pipelineCache);

        // G2: Also rebuild bindless pipeline if active.
        if (m_useBindless) {
            Shader bindlessFrag;
            bindlessFrag.load(m_ctx.device(),
                              resolveShaderPath("default_bindless.frag.spv"));
            auto bindlessFragStage = bindlessFrag.stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT);
            bindlessFragStage.pSpecializationInfo = &fragSpecs;

            newBindlessPipeline = PipelineBuilder{}
                .addShaderStage(vertStage)
                .addShaderStage(bindlessFragStage)
                .setVertexInput(bindings, attrs)
                .setDepthTest(true, true)
                .setCullMode(VK_CULL_MODE_BACK_BIT)
                .setMultisampling(VK_SAMPLE_COUNT_1_BIT)
                .build(m_ctx.device(), m_bindlessPipelineLayout,
                       m_postProcess.hdrRenderPass(), 0, m_pipelineCache);
        }
    } catch (const std::exception& e) {
        if (newPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(m_ctx.device(), newPipeline, nullptr);
        if (newBindlessPipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(m_ctx.device(), newBindlessPipeline, nullptr);
        std::fprintf(stderr, "[HotReload] Pipeline rebuild failed: %s\n", e.what());
        return false;
    }

    if (newPipeline == VK_NULL_HANDLE) return false;

    // F2: Queue old pipelines for deferred destruction after MAX_FRAMES_IN_FLIGHT frames.
    const uint64_t safeFrame = m_frameCount + static_cast<uint64_t>(MAX_FRAMES_IN_FLIGHT);
    m_deferredPipelineDestroys.push_back({m_pipeline, safeFrame});
    m_pipeline = newPipeline;
    debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_PIPELINE,
                         m_pipeline, "Main Pipeline (Hot Reloaded)");

    if (newBindlessPipeline != VK_NULL_HANDLE) {
        m_deferredPipelineDestroys.push_back({m_bindlessPipeline, safeFrame});
        m_bindlessPipeline = newBindlessPipeline;
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_PIPELINE,
                             m_bindlessPipeline, "Main Pipeline Bindless (Hot Reloaded)");
    }

    std::printf("[HotReload] Main pipeline(s) reloaded successfully.\n");
    return true;
}

void Application::createStreamLoader() {
    // Use the 1x1 white placeholder image so any in-flight request renders
    // a solid white texel instead of crashing from a null descriptor.
    StreamLoader::Config cfg;
    cfg.maxQueueDepth = 32;
    cfg.workerThreads = 2;
    m_streamLoader.init(
        m_ctx.device(), m_ctx.physicalDevice(), m_allocator.handle(),
        m_ctx.graphicsQueue(), m_ctx.graphicsQueueIndex(),
        m_defaultImage.view(), m_defaultSampler.handle(),
        cfg);
}

// -- Render graph -------------------------------------------------------------

void Application::createRenderGraph() {
    m_renderGraph.reset();

    // Register resources: initial state = HOST_WRITE (CPU fills them in prepareFrameData
    // before vkBeginCommandBuffer is called, so the barriers can be derived correctly).

    // Instance SSBO: CPU -> shadow vertex -> compute r/w -> geometry vertex.
    const auto rInstance = m_renderGraph.addBuffer({
        "InstanceSSBO",
        [this]{ return m_instanceBuffers[m_currentFrame].handle(); },
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT
    });
    // Indirect buffer: CPU zero-fills -> compute atomicAdd -> draw-indirect reads.
    const auto rIndirect = m_renderGraph.addBuffer({
        "IndirectBuffer",
        [this]{ return m_indirectBuffers[m_currentFrame].handle(); },
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT
    });
    // Cull object SSBO: CPU writes bounding info -> compute reads for culling.
    const auto rCullObj = m_renderGraph.addBuffer({
        "CullObjectSSBO",
        [this]{ return m_cullPass.objectBuffer(m_currentFrame); },
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT
    });

    std::vector<RenderGraph::ResourceId> rShadowMaps;
    rShadowMaps.reserve(m_shadowPass.cascadeCount());
    for (uint32_t c = 0; c < m_shadowPass.cascadeCount(); ++c) {
        rShadowMaps.push_back(m_renderGraph.addImage({
            c == 0 ? "ShadowMap[0]" : (c == 1 ? "ShadowMap[1]" : (c == 2 ? "ShadowMap[2]" : "ShadowMap[3]")),
            [this, c]{ return m_shadowPass.depthImage(c); },
            VK_IMAGE_ASPECT_DEPTH_BIT,
            1,
            1,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            0
        }));
    }

    const auto rHdrColor = m_renderGraph.addImage({
        "HdrColor",
        [this]{ return m_postProcess.hdrImage(m_currentFrame); },
        VK_IMAGE_ASPECT_COLOR_BIT,
        1,
        1,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0
    });
    const auto rHdrDepth = m_renderGraph.addImage({
        "HdrDepth",
        [this]{ return m_postProcess.hdrDepthImage(); },
        VK_IMAGE_ASPECT_DEPTH_BIT,
        1,
        1,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0
    });
    const auto rSwapchainColor = m_renderGraph.addImage({
        "SwapchainColor",
        [this]{ return m_swapchain.images()[m_currentImageIndex]; },
        VK_IMAGE_ASPECT_COLOR_BIT,
        1,
        1,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0
    });
    const auto rSwapchainDepth = m_renderGraph.addImage({
        "SwapchainDepth",
        [this]{ return m_swapchain.depthImage(); },
        VK_IMAGE_ASPECT_DEPTH_BIT,
        1,
        1,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0
    });
    RenderGraph::ResourceId rSwapchainMsaa = RenderGraph::kInvalid;
    if (m_swapchain.isMsaaEnabled()) {
        rSwapchainMsaa = m_renderGraph.addImage({
            "SwapchainMsaaColor",
            [this]{ return m_swapchain.msaaImage(); },
            VK_IMAGE_ASPECT_COLOR_BIT,
            1,
            1,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            0
        });
    }

    // Shadow: depth pre-pass; reads instance SSBO written by CPU.
    // Derived barrier: HOST_WRITE -> VERTEX_SHADER_READ on instanceSSBO.
    auto& shadowPass = m_renderGraph.addPass("Shadow", {0.8f, 0.4f, 0.1f, 1.0f},
        [this](VkCommandBuffer cmd) {
            if (m_shadowPass.enabled())
                m_shadowPass.render(cmd, m_globalSets[m_currentFrame],
                                    m_currentShadowBatches);
        })
        .readBuffer(rInstance, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    for (const auto shadowMap : rShadowMaps) {
        shadowPass.writeAttachmentImage(
            shadowMap,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }

    // GPU Cull: compute frustum culling.
    // Derived barrier: HOST_WRITE(cullObj,indirect) + VERTEX_SHADER_READ(instance)
    //   -> COMPUTE_SHADER_READ|WRITE on all three buffers.
    m_renderGraph.addPass("GPU Cull", {0.5f, 0.9f, 0.5f, 1.0f},
        [this](VkCommandBuffer cmd) {
            const Frustum frustum = Frustum::fromVP(m_viewProj);
            m_cullPass.dispatch(cmd, m_currentFrame,
                                frustum.planes.data(), m_currentObjectCount);
        })
        .readBuffer(rCullObj, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT)
        .readWriteBuffer(rInstance, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
        .readWriteBuffer(rIndirect, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // HDR Geometry: sky + main geometry via GPU-culled indirect draw.
    // Derived barrier: COMPUTE_SHADER_WRITE(instance,indirect)
    //   -> VERTEX_SHADER_READ(instance) + DRAW_INDIRECT_READ(indirect).
    // G3: selects between bindless path (single set=1 bind) and per-batch path.
    auto& hdrPass = m_renderGraph.addPass("HDR Geometry", {0.2f, 0.6f, 0.9f, 1.0f},
        [this](VkCommandBuffer cmd) {
            m_postProcess.beginHdr(cmd, m_currentFrame);
            m_skyPass.draw(cmd, m_globalSets[m_currentFrame]);
            if (m_useBindless) {
                // G2: bind bindless pipeline + global set + bindless texture array once.
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_bindlessPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_bindlessPipelineLayout, 0, 1,
                                        &m_globalSets[m_currentFrame], 0, nullptr);
                const VkDescriptorSet bindlessSet = m_bindlessSet.handle();
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_bindlessPipelineLayout, 1, 1,
                                        &bindlessSet, 0, nullptr);
                m_world.drawIndirect(cmd, m_bindlessPipelineLayout, m_currentCullBatches,
                                     m_indirectBuffers[m_currentFrame].handle(),
                                     true /*skipTextureBinds*/);
            } else {
                // G3: non-bindless fallback -- per-batch texture set binds.
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_pipelineLayout, 0, 1,
                                        &m_globalSets[m_currentFrame], 0, nullptr);
                m_world.drawIndirect(cmd, m_pipelineLayout, m_currentCullBatches,
                                     m_indirectBuffers[m_currentFrame].handle());
            }
            m_postProcess.endHdr(cmd);
        })
        .readBuffer(rInstance, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT)
        .readBuffer(rIndirect, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                    VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    for (const auto shadowMap : rShadowMaps) {
        hdrPass.readImage(
            shadowMap,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }
    hdrPass
        .writeAttachmentImage(
            rHdrColor,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .writeAttachmentImage(
            rHdrDepth,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Swapchain: tonemap blit + ImGui UI.
    auto& swapchainPass = m_renderGraph.addPass("Swapchain", {0.6f, 0.9f, 0.2f, 1.0f},
        [this](VkCommandBuffer cmd) {
            m_renderPass.begin(cmd, m_currentImageIndex);
            m_postProcess.blit(cmd, m_currentFrame);
            m_imgui.newFrame();
            renderUI();
            m_imgui.render(cmd);
            m_renderPass.end(cmd);
        })
        .readImage(rHdrColor, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .writeAttachmentImage(
            rSwapchainColor,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        .writeAttachmentImage(
            rSwapchainDepth,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    if (rSwapchainMsaa != RenderGraph::kInvalid) {
        swapchainPass.writeAttachmentImage(
            rSwapchainMsaa,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    m_renderGraph.compile();
}

// CPU-side frame setup: populate batch lists and flush per-frame GPU buffers.
// All HOST writes must complete before m_renderGraph.execute() records any GPU commands.
void Application::prepareFrameData() {
    // Shadow batches: all alive objects including off-screen casters so they can
    // cast shadows into the camera view. Compute cull overwrites instance SSBO after.
    InstanceBuffer& ib = m_instanceBuffers[m_currentFrame];
    m_currentShadowBatches = m_world.buildBatchesNoCull(
        ib.data(), ib.capacity(), m_camera.worldPosition());
    uint32_t shadowInstanceCount = 0;
    for (const auto& b : m_currentShadowBatches) shadowInstanceCount += b.count;
    ib.flush(shadowInstanceCount);

    // Flush material edits staged via the UI this frame (no-op if nothing changed).
    m_materials.flush();

    // Cull data: fill the CullObject SSBO (inside CullPass) and the indirect buffer.
    // CullPass::dispatch() will flush the CullObject SSBO before vkCmdDispatch.
    m_currentCullBatches.clear();
    m_currentObjectCount = m_world.buildCullData(
        m_cullPass.objectData(m_currentFrame),
        m_cullPass.objectCapacity(),
        RenderLimits::kMaxBatches,
        m_currentCullBatches,
        m_camera.worldPosition());

    IndirectBuffer& indr = m_indirectBuffers[m_currentFrame];
    for (const auto& b : m_currentCullBatches) {
        auto& entry         = indr.data()[b.batchId];
        entry.indexCount    = b.mesh->indexCount();
        entry.instanceCount = 0;  // GPU compute fills this via atomicAdd
        entry.firstIndex    = 0;  // each Mesh has a dedicated index buffer starting at 0
        entry.vertexOffset  = 0;  // no base-vertex offset
        entry.firstInstance = b.firstInstance;
    }
    indr.flush(static_cast<uint32_t>(m_currentCullBatches.size()));
}

// -- Scene ----------------------------------------------------------------------

void Application::loadScene() {
    // Borrow frame 0's pool for all one-shot uploads.  Safe because this is
    // called from init() before run() starts recording frames.
    VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool        = m_frames[0].commandPool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;

    VkCommandBuffer uploadCmd;
    if (vkAllocateCommandBuffers(m_ctx.device(), &cbInfo, &uploadCmd) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to allocate upload command buffer");

    try {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(uploadCmd, &begin) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to begin upload command buffer");

        // All staging buffers are kept alive until vkQueueWaitIdle below.
        std::vector<Buffer> staging;

        // -- Default 1x1 white texture ---------------------------------------------
        // Always registered as "default" so SceneLoader can fall back to it.
        {
            m_defaultImage.create(m_allocator.handle(), m_ctx.device(), {
                1, 1, VK_FORMAT_R8G8B8A8_SRGB,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, 1
            });
            constexpr uint8_t white[4] = {255, 255, 255, 255};
            auto& texStg = staging.emplace_back();
            texStg.createCpuVisible(m_allocator.handle(), 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            texStg.uploadData(white, 4);
            Image::transitionLayout(uploadCmd, m_defaultImage.handle(),
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);
            Image::copyFromBuffer(uploadCmd, texStg.handle(), m_defaultImage.handle(), 1, 1);
            Image::transitionLayout(uploadCmd, m_defaultImage.handle(),
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        }

        // -- Default 1x1 flat normal map ------------------------------------------
        // Tangent-space neutral up-vector: (128, 128, 255, 255) in RGBA8.
        // Written to binding 1 of every material set that lacks a real normal texture.
        {
            m_defaultNormalImage.create(m_allocator.handle(), m_ctx.device(), {
                1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, 1
            });
            constexpr uint8_t flatNormal[4] = {128, 128, 255, 255};
            auto& nStg = staging.emplace_back();
            nStg.createCpuVisible(m_allocator.handle(), 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            nStg.uploadData(flatNormal, 4);
            Image::transitionLayout(uploadCmd, m_defaultNormalImage.handle(),
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);
            Image::copyFromBuffer(uploadCmd, nStg.handle(), m_defaultNormalImage.handle(), 1, 1);
            Image::transitionLayout(uploadCmd, m_defaultNormalImage.handle(),
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        }

        m_defaultSampler.create(m_ctx.device(), m_ctx.physicalDevice(), Sampler::linearConfig());
        m_defaultNormalSampler.create(m_ctx.device(), m_ctx.physicalDevice(), Sampler::linearConfig());

        m_defaultMaterialSet = m_descriptorPool.allocate(m_materialLayout);
        DescriptorWriter{}
            .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        m_defaultImage.view(), m_defaultSampler.handle(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        m_defaultNormalImage.view(), m_defaultNormalSampler.handle(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .update(m_ctx.device(), m_defaultMaterialSet);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_IMAGE,
                             m_defaultImage.handle(), "Default White Texture");
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_IMAGE,
                             m_defaultNormalImage.handle(), "Default Flat Normal Map");

        // Register with the world so SceneLoader / user code can look it up by name.
        m_world.registerTexture("default", m_defaultMaterialSet);

        // G1: Register default textures into the bindless array.
        // Slot 0 = default white albedo, slot 1 = default flat normal.
        // These indices are passed to SceneLoader and used for materials with no texture.
        if (m_useBindless) {
            m_bindlessSet.registerTexture(m_ctx.device(),
                m_defaultImage.view(), m_defaultSampler.handle());           // idx 0
            m_bindlessSet.registerTexture(m_ctx.device(),
                m_defaultNormalImage.view(), m_defaultNormalSampler.handle()); // idx 1
        }

        // -- Scene objects ---------------------------------------------------------
        // -- Demo materials --------------------------------------------------------
        // Registered before scene objects so IDs can be passed to spawn().
        // Index 0 is always the default white material (registered in init()).
        const uint32_t matGround = m_materials.add("ground",
            GPUMaterial{ {0.45f, 0.45f, 0.45f, 1.0f}, 0.8f, 0.0f, 0.0f, 0 }); // dark grey
        // Sphere: blue metallic PBR material (UsePBR flag = bit2).
        const uint32_t matSphere = m_materials.add("sphere",
            GPUMaterial{ {0.2f, 0.6f, 1.0f, 1.0f}, 0.3f, 0.8f, 0.4f,
                         MaterialFlags::UsePBR });   // blue metallic + glow + PBR
        const uint32_t matCube   = m_materials.add("cube",
            GPUMaterial{ {1.0f, 0.35f, 0.1f, 1.0f}, 0.5f, 0.0f, 0.0f, 0 }); // orange

        if (!m_cfg.sceneFile.empty()) {
            // File-based path: SceneLoader creates meshes, loads textures, spawns objects.
            SceneLoadContext ctx{};
            ctx.allocator            = m_allocator.handle();
            ctx.device               = m_ctx.device();
            ctx.physDevice           = m_ctx.physicalDevice();
            ctx.uploadCmd            = uploadCmd;
            ctx.descriptorPool       = &m_descriptorPool;
            ctx.materialLayout       = m_materialLayout;
            ctx.materials            = &m_materials;
            ctx.defaultAlbedoView    = m_defaultImage.view();
            ctx.defaultAlbedoSampler = m_defaultSampler.handle();
            ctx.defaultNormalView          = m_defaultNormalImage.view();
            ctx.defaultNormalSampler       = m_defaultNormalSampler.handle();
            ctx.bindlessSet                = m_useBindless ? &m_bindlessSet : nullptr;
            ctx.defaultAlbedoBindlessIdx   = 0; // default white at slot 0
            ctx.defaultNormalBindlessIdx   = 1; // default normal at slot 1
            SceneLoader::load(ctx, m_world, m_cfg.sceneFile, staging, m_sceneTextures);
            // m_groundId / m_sphereId / m_cubeId stay INVALID_HANDLE; animation skipped.
        } else {
            // Demo fallback: procedurally create the built-in three-object scene.
            // makeQuad() is in the XY plane; rotate -90 deg around X to lay it flat.
            {
                auto [verts, indices] = Mesh::makeQuad();
                Buffer vs, is;
                Mesh m; m.upload(m_allocator.handle(), uploadCmd, verts, indices, vs, is);
                staging.push_back(std::move(vs));
                staging.push_back(std::move(is));
                const MeshHandle mh = m_world.registerMesh("ground", std::move(m));
                const TextureHandle th = m_world.findTexture("default");
                const glm::mat4 t = glm::scale(
                    glm::rotate(glm::mat4{1.0f}, glm::radians(-90.0f), {1.0f, 0.0f, 0.0f}),
                    {8.0f, 8.0f, 1.0f});
                m_groundId = m_world.spawn(mh, th, t, true, matGround);
            }
            {
                auto [verts, indices] = Mesh::makeUVSphere(32, 64);
                Buffer vs, is;
                Mesh m; m.upload(m_allocator.handle(), uploadCmd, verts, indices, vs, is);
                staging.push_back(std::move(vs));
                staging.push_back(std::move(is));
                const MeshHandle mh = m_world.registerMesh("sphere", std::move(m));
                const TextureHandle th = m_world.findTexture("default");
                m_sphereId = m_world.spawn(mh, th,
                    glm::translate(glm::mat4{1.0f}, {0.0f, 0.5f, 0.0f}), true, matSphere);
            }
            {
                auto [verts, indices] = Mesh::makeCube();
                Buffer vs, is;
                Mesh m; m.upload(m_allocator.handle(), uploadCmd, verts, indices, vs, is);
                staging.push_back(std::move(vs));
                staging.push_back(std::move(is));
                const MeshHandle mh = m_world.registerMesh("cube", std::move(m));
                const TextureHandle th = m_world.findTexture("default");
                m_cubeId = m_world.spawn(mh, th,
                    glm::translate(glm::mat4{1.0f}, {2.5f, 0.5f, 0.0f}), true, matCube);
            }
        }

        if (vkEndCommandBuffer(uploadCmd) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to end upload command buffer");

        // Use a fence so only this submission is waited on, not the whole queue.
        // Upload is submitted on the graphics queue.  No queue-family ownership
        // transfer is needed as long as resources are only read by the graphics
        // pipeline (same queue family).  If a future optimization moves uploads to
        // a dedicated transfer queue (transferQueueIndex() != graphicsQueueIndex()),
        // insert TRANSFER→GRAPHICS ownership transfer barriers (VK_SHARING_MODE_EXCLUSIVE)
        // before the vertex/fragment stages read these buffers and images.
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence uploadFence = VK_NULL_HANDLE;
        if (vkCreateFence(m_ctx.device(), &fenceInfo, nullptr, &uploadFence) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed to create upload fence");

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &uploadCmd;
        if (vkQueueSubmit(m_ctx.graphicsQueue(), 1, &submit, uploadFence) != VK_SUCCESS) {
            vkDestroyFence(m_ctx.device(), uploadFence, nullptr);
            throw std::runtime_error("[App] Failed to submit upload command buffer");
        }
        // Block until this upload submission finishes so staging buffers can be freed.
        if (vkWaitForFences(m_ctx.device(), 1, &uploadFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            vkDestroyFence(m_ctx.device(), uploadFence, nullptr);
            throw std::runtime_error("[App] Failed while waiting for upload fence");
        }
        vkDestroyFence(m_ctx.device(), uploadFence, nullptr);
    } catch (...) {
        vkFreeCommandBuffers(m_ctx.device(), m_frames[0].commandPool, 1, &uploadCmd);
        throw;
    }
    vkFreeCommandBuffers(m_ctx.device(), m_frames[0].commandPool, 1, &uploadCmd);

    // -- Lights ----------------------------------------------------------------
    m_lights.add(std::make_shared<DirectionalLight>(
        glm::vec3(-0.5f, -1.0f, -0.5f), // direction
        glm::vec3(1.0f, 0.95f, 0.85f),  // warm white
        1.2f                             // intensity
    ));
    m_lights.add(std::make_shared<PointLight>(
        glm::vec3(2.0f, 2.0f, 2.0f),    // position
        glm::vec3(0.3f, 0.5f, 1.0f),    // cool blue fill
        1.5f,                            // intensity
        8.0f                             // range
    ));
}

// -- Run loop -------------------------------------------------------------------

void Application::run() {
    using Clock = std::chrono::steady_clock;
    using Sec   = std::chrono::duration<double>;

#if defined(_WIN32)
    // Raise the Windows timer resolution to 1 ms so sleep_for is accurate
    // enough for the sleep+spin hybrid below. Default resolution is ~15 ms.
    timeBeginPeriod(1);
#endif

    // Seconds per frame for the software FPS cap (0 = no cap).
    const double minFrameTime = m_cfg.targetFps > 0
                                ? 1.0 / static_cast<double>(m_cfg.targetFps)
                                : 0.0;
    // Spin the last 2 ms after waking from sleep for sub-millisecond accuracy.
    // sleep_for with timeBeginPeriod(1) is accurate to ~1 ms, so 2 ms margin
    // comfortably covers the OS wakeup jitter without burning too much CPU.
    constexpr double kSpinMargin = 0.002;

    double lastTime  = glfwGetTime();
    double fpsAccum  = 0.0;
    int    fpsFrames = 0;

    while (!glfwWindowShouldClose(m_window) && m_running) {
        const auto frameStart = Clock::now();

        // When minimized the framebuffer is 0×0 and drawFrame() is a no-op.
        // Without throttling the loop spins at ~500k FPS burning CPU for nothing.
        {
            int fbW = 0, fbH = 0;
            glfwGetFramebufferSize(m_window, &fbW, &fbH);
            if (fbW == 0 || fbH == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
                glfwPollEvents();
                lastTime = glfwGetTime(); // reset so dt doesn't spike on restore
                continue;
            }
        }

        const double now      = glfwGetTime();
        const double doubleDt = now - lastTime;
        const float  dt       = static_cast<float>(doubleDt);
        lastTime = now;

        // Rolling FPS counter in the window title (updated every 0.5 s)
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5) {
            char title[128];
            std::snprintf(title, sizeof(title), "Vulkan Template - %.1f FPS",
                          static_cast<double>(fpsFrames) / fpsAccum);
            glfwSetWindowTitle(m_window, title);
            fpsAccum  = 0.0;
            fpsFrames = 0;
        }

        glfwPollEvents();
        processInput(dt);
        m_animTime += dt;
        onBeforeDraw(doubleDt);
        drawFrame();
        if (m_deviceLost) break;

        // Software FPS cap: sleep+spin hybrid.
        // Sleep covers most of the remaining budget (OS-scheduled, low CPU),
        // then spin the last kSpinMargin seconds for precise wakeup timing.
        // If the frame already exceeded the budget, both paths are skipped.
        if (minFrameTime > 0.0) {
            const double remaining = minFrameTime - Sec(Clock::now() - frameStart).count();
            if (remaining > kSpinMargin)
                std::this_thread::sleep_for(Sec(remaining - kSpinMargin));
            while (Sec(Clock::now() - frameStart).count() < minFrameTime)
                std::this_thread::yield();
        }
    }
#if defined(_WIN32)
    timeEndPeriod(1);
#endif
    if (!m_deviceLost) {
        if (vkDeviceWaitIdle(m_ctx.device()) != VK_SUCCESS)
            throw std::runtime_error("[App] Failed while waiting for device idle at end of run");
    }
}

void Application::drawFrame() {
    // Skip rendering while minimized (0x0 framebuffer).
    // This also prevents handleResize() from being called with a zero extent.
    {
        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(m_window, &fbW, &fbH);
        if (fbW == 0 || fbH == 0) return;
    }

    // Rebuild the swapchain before acquiring if a resize was requested by callbacks.
    if (m_framebufferResized) {
        handleResize();
    }

    FrameData& f = m_frames[m_currentFrame];

    {
        const VkResult r = vkWaitForFences(m_ctx.device(), 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);
        if (r == VK_ERROR_DEVICE_LOST) { reportDeviceLost(r, "vkWaitForFences"); return; }
        if (r != VK_SUCCESS) throw std::runtime_error("[App] Failed while waiting for in-flight fence");
    }

    // E1: Retrieve GPU timestamps for this frame slot now that its GPU work is done.
    // Skip the first MAX_FRAMES_IN_FLIGHT frames (query pools not yet written).
    if (m_gpuTimestamps.supported() && m_frameCount >= static_cast<uint64_t>(MAX_FRAMES_IN_FLIGHT))
        m_gpuTimestamps.retrieve(m_ctx.device(), m_currentFrame, m_timestampPeriod);

    // F2: Destroy pipelines that are no longer in flight.
    for (auto it = m_deferredPipelineDestroys.begin(); it != m_deferredPipelineDestroys.end(); ) {
        if (m_frameCount >= it->destroyAfterFrame) {
            vkDestroyPipeline(m_ctx.device(), it->pipeline, nullptr);
            it = m_deferredPipelineDestroys.erase(it);
        } else {
            ++it;
        }
    }

    // F1/F2: Check if ShaderWatcher compiled new SPIR-V; rebuild pipeline if so.
    tryHotReloadPipeline();

    uint32_t imageIndex;
    VkResult acqResult = m_swapchain.acquireNextImage(f.imageAvailableSemaphore, imageIndex);
    if (acqResult == VK_ERROR_DEVICE_LOST) { reportDeviceLost(acqResult, "vkAcquireNextImageKHR"); return; }
    if (acqResult == VK_ERROR_OUT_OF_DATE_KHR) {
        handleResize();
        return;
    }
    if (acqResult == VK_SUBOPTIMAL_KHR) {
        m_framebufferResized = true;
    } else if (acqResult != VK_SUCCESS) {
        throw std::runtime_error("[App] Failed to acquire swapchain image");
    }

    if (vkResetFences(m_ctx.device(), 1, &f.inFlightFence) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to reset in-flight fence");

    // Async streaming: promote decoded results to GPU uploads and retire fences.
    // Must run before recording commands so newly Ready textures can be used.
    m_streamLoader.tick();

    // -- Update uniform buffer --------------------------------------------------
    // pack() fills the light array and returns the count atomically (one lock).
    GlobalUBO ubo{};
    ubo.view         = m_camera.viewMatrix();
    ubo.proj         = m_camera.projectionMatrix();
    ubo.ambientColor = glm::vec4(m_cfg.ambientColor, 1.0f);
    // Floating origin: the camera is at (0,0,0) in camera-relative rendering space.
    // All object positions are expressed relative to the camera before upload,
    // so fragment shader lighting calculations see the camera at the origin.
    ubo.cameraPos = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    auto packedLights = m_lights.pack(&ubo.lightCount);
    std::copy(packedLights.begin(), packedLights.end(), ubo.lights);
    // Adjust point and spot light positions to camera-relative space.
    // Directional lights (type 0) use a direction vector, not a position.
    {
        const glm::dvec3 camWorld = m_camera.worldPosition();
        for (int32_t i = 0; i < ubo.lightCount; ++i) {
            const int type = static_cast<int>(ubo.lights[i].position.w);
            if (type == 1 || type == 2) { // point or spot
                const glm::dvec3 lightWorld = glm::dvec3(ubo.lights[i].position);
                const glm::vec3  lightRel   = glm::vec3(lightWorld - camWorld);
                ubo.lights[i].position = glm::vec4(lightRel, ubo.lights[i].position.w);
            }
        }
    }

    // Select the directional light that drives shadow cascade projection.
    // If m_shadowLightIndex is pinned to a valid directional light, use it.
    // Otherwise (auto mode, index -1), pick the directional with highest intensity
    // so the dominant light's shadow is always visible without manual setup.
    auto isValidDir = [&](int32_t i) {
        if (i < 0 || i >= ubo.lightCount) return false;
        if (static_cast<int>(ubo.lights[i].position.w) != 0) return false;
        const glm::vec3 d = glm::vec3(ubo.lights[i].direction);
        return glm::dot(d, d) > 1e-8f;
    };
    glm::vec3 shadowLightDir = glm::normalize(kShadowLightDirFallback);
    if (isValidDir(m_shadowLightIndex)) {
        shadowLightDir = glm::normalize(glm::vec3(ubo.lights[m_shadowLightIndex].direction));
    } else {
        float bestIntensity = -1.0f;
        for (int32_t i = 0; i < ubo.lightCount; ++i) {
            if (!isValidDir(i)) continue;
            const float intensity = ubo.lights[i].color.a;
            if (intensity > bestIntensity) {
                bestIntensity  = intensity;
                shadowLightDir = glm::normalize(glm::vec3(ubo.lights[i].direction));
            }
        }
    }
    glm::mat4 shadowCameraProj = m_camera.projectionMatrix();
    float shadowNear = m_camera.nearPlane();
    float shadowFar  = m_camera.farPlane();
    if (m_camera.projectionMode() == Camera::ProjectionMode::Orthographic) {
        // CSM fitting expects forward-depth ranges; orthographic negative-near
        // includes behind-camera space and causes incorrect cascade assignment.
        shadowNear = std::max(0.01f, shadowNear);
        shadowFar  = std::max(shadowNear + 0.1f, shadowFar);
        const VkExtent2D extent = m_swapchain.extent();
        const float aspect = (extent.height > 0)
            ? static_cast<float>(extent.width) / static_cast<float>(extent.height)
            : 1.0f;
        const float hh = m_camera.orthoSize();
        const float hw = hh * aspect;
        shadowCameraProj = glm::ortho(-hw, hw, -hh, hh, shadowNear, shadowFar);
        shadowCameraProj[1][1] *= -1.0f;
    }
    m_shadowPass.updateCascades(
        shadowLightDir,
        m_camera.viewMatrix(),
        shadowCameraProj,
        shadowNear,
        shadowFar);
    const uint32_t cascadeCount = m_shadowPass.cascadeCount();
    const auto& cascadeMats = m_shadowPass.cascadeMatrices();
    const auto& cascadeSplits = m_shadowPass.cascadeSplits();
    for (uint32_t c = 0; c < RenderLimits::kMaxShadowCascades; ++c) {
        ubo.lightSpaceMatrices[c] = cascadeMats[c];
        ubo.cascadeSplits[c]      = cascadeSplits[c];
    }
    ubo.shadowMeta.x   = static_cast<int32_t>(cascadeCount);
    ubo.shadowMeta.y   = m_shadowPass.enabled() ? 1 : 0;
    ubo.shadowParams.x = m_shadowReceiverBias;
    ubo.iblParams.x    = m_cfg.iblEnabled ? 1 : 0;
    ubo.iblParams.y    = m_iblBakedMaxPrefilterLod;
    ubo.iblParams.z    = m_debugVizMode;  // E3: debug visualization overlay mode
    m_uniformBuffers[m_currentFrame].uploadData(&ubo, sizeof(ubo));
    m_viewProj = ubo.proj * ubo.view;

    // -- Animate demo-scene objects (skipped when a sceneFile is loaded) -------
    // Sphere: rotates around Y at 45 deg/s
    if (m_world.alive(m_sphereId)) {
        const float angle = glm::radians(m_animTime * 45.0f);
        m_world.setTransform(m_sphereId,
            glm::translate(glm::mat4{1.0f}, {0.0f, 0.5f, 0.0f}) *
            glm::rotate(glm::mat4{1.0f}, angle, {0.0f, 1.0f, 0.0f}));
    }
    // Cube: counter-rotates around Y at 30 deg/s
    if (m_world.alive(m_cubeId)) {
        const float angle = glm::radians(-m_animTime * 30.0f);
        m_world.setTransform(m_cubeId,
            glm::translate(glm::mat4{1.0f}, {2.5f, 0.5f, 0.0f}) *
            glm::rotate(glm::mat4{1.0f}, angle, {0.0f, 1.0f, 0.0f}));
    }

    // -- Record + submit --------------------------------------------------------
    if (vkResetCommandBuffer(f.commandBuffer, 0) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to reset command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(f.commandBuffer, &begin) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to begin command buffer");
    recordCommands(f.commandBuffer, imageIndex);
    if (vkEndCommandBuffer(f.commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed to end command buffer");

    VkSemaphore renderFinished = m_renderFinishedSemaphores[imageIndex];

    // Wait at COLOR_ATTACHMENT_OUTPUT so the pipeline can run vertex work before
    // the acquired image is actually needed for writing.
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &f.imageAvailableSemaphore;
    submit.pWaitDstStageMask    = &waitStage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &f.commandBuffer;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &renderFinished;

    {
        const VkResult r = vkQueueSubmit(m_ctx.graphicsQueue(), 1, &submit, f.inFlightFence);
        if (r == VK_ERROR_DEVICE_LOST) { reportDeviceLost(r, "vkQueueSubmit"); return; }
        if (r != VK_SUCCESS) throw std::runtime_error("[App] Queue submit failed");
    }

    VkSwapchainKHR sc = m_swapchain.handle();
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &renderFinished;
    present.swapchainCount     = 1;
    present.pSwapchains        = &sc;
    present.pImageIndices      = &imageIndex;

    VkResult presResult = vkQueuePresentKHR(m_ctx.presentQueue(), &present);
    if (presResult == VK_ERROR_DEVICE_LOST) { reportDeviceLost(presResult, "vkQueuePresentKHR"); return; }
    if (presResult == VK_ERROR_OUT_OF_DATE_KHR || presResult == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
        handleResize();
    } else if (presResult != VK_SUCCESS) {
        throw std::runtime_error("[App] Failed to present swapchain image");
    }

    if (m_skyPass.isCaptureStale()) {
        if ((m_animTime - m_lastSkyProbeUpdateTime) >= m_skyProbeUpdateInterval)
            rebuildSkyIblProbe();
    }

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    ++m_frameCount; // E1: used to skip timestamp retrieve for uninitialized slots
}

void Application::recordCommands(VkCommandBuffer cmd, uint32_t imageIndex) {
    m_currentImageIndex = imageIndex;

    // E4: Optional frame capture marker for RenderDoc one-click capture.
    // Wraps the entire frame in a named debug label (visible as a group in RenderDoc).
    const bool doCapture = m_captureNextFrame;
    m_captureNextFrame = false;
    if (doCapture)
        debug::beginLabel(cmd, "CAPTURE_FRAME", {1.0f, 0.5f, 0.0f, 1.0f});

    // E1: Reset timestamp queries for this frame slot (must happen before execute()).
    m_gpuTimestamps.reset(cmd, m_currentFrame);

    prepareFrameData(); // CPU: build batch lists and flush per-frame GPU buffers

    // GPU: emit barriers, debug labels, timestamps, invoke pass callbacks.
    m_renderGraph.execute(cmd, m_currentFrame, &m_gpuTimestamps);

    if (doCapture)
        debug::endLabel(cmd);
}

void Application::renderUI() {
    ImGui::Begin("Scene");
    ImGui::Text("GPU: %s", m_ctx.deviceName());

    ImGui::Separator();
    ImGui::Text("Camera");
    const auto pos = m_camera.worldPosition();
    ImGui::Text("  Pos: %.6g, %.6g, %.6g", pos.x, pos.y, pos.z);
    ImGui::Text("  RMB + WASD/QE to fly, Arrows to rotate");

    // Projection mode toggle
    {
        using PM = Camera::ProjectionMode;
        const bool isPerspective = m_camera.projectionMode() == PM::Perspective;
        if (ImGui::RadioButton("Perspective",  isPerspective))  m_camera.setProjectionMode(PM::Perspective);
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic", !isPerspective)) m_camera.setProjectionMode(PM::Orthographic);
        ImGui::SameLine();
        ImGui::TextDisabled("(O)");

        if (isPerspective) {
            float fov = m_camera.fov();
            if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f, "%.0f deg"))
                m_camera.setFov(fov);
        } else {
            float orthoSize = m_camera.orthoSize();
            if (ImGui::SliderFloat("Ortho size", &orthoSize, 0.5f, 50.0f, "%.1f"))
                m_camera.setOrthoSize(orthoSize);
        }

        // Clip planes -- shared between both modes; each mode keeps its own defaults.
        float nearV = m_camera.nearPlane();
        float farV  = m_camera.farPlane();
        const float nearMin = 0.001f;
        const float nearMax = isPerspective ? 100.0f : 10000.0f;
        bool changed = ImGui::DragFloat("Near", &nearV, isPerspective ? 0.01f : 10.0f,
                                        nearMin, nearMax, "%.3f");
        changed |= ImGui::DragFloat("Far",  &farV,  10.0f, 1.0f, 100000.0f, "%.0f");
        if (changed)
            m_camera.setNearFar(nearV, farV);
    }

    ImGui::Separator();
    ImGui::Text("Scene  (%zu objects, %zu meshes)",
                m_world.objectCount(), m_world.meshCount());

    ImGui::Separator();
    ImGui::Text("Post-Processing");
    {
        auto& pp = m_postProcess.config();
        ImGui::SliderFloat("Exposure", &pp.exposure, 0.1f, 10.0f, "%.2f");
        ImGui::SliderFloat("Gamma",    &pp.gamma,    1.0f,  3.0f, "%.2f");
        const char* tonemapNames[] = {"Reinhard", "ACES", "Clamp"};
        ImGui::Combo("Tonemap", &pp.tonemapMode, tonemapNames, 3);
    }

    ImGui::Separator();
    ImGui::Text("Materials (%u / %u)", m_materials.count(), m_materials.capacity());
    {
        const uint32_t matCount = m_materials.count();
        for (uint32_t i = 0; i < matCount; ++i) {
            GPUMaterial& mat = m_materials.get(i);
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::TreeNode("mat", "[%u]", i)) {
                bool edited = false;
                edited |= ImGui::ColorEdit4("Base Color", &mat.baseColor.r);
                edited |= ImGui::SliderFloat("Roughness", &mat.roughness, 0.0f, 1.0f);
                edited |= ImGui::SliderFloat("Metallic",  &mat.metallic,  0.0f, 1.0f);
                edited |= ImGui::SliderFloat("Emissive",  &mat.emissive,  0.0f, 8.0f);
                bool unlit = (mat.flags & MaterialFlags::Unlit) != 0;
                if (ImGui::Checkbox("Unlit", &unlit)) {
                    mat.flags = unlit ? (mat.flags | MaterialFlags::Unlit)
                                      : (mat.flags & ~MaterialFlags::Unlit);
                    edited = true;
                }
                if (edited)
                    m_materials.update(i, mat);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    ImGui::Separator();
    ImGui::Text("IBL");
    {
        bool iblOn = m_cfg.iblEnabled;
        if (ImGui::Checkbox("IBL Enabled", &iblOn))
            m_cfg.iblEnabled = iblOn;
        ImGui::Text("Baked Max Prefilter LoD: %d", m_iblBakedMaxPrefilterLod);
        if (!m_cfg.iblPath.empty())
            ImGui::TextDisabled("IBL probe LoD is fixed by the baked environment map.");
    }

    ImGui::Separator();
    ImGui::Text("Sky");
    {
        const char* modeNames[] = {"None", "Color", "Skybox", "SkySphere"};
        int modeIdx = static_cast<int>(m_skyPass.mode());
        if (ImGui::Combo("Sky Mode", &modeIdx, modeNames, 4)) {
            m_skyPass.setMode(static_cast<SkyMode>(modeIdx));
            m_skyPass.requestRecapture();
        }
        if (m_skyPass.mode() != SkyMode::None && m_skyPass.mode() != SkyMode::Skybox) {
            glm::vec3 skyCol = m_skyPass.skyColor();
            if (ImGui::ColorEdit3("Sky Color", &skyCol.r)) {
                m_skyPass.setSkyColor(skyCol);
                m_skyPass.requestRecapture();
            }
        }
    }

    ImGui::Separator();
    ImGui::Text("Shadow");
    bool shadowOn = m_shadowPass.enabled();
    if (ImGui::Checkbox("Shadow maps", &shadowOn))
        m_shadowPass.setEnabled(shadowOn);
    ImGui::Text("Cascades: %u", m_shadowPass.cascadeCount());
    ImGui::SliderFloat("Receiver Bias", &m_shadowReceiverBias, 0.0f, 0.005f, "%.5f");

    // Shadow caster selector: auto picks the highest-intensity directional light.
    const auto lights = m_lights.snapshot();
    {
        char currentLabel[48];
        if (m_shadowLightIndex < 0)
            snprintf(currentLabel, sizeof(currentLabel), "Auto (brightest)");
        else
            snprintf(currentLabel, sizeof(currentLabel), "Light %d", m_shadowLightIndex);

        if (ImGui::BeginCombo("Shadow Caster", currentLabel)) {
            if (ImGui::Selectable("Auto (brightest)", m_shadowLightIndex < 0))
                m_shadowLightIndex = -1;
            if (m_shadowLightIndex < 0)
                ImGui::SetItemDefaultFocus();
            for (int i = 0; i < static_cast<int>(lights.size()); ++i) {
                if (!lights[i] || lights[i]->type() != Light::Type::Directional) continue;
                char label[48];
                snprintf(label, sizeof(label), "Light %d (intensity %.2f)",
                         i, lights[i]->commonState().intensity);
                const bool selected = (m_shadowLightIndex == i);
                if (ImGui::Selectable(label, selected))
                    m_shadowLightIndex = i;
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    ImGui::Separator();
    ImGui::Text("Lights (%d)", m_lights.enabledCount());
    for (size_t i = 0; i < lights.size(); ++i) {
        if (!lights[i]) continue;
        auto* l = lights[i].get();
        const char* typeName[] = {"Directional", "Point", "Spot"};
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::TreeNode("light", "[%zu] %s", i, typeName[static_cast<int>(l->type())])) {
            auto state = l->commonState();
            ImGui::Checkbox("Enabled",    &state.enabled);
            ImGui::ColorEdit3("Color",    &state.color.r);
            ImGui::DragFloat("Intensity", &state.intensity, 0.05f, 0.0f, 20.0f);
            l->setCommonState(state);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Render Graph (B3 / E3)");
    {
        if (ImGui::Button("Print DAG to stdout"))
            m_renderGraph.printOnNextExecute() = true;
        ImGui::SameLine();
        bool& trace = m_renderGraph.barrierTraceEnabled();
        ImGui::Checkbox("Barrier trace (E2)", &trace);
        ImGui::TextDisabled("Pass execution order:");
        const uint32_t nPasses = m_renderGraph.passCount();
        for (uint32_t i = 0; i < nPasses; ++i)
            ImGui::BulletText("[%u] %s", i, m_renderGraph.passName(i));
    }

    // -- E1: Per-Pass GPU Timing -----------------------------------------------
    ImGui::Separator();
    ImGui::Text("GPU Pass Timings (E1)");
    if (!m_gpuTimestamps.supported()) {
        ImGui::TextDisabled("  (not supported on this device/queue)");
    } else {
        const uint32_t nResults = m_gpuTimestamps.resultPassCount();
        if (nResults == 0) {
            ImGui::TextDisabled("  (waiting for first results...)");
        } else {
            // Collect and sort to identify top-3 most expensive passes.
            struct PassTiming { uint32_t idx; float avgMs; };
            std::vector<PassTiming> sorted;
            sorted.reserve(nResults);
            float totalMs = 0.0f;
            for (uint32_t i = 0; i < nResults; ++i) {
                const float avg = m_gpuTimestamps.rollingAvgMs(i);
                sorted.push_back({i, avg});
                totalMs += avg;
            }
            std::sort(sorted.begin(), sorted.end(),
                      [](const PassTiming& a, const PassTiming& b) {
                          return a.avgMs > b.avgMs;
                      });

            ImGui::Text("  %-28s  %8s  %8s", "Pass", "Last ms", "Avg ms");
            ImGui::Separator();
            for (uint32_t ri = 0; ri < nResults; ++ri) {
                const uint32_t i = sorted[ri].idx;
                const float last = m_gpuTimestamps.lastMs(i);
                const float avg  = m_gpuTimestamps.rollingAvgMs(i);
                const bool top3  = ri < 3;
                if (top3) ImGui::PushStyleColor(ImGuiCol_Text, {1.0f, 0.85f, 0.3f, 1.0f});
                ImGui::Text("  %-28s  %7.3f   %7.3f",
                            m_gpuTimestamps.passName(i), last, avg);
                if (top3) ImGui::PopStyleColor();
            }
            ImGui::Separator();
            ImGui::Text("  %-28s  %7.3f", "Total (sum of passes)", totalMs);
        }
    }

    // -- E3: Debug Visualization Modes -----------------------------------------
    ImGui::Separator();
    ImGui::Text("Debug Visualization (E3)");
    {
        const char* vizNames[] = {"Off", "Cascade Index", "World Normal", "Roughness/Metallic"};
        ImGui::Combo("Viz Mode", &m_debugVizMode, vizNames, 4);
        if (m_debugVizMode == 1) {
            ImGui::TextDisabled("  red=cascade0, green=cascade1, blue=cascade2, yellow=cascade3");
            ImGui::TextDisabled("  Enable shadow maps to see cascade boundaries.");
        } else if (m_debugVizMode == 2) {
            ImGui::TextDisabled("  RGB = world normal * 0.5 + 0.5");
        } else if (m_debugVizMode == 3) {
            ImGui::TextDisabled("  R=metallic, G=roughness (PBR materials only)");
        }
        // Culling stats.
        ImGui::Text("  Objects: %u  Batches: %u",
                    m_currentObjectCount,
                    static_cast<uint32_t>(m_currentCullBatches.size()));
    }

    // -- F3: Shader Hot Reload Status ------------------------------------------
    ImGui::Separator();
    ImGui::Text("Shader Hot Reload (F)");
    {
        using Status = ShaderWatcher::Status;
        const Status   st     = m_shaderWatcher.status();
        const bool     active = m_shaderWatcher.active();

        if (!active) {
            ImGui::TextDisabled("  Watcher inactive -- source dir not found.");
            ImGui::TextDisabled("  Set VKT_SHADER_SOURCE_DIR to the shaders/ directory.");
        } else {
            const char* statusStr[] = {"Idle", "Watching", "Compiling", "Ready", "Failed"};
            const int   statusIdx   = static_cast<int>(st);
            if (st == Status::Compiling)
                ImGui::TextColored({1.0f, 0.9f, 0.3f, 1.0f}, "  Status: %s", statusStr[statusIdx]);
            else if (st == Status::Failed)
                ImGui::TextColored({1.0f, 0.3f, 0.3f, 1.0f}, "  Status: %s", statusStr[statusIdx]);
            else if (st == Status::Ready)
                ImGui::TextColored({0.3f, 1.0f, 0.4f, 1.0f}, "  Status: %s", statusStr[statusIdx]);
            else
                ImGui::TextDisabled("  Status: %s", statusStr[statusIdx]);

            const auto result = m_shaderWatcher.lastResult();
            if (!result.timestamp.empty()) {
                if (result.success)
                    ImGui::Text("  Last compile: %s  [OK]", result.timestamp.c_str());
                else
                    ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f},
                                       "  Last compile: %s  [FAILED]", result.timestamp.c_str());
            }

            ImGui::TextDisabled("  Watching:");
            for (const auto& name : m_shaderWatcher.watchedNames())
                ImGui::BulletText("%s", name.c_str());

            if (ImGui::Button("Recompile Now"))
                m_shaderWatcher.requestRecompile();

            // Error log -- shown only on failure so it doesn't clutter normal use.
            if (st == Status::Failed && !result.errors.empty()) {
                ImGui::Separator();
                ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "  Compile errors:");
                ImGui::BeginChild("shader_errors", {0.0f, 120.0f}, true);
                ImGui::TextUnformatted(result.errors.c_str());
                ImGui::EndChild();
            }
        }
    }

    // -- G3: Bindless Textures (G) ---------------------------------------------
    ImGui::Separator();
    ImGui::Text("Bindless Textures (G)");
    {
        if (!m_bindlessSupported) {
            ImGui::TextColored({0.8f, 0.8f, 0.4f, 1.0f},
                               "  Not supported on this GPU (per-bind fallback)");
        } else {
            ImGui::Text("  Textures registered: %u / %u",
                        m_bindlessSet.count(), RenderLimits::kMaxBindlessTextures);
            ImGui::Checkbox("Enable Bindless Path (G3)", &m_useBindless);
            if (!m_useBindless)
                ImGui::TextDisabled("  Using per-batch descriptor set binds (fallback)");
        }
    }

    // -- E4: Capture Integration -----------------------------------------------
    ImGui::Separator();
    ImGui::Text("Capture (E4)");
    {
        ImGui::TextDisabled("F11 or button below wraps the next frame in a");
        ImGui::TextDisabled("\"CAPTURE_FRAME\" debug label for RenderDoc.");
        if (ImGui::Button("Mark Next Frame (F11)"))
            m_captureNextFrame = true;
        ImGui::SameLine();
        if (m_captureNextFrame)
            ImGui::TextColored({1.0f, 0.5f, 0.0f, 1.0f}, "PENDING");
        else
            ImGui::TextDisabled("idle");
    }

    ImGui::Separator();
    ImGui::Text("Async Streaming (C4)");
    {
        const auto c = m_streamLoader.counters();
        ImGui::Text("  Queued:    %u", c.queued);
        ImGui::Text("  Loading:   %u", c.loading);
        ImGui::Text("  Uploading: %u", c.uploading);
        ImGui::Text("  Completed: %u", c.completed);
        ImGui::Text("  Failed:    %u", c.failed);
        ImGui::Text("  Cancelled: %u", c.cancelled);

        // Demo: stream the IBL path on demand so the path can be exercised
        // without modifying scene files. The request is fire-and-forget;
        // the caller-side hot-swap pattern (write descriptor when Ready) is
        // shown as a comment because it depends on the material pipeline.
        if (!m_cfg.iblPath.empty()) {
            if (ImGui::Button("Stream Test: reload IBL texture")) {
                StreamHandle h = m_streamLoader.request(
                    m_cfg.iblPath, StreamLoader::ColorSpace::Linear);
                if (h == kInvalidStream)
                    ImGui::OpenPopup("stream_full");
                // When status(h) == StreamStatus::Ready, call result(h) to
                // get the final VkImageView + VkSampler and write the
                // descriptor (e.g. via DescriptorWriter::update).
            }
            if (ImGui::BeginPopup("stream_full")) {
                ImGui::Text("Stream queue full -- try again next frame.");
                ImGui::EndPopup();
            }
        }
    }

    // -- I1: GPU Crash Resiliency Status --------------------------------------
    ImGui::Separator();
    ImGui::Text("GPU Crash Resiliency (I)");
    {
        if (m_deviceLost) {
            ImGui::TextColored({1.0f, 0.2f, 0.2f, 1.0f}, "  DEVICE LOST");
            ImGui::Text("  Frame : %llu", static_cast<unsigned long long>(m_deviceLostInfo.frame));
            ImGui::Text("  Stage : %s",  m_deviceLostInfo.stage.c_str());
            ImGui::TextWrapped("  Passes: %s", m_deviceLostInfo.passNames.c_str());
            ImGui::TextDisabled("  Diagnostic logged to stderr. Close and restart.");
        } else {
            ImGui::TextColored({0.3f, 1.0f, 0.4f, 1.0f}, "  OK");
            ImGui::TextDisabled("  Monitoring: WaitForFences / AcquireNextImage /");
            ImGui::TextDisabled("              QueueSubmit / QueuePresent");
        }
    }

    // -- H1: Memory Budget Dashboard ------------------------------------------
    ImGui::Separator();
    ImGui::Text("Memory Budget (H)");
    {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(m_allocator.handle(), budgets);

        ImGui::Text("  %-6s  %-10s  %-10s  %-10s  %s",
                    "Heap", "Used (MB)", "Peak (MB)", "Budget (MB)", "Usage");
        ImGui::Separator();
        for (uint32_t i = 0; i < m_heapCount; ++i) {
            const VkDeviceSize used   = budgets[i].statistics.blockBytes;
            const VkDeviceSize budget = budgets[i].budget;

            // Update session peak.
            if (used > m_heapPeaks[i])
                m_heapPeaks[i] = used;

            const float usedMB   = static_cast<float>(used)          / (1024.0f * 1024.0f);
            const float peakMB   = static_cast<float>(m_heapPeaks[i])/ (1024.0f * 1024.0f);
            const float budgetMB = static_cast<float>(budget)        / (1024.0f * 1024.0f);
            const float fraction = (budget > 0)
                                   ? static_cast<float>(used) / static_cast<float>(budget)
                                   : 0.0f;

            // Colour the bar: green < 70%, yellow < 90%, red >= 90%.
            ImVec4 barCol = (fraction < 0.7f) ? ImVec4{0.2f, 0.8f, 0.3f, 1.0f}
                          : (fraction < 0.9f) ? ImVec4{0.9f, 0.8f, 0.1f, 1.0f}
                                              : ImVec4{0.9f, 0.2f, 0.2f, 1.0f};

            ImGui::PushID(static_cast<int>(i));
            ImGui::Text("  H%-5u  %10.2f  %10.2f  %10.2f",
                        i, usedMB, peakMB, budgetMB);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barCol);
            char overlay[32];
            snprintf(overlay, sizeof(overlay), "%.1f%%", fraction * 100.0f);
            ImGui::ProgressBar(fraction, {-1.0f, 0.0f}, overlay);
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
        if (ImGui::Button("Reset Peaks"))
            for (uint32_t i = 0; i < m_heapCount; ++i)
                m_heapPeaks[i] = 0;
    }

    ImGui::End();
}

void Application::handleResize() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    if (w == 0 || h == 0) {
        // Still minimized ? keep the flag set so drawFrame() retries next time.
        m_framebufferResized = true;
        return;
    }
    m_framebufferResized = false;
    if (vkDeviceWaitIdle(m_ctx.device()) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed while waiting for device idle during resize");

    const uint32_t oldImageCount = m_swapchain.imageCount();
    m_swapchain.rebuild(static_cast<uint32_t>(w), static_cast<uint32_t>(h));

    // Recreate per-image semaphores if the swapchain image count changed
    if (m_swapchain.imageCount() != oldImageCount) {
        destroyRenderFinishedSemaphores();
        createRenderFinishedSemaphores();
    }

    // Resize the HDR offscreen images + framebuffers to match the new window size.
    // The HDR render pass and blit pipeline are stable and do not need rebuilding.
    m_postProcess.rebuild(m_allocator.handle(),
                          VkExtent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)});

    m_imgui.onSwapchainRebuilt(m_swapchain.imageCount());
    m_camera.setAspect(static_cast<float>(w) / static_cast<float>(h));
    createRenderGraph();
}

// -- Input ----------------------------------------------------------------------

void Application::processInput(float dt) {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        m_running = false;

    // O key toggles perspective / orthographic projection (rising-edge only)
    const bool oNow = glfwGetKey(m_window, GLFW_KEY_O) == GLFW_PRESS;
    if (oNow && !m_oPrevPressed) {
        using PM = Camera::ProjectionMode;
        m_camera.setProjectionMode(
            m_camera.projectionMode() == PM::Perspective ? PM::Orthographic : PM::Perspective);
    }
    m_oPrevPressed = oNow;

    // F11 key: trigger a "CAPTURE_FRAME" debug label for RenderDoc one-click capture (E4).
    const bool captureNow = glfwGetKey(m_window, GLFW_KEY_F11) == GLFW_PRESS;
    if (captureNow && !m_captureKeyPrev)
        m_captureNextFrame = true;
    m_captureKeyPrev = captureNow;

    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && !m_mouseCapture) {
        m_mouseCapture = true;
        m_firstMouse   = true;
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE && m_mouseCapture) {
        m_mouseCapture = false;
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    const float speed    = m_cfg.cameraSpeed;
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) m_camera.moveForward( speed * dt);
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) m_camera.moveForward(-speed * dt);
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) m_camera.moveRight( speed * dt);
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) m_camera.moveRight(-speed * dt);
    if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS) m_camera.moveUp( speed * dt);
    if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS) m_camera.moveUp(-speed * dt);

    // Arrow keys rotate the camera (no mouse required)
    const float lookDelta = m_cfg.keyboardLookSpeed * dt;
    float dyaw = 0.0f, dpitch = 0.0f;
    if (glfwGetKey(m_window, GLFW_KEY_LEFT)  == GLFW_PRESS) dyaw   -= lookDelta;
    if (glfwGetKey(m_window, GLFW_KEY_RIGHT) == GLFW_PRESS) dyaw   += lookDelta;
    if (glfwGetKey(m_window, GLFW_KEY_UP)    == GLFW_PRESS) dpitch += lookDelta;
    if (glfwGetKey(m_window, GLFW_KEY_DOWN)  == GLFW_PRESS) dpitch -= lookDelta;
    if (dyaw != 0.0f || dpitch != 0.0f)
        m_camera.rotate(dyaw, dpitch);
}

void Application::onMouseMove(GLFWwindow* window, double x, double y) {
    // glfwSetWindowUserPointer(m_window, this) stores an Application*.
    // GLFW guarantees the pointer is passed back unchanged, so this cast is safe.
    auto* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app->m_mouseCapture) return;

    if (app->m_firstMouse) {
        app->m_lastMouseX = x;
        app->m_lastMouseY = y;
        app->m_firstMouse = false;
    }
    const float dx = static_cast<float>(x - app->m_lastMouseX) * app->m_mouseSensitivity;
    const float dy = static_cast<float>(app->m_lastMouseY - y) * app->m_mouseSensitivity; // inverted Y
    app->m_lastMouseX = x;
    app->m_lastMouseY = y;
    app->m_camera.rotate(dx, dy);
}

void Application::onFramebufferResize(GLFWwindow* window, int w, int h) {
    // Safe: window user pointer is always Application* (set in createWindow).
    auto* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    if (w == 0 || h == 0) return;
    app->m_framebufferResized = true;
}

void Application::onWindowScale(GLFWwindow* window, float xscale, float) {
    // Safe: window user pointer is always Application* (set in createWindow).
    auto* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    app->m_framebufferResized = true;
    app->m_imgui.onDpiChange(xscale);
}

// -- Track I: GPU Crash Resiliency ----------------------------------------------

void Application::reportDeviceLost(VkResult result, const char* stage) {
    if (m_deviceLost) return; // report only the first occurrence

    m_deviceLost = true;

    // Collect active render pass names for context.
    std::string passNames;
    const uint32_t nPasses = m_renderGraph.passCount();
    for (uint32_t i = 0; i < nPasses; ++i) {
        if (i > 0) passNames += "; ";
        passNames += m_renderGraph.passName(i);
    }
    if (passNames.empty()) passNames = "(no passes registered)";

    m_deviceLostInfo.frame     = m_frameCount;
    m_deviceLostInfo.stage     = stage ? stage : "(unknown)";
    m_deviceLostInfo.passNames = passNames;

    std::fprintf(stderr,
        "[App] VK_ERROR_DEVICE_LOST -- result=0x%08X\n"
        "  Stage      : %s\n"
        "  Frame      : %llu\n"
        "  RG passes  : %s\n"
        "  Action     : beginning graceful shutdown (no device recreation).\n",
        static_cast<unsigned>(result),
        m_deviceLostInfo.stage.c_str(),
        static_cast<unsigned long long>(m_deviceLostInfo.frame),
        m_deviceLostInfo.passNames.c_str());
    std::fflush(stderr);
}

// -- Shutdown -------------------------------------------------------------------

void Application::shutdown() {
    if (!m_window) return;

    // I1: Skip vkDeviceWaitIdle on a lost device -- it will fail or hang.
    // All Vulkan destroy calls below are still valid on a lost device (per spec).
    if (m_deviceLost) {
        std::fprintf(stderr, "[App] shutdown: device is lost -- skipping vkDeviceWaitIdle\n");
    } else if (vkDeviceWaitIdle(m_ctx.device()) != VK_SUCCESS) {
        // shutdown() can be reached from the destructor; avoid throwing here.
        std::fprintf(stderr, "[App] Warning: vkDeviceWaitIdle failed during shutdown\n");
    }

    // Serialize pipeline cache before any resources are freed
    savePipelineCache();

    m_imgui.destroy();

    destroyRenderFinishedSemaphores();

    for (auto& f : m_frames) {
        vkDestroyCommandPool(m_ctx.device(), f.commandPool, nullptr);
        vkDestroySemaphore(m_ctx.device(), f.imageAvailableSemaphore, nullptr);
        vkDestroyFence(m_ctx.device(), f.inFlightFence, nullptr);
    }

    // Destroy async streaming before any GPU resources are freed.
    // destroy() waits for any in-flight upload fences before releasing Images.
    m_streamLoader.destroy();

    // F1: Stop the shader watcher thread before Vulkan objects are destroyed.
    m_shaderWatcher.destroy();

    // F2: Destroy any deferred pipelines that are still pending.
    for (auto& dp : m_deferredPipelineDestroys)
        if (dp.pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(m_ctx.device(), dp.pipeline, nullptr);
    m_deferredPipelineDestroys.clear();

    // E1: Destroy GPU timestamp query pools.
    m_gpuTimestamps.destroy();

    // Destroy shadow + post-process + cull + sky passes before the pipeline and allocator.
    m_shadowPass.destroy();
    m_postProcess.destroy();
    m_cullPass.destroy();
    for (auto& ib : m_indirectBuffers) ib.destroy();
    m_skyPass.destroy();

    // Destroy material SSBO before the allocator.
    m_materials.destroy();

    // Destroy all mesh GPU resources (vertex/index buffers) before the allocator.
    m_world.destroy();
    m_groundId = INVALID_HANDLE;
    m_sphereId = INVALID_HANDLE;
    m_cubeId   = INVALID_HANDLE;

    // Destroy file-loaded textures (Image GPU resources) before the allocator.
    m_sceneTextures.clear();

    // Destroy IBL resources before the allocator.
    m_iblBrdfLut.destroy();
    m_iblPrefilter.destroy();
    m_iblIrradiance.destroy();
    m_iblEnvCubemap.destroy();
    m_iblLutSampler.destroy();
    m_iblCubeSampler.destroy();

    // Destroy default normal image + sampler.
    m_defaultNormalSampler.destroy();
    m_defaultNormalImage.destroy();

    m_defaultSampler.destroy();
    m_defaultImage.destroy();
    for (auto& ub : m_uniformBuffers) ub.destroy();
    for (auto& ib : m_instanceBuffers) ib.destroy();
    m_descriptorPool.destroy();
    vkDestroyDescriptorSetLayout(m_ctx.device(), m_globalLayout,   nullptr);
    vkDestroyDescriptorSetLayout(m_ctx.device(), m_materialLayout, nullptr);
    vkDestroyPipeline(m_ctx.device(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_ctx.device(), m_pipelineLayout, nullptr);

    // G1/G2: Destroy bindless resources.
    if (m_bindlessPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx.device(), m_bindlessPipeline, nullptr);
        m_bindlessPipeline = VK_NULL_HANDLE;
    }
    if (m_bindlessPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx.device(), m_bindlessPipelineLayout, nullptr);
        m_bindlessPipelineLayout = VK_NULL_HANDLE;
    }
    m_bindlessSet.destroy();

    if (m_pipelineCache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(m_ctx.device(), m_pipelineCache, nullptr);
        m_pipelineCache = VK_NULL_HANDLE;
    }

    m_renderPass.destroy();
    m_swapchain.destroy();
    m_allocator.destroy();
    m_ctx.destroy();

    glfwDestroyWindow(m_window);
    glfwTerminate();
    m_window = nullptr;
}

Application::~Application() { shutdown(); }

} // namespace vkt
