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
    createDescriptors();
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
    loadScene();
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
    //        + material SSBO (binding 3) -- one per frame in flight.
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
        .build(m_ctx.device());

    // Set 1: per-material texture
    m_materialLayout = DescriptorLayoutBuilder{}
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
        .build(m_ctx.device());

    // Pool capacity is runtime-configurable for scene scalability.
    // Extra COMBINED_IMAGE_SAMPLER ratio: global sets (binding=2, shadow map) + per-material sets.
    const uint32_t maxSets = std::max(64u, m_cfg.maxDescriptorSets);
    m_descriptorPool.init(m_ctx.device(), maxSets, {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1.0f},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1.0f},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6.0f},
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

// -- Pipeline cache -------------------------------------------------------------

void Application::createPipelineCache() {
    VkPipelineCacheCreateInfo cacheInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};

    // Attempt to seed the cache with data saved from a previous run.
    // The driver validates the data internally; corrupted or mismatched data
    // (different GPU / driver version) is silently discarded.
    std::vector<uint8_t> cacheData;
    if (std::ifstream f("pipeline_cache.bin", std::ios::binary | std::ios::ate); f.good()) {
        const auto size = static_cast<size_t>(f.tellg());
        f.seekg(0);
        cacheData.resize(size);
        f.read(reinterpret_cast<char*>(cacheData.data()),
               static_cast<std::streamsize>(size));
        cacheInfo.initialDataSize = cacheData.size();
        cacheInfo.pInitialData    = cacheData.data();
        std::printf("[App] Pipeline cache loaded (%zu bytes)\n", size);
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

        m_defaultSampler.create(m_ctx.device(), m_ctx.physicalDevice(), Sampler::linearConfig());
        m_defaultMaterialSet = m_descriptorPool.allocate(m_materialLayout);
        DescriptorWriter{}
            .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        m_defaultImage.view(), m_defaultSampler.handle(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .update(m_ctx.device(), m_defaultMaterialSet);
        debug::setObjectName(m_ctx.device(), VK_OBJECT_TYPE_IMAGE,
                             m_defaultImage.handle(), "Default White Texture");

        // Register with the world so SceneLoader / user code can look it up by name.
        m_world.registerTexture("default", m_defaultMaterialSet);

        // -- Scene objects ---------------------------------------------------------
        // -- Demo materials --------------------------------------------------------
        // Registered before scene objects so IDs can be passed to spawn().
        // Index 0 is always the default white material (registered in init()).
        const uint32_t matGround = m_materials.add("ground",
            GPUMaterial{ {0.45f, 0.45f, 0.45f, 1.0f} });          // dark grey
        const uint32_t matSphere = m_materials.add("sphere",
            GPUMaterial{ {0.2f, 0.6f, 1.0f, 1.0f}, 0.5f, 0.0f, 0.4f }); // blue+glow
        const uint32_t matCube   = m_materials.add("cube",
            GPUMaterial{ {1.0f, 0.35f, 0.1f, 1.0f} });             // orange

        if (!m_cfg.sceneFile.empty()) {
            // File-based path: SceneLoader creates meshes, loads textures, spawns objects.
            SceneLoadContext ctx{
                m_allocator.handle(), m_ctx.device(), m_ctx.physicalDevice(),
                uploadCmd, &m_descriptorPool, m_materialLayout
            };
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
    if (vkDeviceWaitIdle(m_ctx.device()) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed while waiting for device idle at end of run");
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

    if (vkWaitForFences(m_ctx.device(), 1, &f.inFlightFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        throw std::runtime_error("[App] Failed while waiting for in-flight fence");

    uint32_t imageIndex;
    VkResult acqResult = m_swapchain.acquireNextImage(f.imageAvailableSemaphore, imageIndex);
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

    if (vkQueueSubmit(m_ctx.graphicsQueue(), 1, &submit, f.inFlightFence) != VK_SUCCESS)
        throw std::runtime_error("[App] Queue submit failed");

    VkSwapchainKHR sc = m_swapchain.handle();
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &renderFinished;
    present.swapchainCount     = 1;
    present.pSwapchains        = &sc;
    present.pImageIndices      = &imageIndex;

    VkResult presResult = vkQueuePresentKHR(m_ctx.presentQueue(), &present);
    if (presResult == VK_ERROR_OUT_OF_DATE_KHR || presResult == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
        handleResize();
    } else if (presResult != VK_SUCCESS) {
        throw std::runtime_error("[App] Failed to present swapchain image");
    }

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Application::recordCommands(VkCommandBuffer cmd, uint32_t imageIndex) {
    // -- Shadow pass: CPU-built batches -----------------------------------------
    // buildBatchesNoCull() writes all alive objects (up to SSBO capacity) so
    // off-screen casters can still cast shadows into the camera view. The
    // compute cull pass then overwrites the same SSBO for the main geometry pass.
    InstanceBuffer& ib = m_instanceBuffers[m_currentFrame];
    const auto shadowBatches = m_world.buildBatchesNoCull(
        ib.data(), ib.capacity(), m_camera.worldPosition());
    uint32_t shadowInstanceCount = 0;
    for (const auto& b : shadowBatches) shadowInstanceCount += b.count;
    ib.flush(shadowInstanceCount);

    // Flush material edits. No-op if no material was changed this frame.
    m_materials.flush();

    if (m_shadowPass.enabled()) {
        debug::LabelScope shadowLabel(cmd, "Shadow", {0.8f, 0.4f, 0.1f, 1.0f});
        m_shadowPass.render(cmd, m_globalSets[m_currentFrame], shadowBatches);
    }

    // -- GPU compute cull pass (Phase 6) ---------------------------------------
    // Build CullObject array (all alive objects, no CPU frustum test).
    std::vector<CullBatch> cullBatches;
    const uint32_t objectCount = m_world.buildCullData(
        m_cullPass.objectData(m_currentFrame),
        m_cullPass.objectCapacity(),
        RenderLimits::kMaxBatches,
        cullBatches,
        m_camera.worldPosition());

    // Fill the indirect buffer: CPU sets static fields + instanceCount = 0.
    // Compute shader atomicAdd-increments instanceCount for each visible object.
    IndirectBuffer& indr = m_indirectBuffers[m_currentFrame];
    for (const auto& b : cullBatches) {
        auto& entry             = indr.data()[b.batchId];
        entry.indexCount        = b.mesh->indexCount();
        entry.instanceCount     = 0; // GPU compute fills this via atomicAdd
        entry.firstIndex        = 0; // each Mesh owns a dedicated index buffer starting at 0
        entry.vertexOffset      = 0; // no base-vertex offset (each Mesh has its own vertex buffer)
        entry.firstInstance     = b.firstInstance;
    }
    indr.flush(static_cast<uint32_t>(cullBatches.size()));

    // Barrier: HOST writes (instance SSBO from buildBatches + indirect + cull objects)
    // must be visible to the compute shader before it reads / writes them.
    // Also synchronizes: shadow VERTEX_SHADER reads -> COMPUTE writes on instance SSBO.
    {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    {
        debug::LabelScope cullLabel(cmd, "GPU Cull", {0.5f, 0.9f, 0.5f, 1.0f});
        const Frustum frustum = Frustum::fromVP(m_viewProj);
        m_cullPass.dispatch(cmd, m_currentFrame, frustum.planes.data(), objectCount);
    }

    // Barrier: compute SHADER_WRITE (instance SSBO + indirect) must be visible
    // before VERTEX_SHADER reads instances and DRAW_INDIRECT reads commands.
    {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // -- HDR geometry pass: indirect draw (GPU-computed instance counts) -------
    {
        debug::LabelScope hdrLabel(cmd, "HDR Geometry", {0.2f, 0.6f, 0.9f, 1.0f});
        m_postProcess.beginHdr(cmd, m_currentFrame);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        // Bind global set (set 0: UBO + SSBO + shadow map + material SSBO).
        // The instance SSBO at binding 1 now contains compute-written visible data.
        // Set 1 (texture) is bound per-batch inside drawIndirect().
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipelineLayout, 0, 1,
                                &m_globalSets[m_currentFrame], 0, nullptr);
        m_world.drawIndirect(cmd, m_pipelineLayout, cullBatches, indr.handle());

        m_postProcess.endHdr(cmd);
    }

    // -- Swapchain pass: tonemap blit + UI ------------------------------------
    m_renderPass.begin(cmd, imageIndex);
    {
        debug::LabelScope blitLabel(cmd, "Post Blit", {0.6f, 0.9f, 0.2f, 1.0f});
        m_postProcess.blit(cmd, m_currentFrame);
    }
    {
        debug::LabelScope uiLabel(cmd, "UI", {0.9f, 0.6f, 0.2f, 1.0f});
        m_imgui.newFrame();
        renderUI();
        m_imgui.render(cmd);
    }
    m_renderPass.end(cmd);
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

// -- Shutdown -------------------------------------------------------------------

void Application::shutdown() {
    if (!m_window) return;

    if (vkDeviceWaitIdle(m_ctx.device()) != VK_SUCCESS) {
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

    // Destroy shadow + post-process + cull passes before the pipeline and allocator.
    m_shadowPass.destroy();
    m_postProcess.destroy();
    m_cullPass.destroy();
    for (auto& ib : m_indirectBuffers) ib.destroy();

    // Destroy material SSBO before the allocator.
    m_materials.destroy();

    // Destroy all mesh GPU resources (vertex/index buffers) before the allocator.
    m_world.destroy();
    m_groundId = INVALID_HANDLE;
    m_sphereId = INVALID_HANDLE;
    m_cubeId   = INVALID_HANDLE;

    // Destroy file-loaded textures (Image GPU resources) before the allocator.
    m_sceneTextures.clear();

    m_defaultSampler.destroy();
    m_defaultImage.destroy();
    for (auto& ub : m_uniformBuffers) ub.destroy();
    for (auto& ib : m_instanceBuffers) ib.destroy();
    m_descriptorPool.destroy();
    vkDestroyDescriptorSetLayout(m_ctx.device(), m_globalLayout,   nullptr);
    vkDestroyDescriptorSetLayout(m_ctx.device(), m_materialLayout, nullptr);
    vkDestroyPipeline(m_ctx.device(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_ctx.device(), m_pipelineLayout, nullptr);

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
