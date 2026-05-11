#pragma once

#include "core/Context.h"
#include "core/SwapChain.h"
#include "core/Allocator.h"
#include "core/FrameData.h"
#include "pipeline/RenderPass.h"
#include "pipeline/Shader.h"
#include "pipeline/Descriptors.h"
#include "pipeline/PipelineBuilder.h"
#include "resources/Buffer.h"
#include "resources/Image.h"
#include "resources/Sampler.h"
#include "resources/Texture.h"
#include "utils/DebugUtils.h"
#include "utils/Timer.h"
#include "scene/Camera.h"
#include "scene/InstanceBuffer.h"
#include "scene/Light.h"
#include "scene/RenderWorld.h"
#include "render/ShadowPass.h"
#include "render/PostProcessPass.h"
#include "render/CullPass.h"
#include "render/IndirectBuffer.h"
#include "render/SkyPass.h"
#include "render/IblBaker.h"
#include "render/RenderGraph.h"
#include "pipeline/BindlessTextureSet.h"
#include "render/GpuTimestamps.h"
#include "render/ShaderWatcher.h"
#include "render/StreamLoader.h"
#include "material/Material.h"
#include "ui/ImGuiLayer.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>
#include <limits>

namespace vkt {

// Per-frame uniform data uploaded to set 0, binding 0.
// Must stay in sync with GlobalUBO in default.vert/.frag (std140 layout).
// Total size (kMaxShadowCascades=4): 1008 bytes.
struct GlobalUBO {
    glm::mat4 view;                                      // 64 bytes  offset   0
    glm::mat4 proj;                                      // 64 bytes  offset  64
    glm::vec4 ambientColor{0.05f, 0.05f, 0.05f, 1.0f};  // 16 bytes  offset 128  rgb=color, a=intensity
    glm::vec4 cameraPos   {0.0f, 0.0f, 0.0f, 0.0f};     // 16 bytes  offset 144  xyz=world pos, w=unused
    GPULight  lights[MAX_LIGHTS];                        // 512 bytes offset 160  (8 x 64)
    int32_t   lightCount  {0};                           //  4 bytes  offset 672
    float     _pad[3]     {};                            // 12 bytes  offset 676  keeps struct 16-byte aligned
    // -- CSM shadow fields -----------------------------------------------------
    glm::mat4 lightSpaceMatrices[RenderLimits::kMaxShadowCascades]{}; // 256 bytes offset 688
    glm::vec4 cascadeSplits{0.0f};                       // 16 bytes  offset 944
    glm::ivec4 shadowMeta{1, 0, 0, 0};                   // 16 bytes  offset 960  x=cascadeCount, y=enabled
    glm::vec4  shadowParams{0.00015f, 0.0f, 0.0f, 0.0f};  // 16 bytes  offset 976  x=receiver bias
    glm::ivec4 iblParams   {0, 4, 0, 0};                  // 16 bytes  offset 992  x=iblEnabled, y=maxPrefilterLod
};
static_assert(sizeof(GlobalUBO) == 1008, "GlobalUBO size mismatch  --  update shader UBO layout");

// Pipeline specialization constants consumed by shaders at pipeline-create time.
// constant_id = 0 in default.frag maps to lightCap.
struct ShaderSpecializationConstants {
    int32_t lightCap = MAX_LIGHTS;
};

class Application {
public:
    struct Config {
        std::string           title            = "Vulkan Template";
        uint32_t              width            = 1280;
        uint32_t              height           = 720;
        // Desired MSAA level; clamped to the device maximum at runtime.
        // Pass VK_SAMPLE_COUNT_1_BIT to disable MSAA entirely.
        VkSampleCountFlagBits preferredMsaa    = VK_SAMPLE_COUNT_4_BIT;
        // Enable Vulkan validation layers (requires the Vulkan SDK to be installed).
        // Alternatively controlled at compile time via the VKT_VALIDATION define.
        bool                  validation       = false;
        // Degrees of rotation per pixel of mouse movement (RMB camera look).
        float                 mouseSensitivity = 0.1f;
        // Open in borderless fullscreen on the primary monitor.
        // The window size fields are ignored when fullscreen is true.
        bool                  fullscreen       = false;
        // Fragment shader light loop cap (specialization constant id 0).
        // Clamped to [1, MAX_LIGHTS].
        uint32_t              shaderLightCap   = MAX_LIGHTS;
        // Descriptor pool capacity for this app instance.
        // Increase for large scene files with many unique textures/material sets.
        uint32_t              maxDescriptorSets = 1024;
        // Camera fly speed in world units per second (WASD/QE keys).
        float                 cameraSpeed      = 5.0f;
        // Camera rotation speed in degrees per second (arrow keys).
        float                 keyboardLookSpeed = 90.0f;
        // Camera perspective projection parameters.
        float                 cameraFov        = 60.0f;   // vertical FoV in degrees
        float                 cameraNear       = 0.1f;    // near plane distance
        float                 cameraFar        = 1000.0f; // far plane distance
        // Ambient light color (rgb).  Intensity is always 1 (use rgb brightness to adjust).
        glm::vec3             ambientColor     = {0.05f, 0.05f, 0.05f};
        // Path to a JSON scene file (see assets/scene.json for the format).
        // Leave empty to use the built-in demo scene.
        std::string           sceneFile        = "";
        // Post-processing defaults (adjustable at runtime via ImGui).
        float                 exposure         = 1.0f;
        float                 gamma            = 2.2f;
        int                   tonemapMode      = 0;   // 0=Reinhard, 1=ACES, 2=Clamp
        // CSM startup settings.
        uint32_t              shadowCascades   = 4;     // clamped to RenderLimits::kMaxShadowCascades
        uint32_t              shadowResolution = 2048;  // per-cascade map size
        float                 shadowSplitLambda = 0.65f;
        // Receiver bias used in the shading pass depth compare.
        float                 shadowReceiverBias = 0.00015f;
        // IBL settings.
        bool          iblEnabled          = true; // enable IBL diffuse+specular
        int           iblMaxPrefilterLod  = 4;     // mip count in prefiltered env map
        std::string   iblPath             = "assets/sample.hdr";    // equirectangular .hdr file; empty = placeholder IBL
        // Sky settings.
        SkyPass::Config sky;
        // VSync: locks the frame rate to the monitor refresh via FIFO present mode
        // (zero CPU overhead, driver-level frame pacing, no tearing).
        // Disabling uses Mailbox (triple-buffer, uncapped, low latency).
        // When vsync is on, targetFps still works as an additional software cap.
        bool                  vsync              = false;
        // Software FPS cap (sleep+spin hybrid for accuracy). 0 = no cap.
        // The loop always throttles to ~30 FPS while minimized regardless.
        uint32_t              targetFps          = 120;
    };

    Application() = default;
    ~Application();
    Application(const Application&)            = delete;
    Application& operator=(const Application&) = delete;

    void init(const Config& cfg = {});
    void run();
    void shutdown();

    // Access the scene graph for dynamic object management at runtime.
    // Typical simulation use: spawn/despawn objects and call setTransform each tick.
    RenderWorld& world() { return m_world; }

protected:
    // Called once per render frame, before drawFrame().
    // dt is double-precision seconds (same precision as UpdateLoop::simulationTime).
    //
    // Override in a subclass to integrate an UpdateLoop:
    //   void onBeforeDraw(double dt) override {
    //       m_updater.advance(dt, world());       // non-threaded
    //       // or:
    //       m_updater.applyToWorld(world());      // threaded
    //   }
    virtual void onBeforeDraw(double /*dt*/) {}

private:
    // -- Initialization steps --------------------------------------------------
    void createWindow(const Config& cfg);
    void initVulkan(const Config& cfg);
    void createFrameData();
    void createDescriptors();
    void createPipelineCache();   // load from disk or create fresh
    void createPipeline();
    void loadScene();
    std::string resolveShaderPath(const char* fileName) const;
    // Write shadow sampler + image view into global descriptor sets (binding=2).
    // Called after ShadowPass::init() so the image and sampler exist.
    void updateShadowDescriptors();

    // Initialize the PostProcessPass (HDR render pass + blit pipeline).
    // Must be called after createPipelineCache() and before createPipeline().
    void createPostProcess(const Config& cfg);

    // Initialize the CullPass and IndirectBuffers for GPU compute culling.
    // Must be called after createDescriptors() (needs instance buffer handles).
    void createCullPass();

    // Create placeholder IBL textures and write them into global descriptor sets
    // (bindings 4=irradiance, 5=prefilter, 6=BRDF LUT).
    void createIblResources();
    void updateIblDescriptors();
    void rebuildSkyIblProbe();
    void updateSolidCubemap(Image& image, const glm::vec4& color);
    void updateSolidImage2D(Image& image, const glm::vec4& color);
    VkCommandBuffer beginImmediateCommands(VkCommandPool& pool) const;
    void endImmediateCommands(VkCommandPool pool, VkCommandBuffer cmd) const;

    // Initialize the sky background pass.
    void createSkyPass(const Config& cfg);

    // Build the render graph: register resources and passes, derive barriers.
    // Called from init() after all passes are initialized.
    // Re-called from handleResize() if image resource handles change.
    void createRenderGraph();

    // G1: Initialize bindless texture set and detect descriptor indexing support.
    // Must be called after createDescriptors() (needs device + descriptor indexing query).
    void createBindlessResources();

    // Initialize GPU timestamp query pools (Track E1).
    // Must be called after initVulkan() (needs device, physDevice).
    void createGpuTimestamps();

    // Start the shader file watcher (Track F1).
    // Must be called after loadScene() (output dir resolved from exe path).
    void createShaderWatcher();

    // F2: Called each frame; if ShaderWatcher has new SPIR-V, rebuild m_pipeline.
    // Returns true if the pipeline was swapped.
    bool tryHotReloadPipeline();

    // Initialize the async texture streaming system (Track C).
    // Must be called after createDescriptors() (needs m_defaultImage + m_defaultSampler).
    void createStreamLoader();

    // CPU-side frame setup: build batch lists and flush per-frame GPU buffers.
    // Called at the top of recordCommands(), before m_renderGraph.execute().
    void prepareFrameData();

    // -- Per-frame logic -------------------------------------------------------
    void drawFrame();
    void recordCommands(VkCommandBuffer cmd, uint32_t imageIndex);
    void renderUI();
    void handleResize();

    // -- Shutdown helpers ------------------------------------------------------
    void savePipelineCache();     // serialize to disk

    // Per-swapchain-image semaphores.  The presentation engine retains a
    // renderFinished semaphore until the image is re-acquired, so there must be
    // one per image  --  not one per frame slot.  Reusing a frame-slot semaphore
    // on a triple-buffered swapchain violates VUID-vkQueueSubmit-00067.
    void createRenderFinishedSemaphores();
    void destroyRenderFinishedSemaphores();

    // -- Input -----------------------------------------------------------------
    void processInput(float dt);
    static void onFramebufferResize(GLFWwindow* window, int w, int h);
    static void onWindowScale(GLFWwindow* window, float xscale, float yscale);
    static void onMouseMove(GLFWwindow* window, double x, double y);

    // -- Vulkan objects --------------------------------------------------------
    GLFWwindow*  m_window   = nullptr;
    Context      m_ctx;
    Allocator    m_allocator;
    SwapChain    m_swapchain;
    RenderPass   m_renderPass;
    ImGuiLayer   m_imgui;

    // Shadow pass: depth-only pre-pass that produces the shadow map.
    // Initialized after createDescriptors() (needs globalLayout + pipelineCache).
    ShadowPass m_shadowPass;

    // Post-process pass: geometry renders to an HDR offscreen target;
    // blit() tonemaps + gamma-corrects into the swapchain render pass.
    // Initialized after createPipelineCache() (needs stable render pass + cache).
    PostProcessPass m_postProcess;

    // Material manager: CPU-side GPUMaterial array flushed to set=0 binding=3 each frame.
    // Initialized in createDescriptors(); index 0 = default white material.
    MaterialManager m_materials;

    // IBL baker: used once during init if iblPath is set; destroyed after baking.
    IblBaker m_iblBaker;

    // IBL resources (set=0 bindings 4,5,6): placeholder cubemaps + BRDF LUT.
    // Destroyed in shutdown() before the allocator.
    Image   m_iblIrradiance;
    Image   m_iblPrefilter;
    Image   m_iblBrdfLut;
    Image   m_iblEnvCubemap;    // 512x512 env cube -- given to SkyPass for sky display
    Sampler m_iblCubeSampler;   // linear+clamp for irradiance and prefiltered
    Sampler m_iblLutSampler;    // linear+clamp for BRDF LUT
    int     m_iblBakedMaxPrefilterLod = 0;

    // Sky background pass: renders before geometry inside the HDR render pass.
    SkyPass m_skyPass;

    // Render graph: declares pass/resource dependencies and derives barriers.
    RenderGraph m_renderGraph;

    // Async texture streaming (Track C): init after m_defaultImage/Sampler exist.
    StreamLoader m_streamLoader;

    // -- Track E: Developer Observability --------------------------------------

    // E1: Per-pass GPU timestamp queries.
    // init() called after createFrameData(); destroy() before m_ctx in shutdown().
    GpuTimestamps m_gpuTimestamps;

    // Nanoseconds per GPU timer tick (from VkPhysicalDeviceLimits.timestampPeriod).
    float    m_timestampPeriod = 1.0f;

    // Total frames submitted so far.  Used to skip retrieve() for the first
    // MAX_FRAMES_IN_FLIGHT frames (query pools are uninit before first submission).
    uint64_t m_frameCount = 0;

    // Per-frame batch lists populated by prepareFrameData() and consumed by graph passes.
    std::vector<DrawBatch> m_currentShadowBatches;
    std::vector<CullBatch> m_currentCullBatches;
    uint32_t               m_currentObjectCount = 0;
    uint32_t               m_currentImageIndex  = 0;

    // Phase 6: GPU compute frustum culling + indirect draw.
    // CullPass owns the compute pipeline and per-frame CullObject SSBOs.
    // IndirectBuffers hold VkDrawIndexedIndirectCommand per batch per frame.
    CullPass m_cullPass;
    std::array<IndirectBuffer, MAX_FRAMES_IN_FLIGHT> m_indirectBuffers;

    // Pipeline
    VkPipelineCache       m_pipelineCache  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_globalLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            m_pipeline       = VK_NULL_HANDLE;

    // -- Track G: Bindless Materials/Textures ----------------------------------

    // G1: Runtime feature detection; set in createDescriptors().
    bool m_bindlessSupported = false;
    // G3: Runtime toggle; true when bindless is both supported and enabled.
    bool m_useBindless = false;

    // G1: Bindless sampler2D[] descriptor set for bindless texture access.
    // Populated during loadScene() (index 0 = default white, index 1 = default normal).
    BindlessTextureSet m_bindlessSet;

    // G2: Bindless pipeline layout: set=0 global, set=1 bindless array.
    // Separate from m_pipelineLayout (which uses per-material set=1 instead).
    VkPipelineLayout m_bindlessPipelineLayout = VK_NULL_HANDLE;

    // G2: Bindless geometry pipeline using default_bindless.frag.
    VkPipeline m_bindlessPipeline = VK_NULL_HANDLE;

    // -- Track H: Memory Budget Dashboard -------------------------------------

    // Number of Vulkan memory heaps on this device (from physicalDeviceMemoryProperties).
    // Queried once in initVulkan(); used to bound the heap loop in renderUI().
    uint32_t    m_heapCount = 0;

    // Session-peak GPU allocation per heap (our allocator's blockBytes, in bytes).
    // Updated each frame in renderUI() as we read VmaBudget.statistics.blockBytes.
    VkDeviceSize m_heapPeaks[VK_MAX_MEMORY_HEAPS]{};

    DescriptorPool m_descriptorPool;

    // Per-frame resources (indexed by currentFrame, 0..MAX_FRAMES_IN_FLIGHT-1)
    FrameArray                                           m_frames;
    std::array<Buffer, MAX_FRAMES_IN_FLIGHT>             m_uniformBuffers;
    std::array<InstanceBuffer, MAX_FRAMES_IN_FLIGHT>     m_instanceBuffers;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>    m_globalSets{};

    // Per-swapchain-image semaphores (indexed by imageIndex, NOT by currentFrame).
    // The presentation engine holds renderFinished until the image is re-acquired,
    // so we need one semaphore per image to avoid signalling a still-in-use semaphore.
    std::vector<VkSemaphore> m_renderFinishedSemaphores;

    // -- Scene -----------------------------------------------------------------
    // RenderWorld owns all mesh GPU resources and the object/transform list.
    // Textures are owned separately (Image/Sampler lifetime outlives the world).
    RenderWorld m_world;

    // File-loaded textures (via SceneLoader); destroyed before the allocator.
    std::vector<std::unique_ptr<Texture>> m_sceneTextures;

    // Default 1x1 white texture  --  used when no per-object texture is assigned.
    // Registered with m_world as "default"; Application owns the GPU resources.
    Image           m_defaultImage;
    Sampler         m_defaultSampler;
    VkDescriptorSet m_defaultMaterialSet = VK_NULL_HANDLE;

    // Default 1x1 flat normal map (tangent-space neutral, 128,128,255,255).
    // Written to binding 1 of every material descriptor set that lacks a real normal map.
    Image   m_defaultNormalImage;
    Sampler m_defaultNormalSampler;

    // ObjectIds for demo-scene animated objects (INVALID_HANDLE if using sceneFile).
    ObjectId m_groundId = INVALID_HANDLE;
    ObjectId m_sphereId = INVALID_HANDLE;
    ObjectId m_cubeId   = INVALID_HANDLE;

    Camera       m_camera;
    LightManager m_lights;
    float        m_animTime = 0.0f; // total elapsed seconds, drives object animation

    Config m_cfg; // stored so loadScene() and callbacks can access it

    // Frame state
    uint32_t m_currentFrame       = 0;
    bool     m_framebufferResized = false;
    bool     m_running            = false;
    ShaderSpecializationConstants m_shaderSpecs{};

    // Cached view-projection matrix; computed once in drawFrame(), reused in recordCommands().
    glm::mat4 m_viewProj{1.0f};

    // Key debounce state (rising-edge detection for toggle keys)
    bool m_oPrevPressed = false;

    // E3: Debug visualization overlay mode (stored in GlobalUBO.iblParams.z each frame).
    //   0 = normal rendering
    //   1 = cascade index (red/green/blue/yellow per cascade)
    //   2 = world normals (N * 0.5 + 0.5)
    //   3 = roughness/metallic (R=metallic, G=roughness)
    int  m_debugVizMode = 0;

    // E4: Trigger a "CAPTURE_FRAME" debug label around the next frame's commands.
    // Set by processInput() on F11 rising edge; consumed and cleared in recordCommands().
    bool m_captureNextFrame = false;
    bool m_captureKeyPrev   = false;

    // -- Track I: GPU Crash Resiliency -----------------------------------------

    // I1: Called at any VK_ERROR_DEVICE_LOST detection point.
    // Logs a structured diagnostic (frame, stage, active passes) to stderr,
    // sets m_deviceLost so the run loop exits and shutdown() skips DeviceWaitIdle.
    void reportDeviceLost(VkResult result, const char* stage);

    // Set to true the first time reportDeviceLost() is called.
    bool m_deviceLost = false;

    // Context captured at the moment of loss (for ImGui display and logs).
    struct DeviceLostInfo {
        uint64_t    frame    = 0;        // m_frameCount at time of loss
        std::string stage;               // "vkQueueSubmit", "vkQueuePresentKHR", etc.
        std::string passNames;           // "; "-joined render graph pass names
    };
    DeviceLostInfo m_deviceLostInfo;

    // -- Track F: Pipeline Hot Reload ------------------------------------------

    // F1: Background file watcher + recompiler.
    ShaderWatcher m_shaderWatcher;

    // F2: Pipelines queued for deferred destruction.
    // An old pipeline is destroyed only after MAX_FRAMES_IN_FLIGHT frames have
    // passed so no in-flight command buffer still references it.
    struct DeferredPipeline {
        VkPipeline pipeline;
        uint64_t   destroyAfterFrame; // destroy when m_frameCount >= this value
    };
    std::vector<DeferredPipeline> m_deferredPipelineDestroys;

    // Mouse state (for camera look)
    float  m_mouseSensitivity = 0.1f;
    double m_lastMouseX       = 0.0, m_lastMouseY = 0.0;
    bool   m_firstMouse       = true;
    bool   m_mouseCapture     = false;

    float  m_shadowReceiverBias = 0.00015f;
    // Shadow caster light index. -1 = auto (highest-intensity directional).
    int    m_shadowLightIndex   = -1;
    float  m_lastSkyProbeUpdateTime = -1000.0f;
    float  m_skyProbeUpdateInterval = 0.5f;
};

} // namespace vkt
