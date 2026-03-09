# VulkanTemplate

A clean, production-quality C++20 Vulkan starter template for 3D rendering and real-time simulation.
Designed to be readable and ready to fork into a real project.

## Features

- **Vulkan 1.2** via [volk](https://github.com/zeux/volk) dynamic dispatch -- no linking against `vulkan-1`
- **vk-bootstrap** for instance, device, and swapchain creation
- **Vulkan Memory Allocator (VMA)** for all GPU resource allocation
- **MSAA** -- configurable sample count, clamped to device maximum at runtime
- **HDR rendering** -- geometry renders to an R16G16B16A16_SFLOAT offscreen target; tonemapped to swapchain via a fullscreen triangle pass
- **Tonemapping** -- Reinhard, ACES, or clamp; exposure and gamma adjustable at runtime via ImGui
- **Blinn-Phong lighting** -- directional, point, and spot lights (up to 8) with smooth falloff
- **Cascaded Shadow Maps (CSM)** -- up to 4 depth cascades, tight per-cascade orthographic fitting, PSSM log/uniform blend splits, texel-snapping stabilization, 3x3 PCF filtering
- **GPU compute frustum culling** -- per-object bounding sphere test on the GPU; results written to an indirect draw buffer via `atomicAdd`; zero CPU overhead per culled object
- **Indirect drawing** -- `vkCmdDrawIndexedIndirect`; GPU writes `instanceCount` per batch; CPU fills all other fields
- **Material system** -- per-object `GPUMaterial` (base color, roughness, metallic, emissive, unlit flag) stored in a persistently-mapped SSBO; editable at runtime via ImGui with no pipeline rebuild
- **Instance SSBO** -- per-frame `InstanceData` SSBO (model matrix, normal matrix, material ID); supports up to 65 536 instances per frame
- **Floating-origin coordinate system** -- world positions stored as `double`; GPU always receives camera-relative `float` coordinates, eliminating precision loss from mm to light-year scale
- **Distance unit literals** -- `1.0_AU`, `384400.0_km`, `4.244_ly`, ... (`src/utils/Units.h`)
- **Dynamic scene graph** (`RenderWorld`) -- O(1) spawn/despawn with free-list recycling; no GPU cost per object
- **JSON scene loader** -- meshes, textures, and objects from a file; editable without recompiling
- **Fixed-step update loop** (`UpdateLoop`) -- subclassable; double-precision time accumulation; threaded mode with lock-free per-step staging and double-buffered hand-off to the render thread; `simulationTime()` continuous across mode switches
- **FPS limiter** -- sleep+spin hybrid with `timeBeginPeriod(1)` on Windows; accurate to ~1 ms; VSync via FIFO present mode also supported
- **Pipeline cache** -- serialized to `pipeline_cache.bin`, speeds up subsequent launches
- **Vulkan debug labels** -- all passes named for RenderDoc / NSight
- **Dear ImGui** integration with Vulkan + GLFW backends; DPI-aware scaling
- **Thread-safe lights** -- mutex-guarded per-light state for simulation use

---

## Requirements

| Tool | Minimum version |
|------|----------------|
| C++ compiler | C++20 (MSVC 19.29+, GCC 11+, Clang 13+) |
| CMake | 3.20 |
| Vulkan SDK | 1.2 (for `glslc` shader compilation and validation layers) |
| Git | Any recent version (FetchContent uses it) |

> The Vulkan SDK is only required to compile shaders (`glslc`) and to use validation layers.
> At runtime only the Vulkan loader (`vulkan-1.dll` / `libvulkan.so`) is needed.

All C++ library dependencies are fetched automatically by CMake:

| Library | Version | Purpose |
|---------|---------|---------|
| volk | 1.3.270 | Vulkan dynamic dispatch |
| vk-bootstrap | v1.4.344 | Instance / device / swapchain setup |
| VulkanMemoryAllocator | v3.3.0 | GPU memory allocation |
| GLFW | 3.4 | Window + input |
| GLM | 1.0.3 | Math |
| nlohmann/json | v3.11.3 | Scene file parsing |
| Dear ImGui | v1.92.6 | Debug UI |

---

## Getting Started

### 1. Clone

```bash
git clone <your-repo-url>
cd VulkanTemplate
```

### 2. Configure

Using CMake presets (recommended):

```bash
cmake --preset default    # Debug build, validation layers on, output: build/
cmake --preset release    # Release build, output: build-release/
```

Or manually:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

CMake will automatically download all dependencies on first configure.

### 3. Build

```bash
cmake --build --preset default
# or
cmake --build build
```

Shaders are compiled by `glslc` automatically as part of the build.
The executable lands in `build/bin/VulkanTemplate` (or `build-release/bin/` for release).

### 4. Run

```bash
./build/bin/VulkanTemplate
```

The working directory must allow the executable to find `assets/` and the compiled shaders in
`build/shaders_build/`.

---

## Controls

| Input | Action |
|-------|--------|
| **RMB hold** | Capture mouse for camera look |
| **Mouse move** (RMB held) | Rotate camera (yaw / pitch) |
| **Arrow keys** | Rotate camera (no mouse required) |
| **W / S** | Fly forward / backward |
| **A / D** | Strafe left / right |
| **Q / E** | Move down / up |
| **O** | Toggle perspective / orthographic projection |
| **Escape** | Quit |

Camera speed and sensitivity are configurable via `Config` (see below).

---

## Project Structure

```
VulkanTemplate/
+-- src/
|   +-- Application.h/.cpp        Main application class, run loop, per-frame logic
|   +-- main.cpp                  Entry point
|   +-- config/
|   |   +-- RenderLimits.h        Compile-time GPU caps (lights, frames, instances, materials, batches)
|   +-- core/
|   |   +-- Allocator.h/.cpp      VMA allocator wrapper
|   |   +-- Context.h/.cpp        Vulkan instance, physical/logical device, queues
|   |   +-- FrameData.h           Per-frame-in-flight sync primitives
|   |   +-- SwapChain.h/.cpp      Swapchain + depth image; resize callback
|   +-- material/
|   |   +-- Material.h/.cpp       GPUMaterial (48 B std430) + MaterialManager SSBO
|   +-- pipeline/
|   |   +-- Descriptors.h/.cpp    DescriptorLayoutBuilder, DescriptorPool, DescriptorWriter
|   |   +-- PipelineBuilder.h/.cpp Fluent graphics pipeline builder
|   |   +-- PushConstants.h       PushConstants struct (128 bytes)
|   |   +-- RenderPass.h/.cpp     Swapchain render pass + framebuffers
|   |   +-- Shader.h/.cpp         SPIR-V loader, VkShaderModule wrapper
|   +-- render/
|   |   +-- CullPass.h/.cpp       GPU compute frustum cull pipeline + per-frame CullObject SSBOs
|   |   +-- IndirectBuffer.h/.cpp Per-frame VkDrawIndexedIndirectCommand buffer
|   |   +-- PostProcessPass.h/.cpp HDR offscreen pass + fullscreen tonemap blit
|   |   +-- ShadowPass.h/.cpp     CSM depth pre-pass (up to 4 cascades)
|   +-- resources/
|   |   +-- Buffer.h/.cpp         VkBuffer + VMA allocation (GPU-only + CPU-visible)
|   |   +-- Image.h/.cpp          VkImage + view, layout transitions
|   |   +-- Sampler.h/.cpp        VkSampler wrapper with device-limit clamping
|   |   +-- Texture.h/.cpp        Full texture: stb load -> staging -> Image + Sampler
|   +-- scene/
|   |   +-- Camera.h/.cpp         FPS camera, perspective + orthographic projection
|   |   +-- CullObject.h          Per-object GPU cull input (160 B std430)
|   |   +-- Frustum.h/.cpp        View frustum extraction + bounding sphere visibility test
|   |   +-- InstanceBuffer.h/.cpp Persistently-mapped instance SSBO (one per frame-in-flight)
|   |   +-- InstanceData.h        Per-instance GPU data (144 B std430): model, normalMatrix, materialId
|   |   +-- Light.h               DirectionalLight, PointLight, SpotLight, LightManager
|   |   +-- Mesh.h/.cpp           Vertex/index buffers, built-in geometry (cube/quad/sphere)
|   |   +-- RenderWorld.h/.cpp    Dynamic object graph, batch building, indirect draw dispatch
|   |   +-- SceneLoader.h/.cpp    JSON scene file parser
|   |   +-- UpdateLoop.h/.cpp     Fixed-step update loop; threaded double-buffer hand-off
|   +-- ui/
|   |   +-- ImGuiLayer.h/.cpp     ImGui Vulkan+GLFW integration, DPI-aware scaling
|   +-- utils/
|       +-- DebugUtils.h          VK_EXT_debug_utils labels and object names (zero cost in Release)
|       +-- Timer.h               High-resolution timer, RAII scope timer
|       +-- Units.h               Distance unit literals and constants (mm to Mpc, base unit = metres)
+-- shaders/
|   +-- CMakeLists.txt            Compiles all shaders to .spv via glslc
|   +-- cull.comp                 GPU compute frustum culling (local_size_x = 64)
|   +-- default.vert              Geometry vertex shader: reads InstanceData SSBO
|   +-- default.frag              Geometry fragment: Blinn-Phong + CSM shadow PCF + GPUMaterial
|   +-- shadow.vert               Depth-only shadow pass vertex shader
|   +-- fullscreen.vert           Fullscreen triangle (no vertex buffer)
|   +-- post_basic.frag           Tonemap: exposure -> Reinhard/ACES/clamp -> gamma
+-- assets/
|   +-- scene.json                Example scene file
+-- CMakeLists.txt
+-- CMakePresets.json             "default" (Debug) and "release" (Release) presets
```

---

## Render Pipeline Overview

Each frame executes four stages in a single command buffer:

```
1. Shadow pre-pass (ShadowPass)
   buildBatchesNoCull() -- all alive objects, no frustum cull (off-screen casters included)
   For each cascade: render depth-only into cascadeCount x D32_SFLOAT shadow maps
   Pipeline barrier: depth write -> fragment shader read

2. GPU compute cull (CullPass)
   buildCullData() -- fill CullObject SSBO: model matrix + bounding sphere per object
   CPU zeros instanceCount in each indirect draw command
   Pipeline barrier: host/vertex write -> compute read
   Dispatch cull.comp: sphere frustum test; atomicAdd instanceCount; write InstanceData SSBO
   Pipeline barrier: compute write -> vertex shader read + draw indirect read

3. HDR geometry pass (PostProcessPass::beginHdr / endHdr)
   vkCmdDrawIndexedIndirect per batch -- GPU-written instanceCount drives draw count
   Reads: InstanceData SSBO, shadow sampler array, GPUMaterial SSBO, GlobalUBO
   Writes: R16G16B16A16_SFLOAT HDR color + D32_SFLOAT depth

4. Swapchain pass (PostProcessPass::blit + ImGui)
   Fullscreen tonemap triangle (exposure, Reinhard/ACES/clamp, gamma)
   ImGui overlay rendered into the same swapchain render pass
```

---

## Application Configuration

Pass a `Config` struct to `Application::init()`:

```cpp
vkt::Application app;
app.init({
    .title             = "My Game",
    .width             = 1920,
    .height            = 1080,
    .preferredMsaa     = VK_SAMPLE_COUNT_4_BIT,  // clamped to device max
    .validation        = true,                    // Vulkan validation layers
    .mouseSensitivity  = 0.1f,                    // degrees per pixel
    .fullscreen        = false,                   // borderless fullscreen on primary monitor
    .shaderLightCap    = 8,                       // specialization constant: max lights in loop
    .maxDescriptorSets = 1024,                    // descriptor pool capacity
    .cameraSpeed       = 5.0f,                    // world units/second (WASD/QE)
    .keyboardLookSpeed = 90.0f,                   // degrees/second (arrow keys)
    .cameraFov         = 60.0f,                   // vertical FoV in degrees
    .cameraNear        = 0.1f,                    // perspective near plane
    .cameraFar         = 1000.0f,                 // perspective far plane
    .ambientColor      = {0.05f, 0.05f, 0.05f},   // ambient light RGB
    .sceneFile         = "assets/scene.json",     // leave empty for built-in demo scene
    .exposure          = 1.0f,                    // HDR exposure multiplier
    .gamma             = 2.2f,                    // gamma correction exponent
    .tonemapMode       = 0,                       // 0=Reinhard, 1=ACES, 2=Clamp
    .shadowCascades    = 4,                       // CSM cascade count [1..4]
    .shadowResolution  = 2048,                    // per-cascade shadow map size (square)
    .shadowSplitLambda = 0.65f,                   // 0=uniform splits, 1=logarithmic
    .shadowReceiverBias = 0.00015f,               // fragment-shader depth bias (NDC)
    .vsync             = false,                   // FIFO (vsync) vs MAILBOX (uncapped)
    .targetFps         = 120,                     // software FPS cap; 0 = no cap
});
```

All fields have sensible defaults; omit any field you do not need to change.

### Validation layers

1. **Runtime** -- set `Config::validation = true`. Requires the Vulkan SDK.
2. **Compile time** -- `-DVKT_VALIDATION` (automatically added in `Debug` builds via CMake).

---

## Scene File Format

Scene files are JSON. Set `Config::sceneFile` to the path of your file.

```jsonc
{
    "meshes": [
        { "name": "ground",  "type": "quad" },
        { "name": "ball",    "type": "sphere", "rings": 32, "sectors": 64 },
        { "name": "box",     "type": "cube" }
    ],

    "textures": [
        // "path" is relative to the executable's working directory.
        { "name": "stone", "path": "assets/textures/stone.png" }
    ],

    "objects": [
        {
            "mesh":        "ground",
            "texture":     "default",       // omit to use the built-in 1x1 white texture
            "position":    [0, 0, 0],
            "rotation":    [-90, 0, 0],     // Euler degrees, applied Y->X->Z
            "scale":       [8, 8, 1],
            "rigidBody":   false,           // true = no non-uniform scale (cheaper normal matrix)
            "boundRadius": 0.0,             // bounding sphere radius (object space); 0 = no cull
            "boundCenter": [0, 0, 0]        // bounding sphere center (object space)
        },
        {
            "mesh":        "ball",
            "texture":     "stone",
            "position":    [0, 0.5, 0],
            "rigidBody":   true,
            "boundRadius": 0.5
        }
    ]
}
```

**Mesh types:**

| Type | Parameters | Notes |
|------|-----------|-------|
| `"sphere"` | `"rings"` (>= 2, default 16), `"sectors"` (>= 3, default 32) | UV sphere, radius 0.5 |
| `"cube"` | -- | Unit cube, +-0.5 on each axis |
| `"quad"` | -- | Unit quad in XY plane, +-0.5 |

**`rigidBody` flag:** Set `true` when the object only rotates and translates (no non-uniform scale).
The renderer uses `mat3(model)` instead of `inverseTranspose(mat3(model))` for the normal matrix,
which is significantly cheaper. All physics rigid bodies should set this.

---

## RenderWorld -- Dynamic Scene Graph

`RenderWorld` manages all mesh GPU resources and object instances.
Spawn/despawn is O(1) with no GPU allocation per call.

Access it from anywhere via `app.world()`.

**Thread-safety:** `RenderWorld` is not thread-safe. Use a frame-boundary handoff: simulation thread
writes to its own buffer; render thread applies `setTransform()` / `setWorldPosition()` before drawing.

### Registering resources

```cpp
MeshHandle    mh = world.registerMesh("sphere", std::move(myMesh));
TextureHandle th = world.registerTexture("stone", stoneDescriptorSet);

// Look up by name (e.g. after SceneLoader has run)
MeshHandle    mh = world.findMesh("sphere");
TextureHandle th = world.findTexture("stone");
```

### Spawning and despawning objects

```cpp
// Spawn -- O(1), no GPU cost; optionally attach a material
ObjectId id = world.spawn(mh, th, glm::mat4{1.0f});
ObjectId id = world.spawn(mh, th, transform, /*rigidBody=*/true, /*materialId=*/myMatId);

// Despawn -- O(1), slot is recycled for the next spawn
world.despawn(id);

if (world.alive(id)) { ... }
```

### Materials

```cpp
// Register a material -- index 0 is always the default white material
uint32_t matId = app.materials().add("rock", vkt::GPUMaterial{
    .baseColor = {0.5f, 0.4f, 0.3f, 1.0f},
    .roughness = 0.8f,
    .metallic  = 0.0f,
    .emissive  = 0.0f,
    .flags     = 0,
});

// Assign to a spawned object
world.setMaterialId(id, matId);

// Edit at runtime -- changes appear next frame
vkt::GPUMaterial& mat = app.materials().get(matId);
mat.emissive = 2.0f;
app.materials().update(matId, mat);
```

Materials are also editable at runtime via the **Materials** panel in the ImGui overlay.

### Updating transforms

```cpp
world.setTransform(id, glm::translate(glm::mat4{1.0f}, newPos)); // float
world.setWorldPosition(id, glm::dvec3(1.0_AU, 0.0, 0.0));       // double, large-scale
```

---

## Render Limits (`RenderLimits.h`)

All compile-time GPU resource caps live in `src/config/RenderLimits.h`:

| Constant | Default | What it controls |
|----------|---------|-----------------|
| `kMaxLights` | `8` | `GlobalUBO` lights array; shader `#define MAX_LIGHTS` |
| `kMaxFramesInFlight` | `2` | Frame slot count (UBOs, SSBOs, fences); raise to 3 only if profiling shows CPU fence stalls |
| `kMaxInstances` | `65536` | InstanceData SSBO capacity per frame (144 B each) |
| `kMaxMaterials` | `256` | GPUMaterial SSBO capacity (48 B each); index 0 = default white |
| `kMaxBatches` | `1024` | Indirect draw command buffer capacity (20 B each); one per unique (mesh, texture) pair |
| `kCullWorkgroupSize` | `64` | Compute workgroup size; must match `local_size_x` in `cull.comp` |
| `kMaxShadowCascades` | `4` | Maximum CSM cascade count; drives `GlobalUBO` array sizes |

**Changing `kMaxLights`** also requires updating `#define MAX_LIGHTS` in `shaders/default.vert`
and `shaders/default.frag` -- GLSL cannot include C++ headers.

The `Config::shaderLightCap` field further limits the shader loop at runtime via a specialization
constant, without rebuilding:

```cpp
app.init({ .shaderLightCap = 4 }); // loop over at most 4 lights; UBO still holds kMaxLights
```

---

## Camera

```cpp
// Perspective (default)
cam.setPerspective(60.0f, aspectRatio, 0.1f, 1000.0f);

// Orthographic
cam.setOrthographic(5.0f, aspectRatio, -1000.0f, 1000.0f);

// Switch at runtime (O key or ImGui)
cam.setProjectionMode(Camera::ProjectionMode::Orthographic);

// Positioning
cam.setPosition({0.0f, 2.0f, 5.0f});         // float
cam.setPosition(glm::dvec3(1.496e11, 0, 0));  // double, full precision

// Read state
glm::mat4  view  = cam.viewMatrix();      // rotation-only (floating origin)
glm::mat4  proj  = cam.projectionMatrix();
glm::dvec3 world = cam.worldPosition();   // double -- use for simulation
```

The initial camera FOV and near/far planes are configurable via `Config::cameraFov`,
`Config::cameraNear`, and `Config::cameraFar`.

---

## Lighting

```cpp
// Directional (sun-like, no attenuation)
m_lights.add(std::make_shared<vkt::DirectionalLight>(
    glm::vec3(-0.5f, -1.0f, -0.5f),  // direction
    glm::vec3(1.0f, 0.95f, 0.85f),   // color
    1.2f                              // intensity
));

// Point (omnidirectional, distance falloff)
m_lights.add(std::make_shared<vkt::PointLight>(
    glm::vec3(0.0f, 3.0f, 0.0f),     // position
    glm::vec3(0.3f, 0.6f, 1.0f),     // color
    2.0f,                             // intensity
    15.0f                             // range (world units)
));

// Spot (cone-shaped)
m_lights.add(std::make_shared<vkt::SpotLight>(
    glm::vec3(0.0f, 5.0f, 0.0f),     // position
    glm::vec3(0.0f,-1.0f, 0.0f),     // direction
    glm::vec3(1.0f, 1.0f, 1.0f),     // color
    3.0f,                             // intensity
    20.0f,                            // range
    15.0f,                            // inner cone angle (degrees)
    30.0f                             // outer cone angle (degrees)
));
```

The ambient light color is configurable via `Config::ambientColor` (default `{0.05, 0.05, 0.05}`).

---

## UpdateLoop -- Fixed-Step Simulation

`UpdateLoop` (`src/scene/UpdateLoop.h`) decouples simulation logic from the render rate.
It accumulates wall-clock time and calls `onStep()` at a fixed interval with double-precision
time values, which prevents float drift for long-running or high-frequency simulations.

### Basic usage (non-threaded)

Subclass `UpdateLoop` and override `onStep()`. Call `setTransform()` / `setWorldPosition()`
instead of `world.setTransform()` directly -- routing to the world or a staging buffer is
handled automatically based on whether the update thread is running.

```cpp
class OrbitalUpdater : public vkt::UpdateLoop {
public:
    vkt::ObjectId m_planetId = vkt::INVALID_HANDLE;

protected:
    void onStep(double t, double /*dt*/) override {
        constexpr double kOrbitalPeriod = 365.25 * 86400.0; // seconds
        const double angle = 2.0 * 3.14159265358979 * (t / kOrbitalPeriod);
        const double x = 1.0_AU * std::cos(angle);
        const double z = 1.0_AU * std::sin(angle);
        setWorldPosition(m_planetId, glm::dvec3(x, 0.0, z));
    }
};
```

Subclass `Application` and override `onBeforeDraw()`:

```cpp
class MyApp : public vkt::Application {
    OrbitalUpdater m_updater{{.fixedStep = 3600.0}};  // 1-hour steps

    void init(const vkt::Application::Config& cfg = {}) {
        vkt::Application::init(cfg);
        m_updater.m_planetId = world().spawn(...);
    }
protected:
    void onBeforeDraw(double dt) override {
        m_updater.advance(dt, world());  // calls onStep() one or more times
    }
};
```

`advance()` throws `std::logic_error` if called while the update thread is running.

### Threaded usage

When `onStep()` is expensive (numerical ODE solvers, large N-body systems), run it on a
dedicated thread so the render rate is not limited by the simulation rate.

```cpp
class MyApp : public vkt::Application {
    MyUpdater m_updater;

    void init(const vkt::Application::Config& cfg = {}) {
        vkt::Application::init(cfg);
        m_updater.m_planetId = world().spawn(...);
        m_updater.startThread();  // starts immediately; first onStep() is safe
    }

    void shutdown() {
        m_updater.stopThread();        // signals thread, blocks until joined
        m_updater.applyToWorld(world()); // flush the last step's staged commands
        vkt::Application::shutdown();
    }

protected:
    void onBeforeDraw(double /*dt*/) override {
        m_updater.applyToWorld(world()); // swap buffers; ~nanoseconds on render thread
    }
};
```

**Staging pipeline per frame:**

```
Update thread (runs at fixedStep rate):
  onStep()          -- writes setTransform calls into a per-step local buffer (no lock)
  flushStepBuffer() -- one mutex acquisition moves the local buffer into the shared write buffer

Render thread (once per frame, in onBeforeDraw):
  applyToWorld()    -- one mutex acquisition to swap write/read buffers; O(N commands) apply to world
```

The mutex is never held during `onStep()` or during world writes — only for the two bulk
transfers (one per step, one per frame). `applyToWorld()` cost is O(total `setTransform` /
`setWorldPosition` calls since the last frame); keep call frequency proportional to live object count.

**Exception safety:** if `onStep()` throws in non-threaded `advance()`, the internal `m_world`
pointer is reset before rethrowing so no stale writes can occur on the next `advance()` call.

**After `stopThread()`**, call `applyToWorld()` once to deliver the last step's commands.
`stopThread()` consolidates all staged commands into `m_writeBuffer` so the swap inside
`applyToWorld()` promotes them into `m_readBuffer` for delivery — the same path used during
normal operation.

**`simulationTime()` is continuous across mode switches.**  `startThread()` initialises the
threaded clock from the current non-threaded `m_time`, so switching from `advance()` to
`startThread()` / `applyToWorld()` (or back) produces no time jump.

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `fixedStep` | `1/60` s | Step duration in seconds. 0 = variable (one `onStep` per frame with raw `frameDt`) |
| `maxStepsPerFrame` | `8` | Spiral-of-death guard (non-threaded only). When hit, the sub-step fractional remainder is kept via `fmod` so phase is preserved. Temporal drift is bounded to `maxStepsPerFrame × fixedStep` seconds. |
| `maxPendingCmds` | `65536` | Threaded mode only. Maximum staged commands (`setTransform` + `setWorldPosition`) per step. Excess commands are silently dropped to bound memory growth under render-thread stalls. Set to `0` for unlimited. |

Use small `fixedStep` values for high-frequency numerical methods:

```cpp
// 1 kHz physics integration (e.g. Verlet / RK4):
vkt::UpdateLoop::Config cfg;
cfg.fixedStep        = 1e-3;
cfg.maxStepsPerFrame = 32;    // tolerate render frames up to ~30 ms before draining
cfg.maxPendingCmds   = 65536; // 65 k transforms per step is sufficient for most simulations
```

### Time precision

All time values (`t`, `dt`, `simulationTime()`) are `double`.
A float counter accumulating 1/60 s increments has a relative error of ~1 part in 10^7;
after ~3 hours it drifts by ~1 ms per step.  The double accumulator keeps sub-microsecond
accuracy across years of simulated time.

---

## Shadow Maps (CSM)

The shadow system uses **Cascaded Shadow Maps (CSM)** -- up to 4 independent depth cascades,
each fitted tightly around a sub-frustum of the camera view. This eliminates the
depth-precision and coverage trade-off of a single shadow map for large scenes.

### How it works

Each frame:
1. The camera frustum is divided into `shadowCascades` sub-frustums (PSSM log/uniform blend).
2. For each cascade, a tight orthographic light projection is fitted around the sub-frustum corners.
3. Texel-snapping stabilization eliminates shimmer when the camera moves.
4. Depth is rendered into separate `D32_SFLOAT` images per cascade.
5. In `default.frag`, the fragment's view-space depth selects the correct cascade, and a
   3x3 PCF kernel samples from it.

### Shadow caster selection

The shadow pass always uses a directional light. In auto mode (default), the directional light with
the highest intensity is selected automatically. The active caster can be overridden in the ImGui
**Shadow** panel via the "Shadow Caster" combo box.

If no directional light is present, a built-in fallback direction is used.

### ShadowPass::Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `cascadeCount` | `4` | Number of active cascades (clamped to `kMaxShadowCascades`) |
| `resolution` | `2048` | Per-cascade shadow map size in pixels (square) |
| `splitLambda` | `0.65` | Cascade split blend: 0 = uniform, 1 = fully logarithmic |
| `biasConstant` | `0.4` | Constant depth bias (passed to `vkCmdSetDepthBias`) |
| `biasSlope` | `1.2` | Slope-scaled depth bias |
| `depthPadding` | `10.0` | Extra z-range padding in light space to avoid near/far clipping |
| `eyeDistBaseline` | `200.0` | Minimum eye-to-cascade-center distance; prevents coordinate collapse for close cameras |
| `stabilize` | `true` | Snap projection to texel grid to reduce edge shimmer |

These are passed via `Config::shadowCascades`, `Config::shadowResolution`, and
`Config::shadowSplitLambda`. The remaining fields use `ShadowPass::Config` defaults.

### Config fields for shadows

| `Config` field | Default | Description |
|----------------|---------|-------------|
| `shadowCascades` | `4` | Active cascade count |
| `shadowResolution` | `2048` | Per-cascade map size |
| `shadowSplitLambda` | `0.65` | Cascade split blend |
| `shadowReceiverBias` | `0.00015` | Fragment depth bias (NDC); adjustable at runtime in ImGui |

### Enabling / disabling at runtime

```cpp
m_shadowPass.setEnabled(false);  // skips the depth pre-pass; sets shadowEnabled=0 in GlobalUBO
m_shadowPass.setEnabled(true);
```

The toggle is also in the **Shadow** panel of the ImGui overlay.

### Tuning for scene scale

Cascade splits are computed automatically from the camera near/far range. For very large scenes
increase `Config::cameraFar` and optionally increase `shadowResolution` to maintain texel density:

```cpp
app.init({
    .cameraFar         = 5000.0f,
    .shadowResolution  = 4096,
    .shadowSplitLambda = 0.7f,   // more logarithmic -- compress detail near the camera
});
```

---

## FPS Limiting

```cpp
app.init({
    .vsync     = false,  // MAILBOX (uncapped, low latency) -- default
    // .vsync  = true,   // FIFO (driver-level vsync, zero CPU overhead)
    .targetFps = 120,    // software cap; active only when vsync=false; 0=no cap
});
```

When `vsync = false` and `targetFps > 0`, the run loop uses a sleep+spin hybrid:
- Sleep for `(remaining - 2 ms)` using `sleep_for` (OS-scheduled).
- Spin with `yield()` for the last ~2 ms for accuracy.
- On Windows, `timeBeginPeriod(1)` is called for the duration of `run()` to improve sleep
  granularity from ~15 ms to ~1 ms.

While the window is minimized the loop throttles to ~30 FPS regardless of other settings.

---

## Large-scale / Scientific Simulation

The template uses a **floating-origin** coordinate system.

| Layer | Type | Value |
|-------|------|-------|
| World position (CPU) | `glm::dvec3` | Actual position in metres |
| Camera position (CPU) | `glm::dvec3` | Actual position in metres |
| Object position (GPU) | `glm::vec3` | `worldPos - cameraWorldPos`, cast to float |
| View matrix | `glm::mat4` | Rotation only -- translation is always zero |

Every frame `buildCullData()` and `buildBatchesNoCull()` subtract the camera's double-precision
world position from each object before sending float coordinates to the GPU.

```cpp
#include "utils/Units.h"
using namespace vkt::units;

world.setWorldPosition(earthId,   glm::dvec3(1.0_AU,   0.0, 0.0));
world.setWorldPosition(moonId,    glm::dvec3(1.0_AU + 384400.0_km, 0.0, 0.0));
world.setWorldPosition(proxima,   glm::dvec3(4.244_ly, 0.0, 0.0));

camera.setPosition(glm::dvec3(1.1_AU, 0.0, 0.0));
```

**Distance unit literals** (`src/utils/Units.h`):

| Literal | SI value |
|---------|---------|
| `_mm` | 1e-3 m |
| `_cm` | 1e-2 m |
| `_m`  | 1 m |
| `_km` | 1e3 m |
| `_Mm` | 1e6 m |
| `_AU` | 1.496e11 m |
| `_ly` | 9.461e15 m |
| `_pc` | 3.086e16 m |
| `_kpc`| 3.086e19 m |
| `_Mpc`| 3.086e22 m |

---

## Development Guide

### Adding a new mesh type

1. Add a static factory to `Mesh` in `src/scene/Mesh.h`:
   ```cpp
   static std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeCylinder(uint32_t sectors = 32);
   ```
2. Implement it in `Mesh.cpp`.
3. Optionally add a `"cylinder"` branch to `SceneLoader.cpp`.

### Adding a new texture

```cpp
auto tex = std::make_unique<vkt::Texture>();
vkt::Buffer staging;
tex->loadFromFile(allocator, device, physDevice, uploadCmd, staging, "assets/textures/rock.png");
// keep staging alive until the command buffer finishes

const VkDescriptorSet set = descriptorPool.allocate(materialLayout);
vkt::DescriptorWriter{}
    .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                tex->view(), tex->sampler(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    .update(device, set);

world.registerTexture("rock", set);
```

### Adding a new shader

1. Add your `.vert` / `.frag` / `.comp` file to `shaders/`.
2. Add the path to `SHADER_SOURCES` in `shaders/CMakeLists.txt`.
3. Load the compiled `.spv` with `Shader::load()` and build a pipeline with `PipelineBuilder`.

### Modifying the GlobalUBO

`GlobalUBO` in `Application.h` must stay in sync with the uniform block in both shaders.
Layout is `std140`. Current size: **992 bytes** (with `kMaxShadowCascades = 4`).

When adding fields:
- Add to the C++ struct in `Application.h`
- Mirror the field in `default.vert` and `default.frag`
- Maintain 16-byte alignment; add `_pad` fields where needed
- Update the `static_assert(sizeof(GlobalUBO) == ...)` line

> **std140 alignment pitfall:** `vec3` in GLSL std140 aligns to 16 bytes, not 12. Use three
> separate `float` fields in the GLSL uniform block where the C++ struct uses `float[3]`.
> See the `_pad0/_pad1/_pad2` fields after `lightCount` as an example.

### Modifying the vertex format

`Vertex::bindingDesc()` and `Vertex::attributeDescs()` in `Mesh.h` must match the
`layout(location = N)` inputs in `default.vert`. Edit both together.

---

## Shader Development

### Shader search path

At runtime the executable searches for `.spv` files in:

1. `VKT_SHADER_DIR_DEFAULT` -- injected by CMake at build time (`build/shaders_build/`)
2. `VKT_SHADER_DIR` -- environment variable override
3. `shaders/`, `../shaders/`, `bin/shaders/`, `../bin/shaders/`

### MAX_LIGHTS in shaders

```glsl
#define MAX_LIGHTS 8
layout(constant_id = 0) const int LIGHT_CAP = MAX_LIGHTS;
```

Keep `MAX_LIGHTS` in sync with `RenderLimits::kMaxLights`.
`LIGHT_CAP` is set from `Config::shaderLightCap` at pipeline creation -- no rebuild needed to change it.

---

## Debugging

### Validation layers

Enable with `Config::validation = true` or build in `Debug` mode.
Errors are prefixed with `VALIDATION ERROR` in the console.

### RenderDoc / NSight

All passes are named via `VK_EXT_debug_utils`. In RenderDoc's Event Browser:

```
Shadow     (orange)
GPU Cull   (green)
HDR Geometry
Blit + UI
```

```cpp
vkt::debug::setObjectName(device, VK_OBJECT_TYPE_BUFFER, buffer, "My Buffer");

{
    vkt::debug::LabelScope scope(cmd, "My Pass", {0.2f, 0.6f, 1.0f, 1.0f});
    // draw calls here
}
```

---

## Known Limitations

- **Light positions are `float`-precision.** `PointLight` and `SpotLight` store world position as `vec3`. For lights at astronomical distances, maintain a `dvec3` in simulation state and push the camera-relative offset each frame.
- **No asset hot-reload.** Changing `scene.json` requires a restart.
- **Single graphics queue for uploads.** Asset upload is submitted on the graphics queue. Moving uploads to the dedicated transfer queue (`Context::transferQueue()`) requires explicit queue-family ownership transfer barriers (`VK_SHARING_MODE_EXCLUSIVE`) before the graphics stages read those resources.
- **Forward rendering only.** One geometry pass per frame. Deferred rendering would require a G-buffer pass and a lighting pass.
- **Directional shadows only.** Point and spot lights do not cast shadows (would require cube-map or atlas shadow maps).
- **`UpdateLoop::maxPendingCmds` silently drops excess.** In threaded mode, commands beyond the cap are dropped without notification. If your simulation updates more objects per step than this limit, raise the cap or batch updates differently.

---

## License

This template is provided as a starting point for your own projects.
Replace this section with your chosen license.
