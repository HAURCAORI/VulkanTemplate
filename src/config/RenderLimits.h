#pragma once

namespace vkt {

// Compile-time GPU resource caps.
// kMaxLights:        also update #define MAX_LIGHTS in shaders/default.vert and default.frag.
// kMaxFramesInFlight: 2 = double-buffer (hides one frame of GPU latency).
//                       Raise to 3 only if profiling shows the CPU stalling on fences.
struct RenderLimits {
    static constexpr int      kMaxLights        = 8;
    static constexpr uint32_t kMaxFramesInFlight = 2;
    // Maximum number of instances that can be batched per frame.
    // Increase if scenes exceed this limit; each entry is 144 bytes of SSBO.
    static constexpr uint32_t kMaxInstances      = 65536;
    // Maximum number of materials in the material SSBO (set=0, binding=3).
    // Each entry is 48 bytes; index 0 is always the default white material.
    static constexpr uint32_t kMaxMaterials      = 256;
    // Maximum number of draw batches in the indirect draw buffer (Phase 6).
    // A batch is a unique (mesh, textureSet) pair in the scene.
    // Each entry is 20 bytes (VkDrawIndexedIndirectCommand).
    static constexpr uint32_t kMaxBatches        = 1024;
    // Maximum entries in the bindless sampler2D[] array (Track G).
    // Each entry is one VkDescriptorImageInfo slot in the UPDATE_AFTER_BIND pool.
    static constexpr uint32_t kMaxBindlessTextures = 1024;
    // Compute workgroup size for cull.comp (local_size_x).
    // Must match the layout qualifier in shaders/cull.comp.
    static constexpr uint32_t kCullWorkgroupSize = 64;
    // Maximum directional-light cascades for CSM.
    // Keep <= 4 for good default performance on mid-range GPUs.
    static constexpr uint32_t kMaxShadowCascades = 4;
};

} // namespace vkt
