#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace vkt {

// Per-object input data written by the CPU and consumed by the GPU compute cull
// pass (cull.comp).  One entry per alive scene object per frame.
//
// The GPU cull pass:
//   1. Tests the world-space bounding sphere against the camera frustum.
//   2. Writes visible instances to the InstanceData SSBO (set=0, binding=1) via
//      atomicAdd on the indirect command's instanceCount field.
//
// Must stay in sync with the CullObject struct in shaders/cull.comp (std430).
// Total size: 64 + 64 + 16 + 4*4 = 160 bytes.
struct CullObject {
    glm::mat4 model;           // 64 bytes offset   0: floating-origin camera-relative transform
    glm::mat4 normalMatrix;    // 64 bytes offset  64: corrects normals under non-uniform scale
    glm::vec4 boundingSphere;  // 16 bytes offset 128: xyz=world center, w=radius (0=no cull)
    uint32_t  batchId;         //  4 bytes offset 144: index into VkDrawIndexedIndirectCommand array
    uint32_t  materialId;      //  4 bytes offset 148: index into GPUMaterial SSBO (set=0, binding=3)
    uint32_t  firstInstance;   //  4 bytes offset 152: base offset in InstanceData SSBO for this batch
    uint32_t  maxInstances;    //  4 bytes offset 156: capacity guard for this batch (object count)
}; // 160 bytes

static_assert(sizeof(CullObject) == 160,
    "CullObject size mismatch -- update CullObject struct in shaders/cull.comp");

} // namespace vkt
