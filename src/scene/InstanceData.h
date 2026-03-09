#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace vkt {

// Per-instance data written to the SSBO at set=0, binding=1.
// Must match the InstanceData struct in default.vert and shadow.vert (std430 layout).
// Total size: 64 + 64 + 4 + 12 = 144 bytes.
struct InstanceData {
    glm::mat4 model;         // 64 bytes -- camera-relative world transform (floating origin)
    glm::mat4 normalMatrix;  // 64 bytes -- mat4(mat3(inverseTranspose(model))) for normals
    uint32_t  materialId = 0;    //  4 bytes -- index into GPUMaterial SSBO (set=0, binding=3)
    uint32_t  _pad[3]    = {};   // 12 bytes -- keeps struct 16-byte aligned (std430)
};
static_assert(sizeof(InstanceData) == 144,
    "InstanceData size mismatch -- update InstanceData struct in default.vert and shadow.vert");

} // namespace vkt
