#pragma once

#include <glm/glm.hpp>

namespace vkt {

// Per-draw push constants (must match layout in default.vert).
// 128 bytes = Vulkan's guaranteed minimum push constant size.
struct PushConstants {
    glm::mat4 model;        // 64 bytes
    glm::mat4 normalMatrix; // 64 bytes  --  inverseTranspose(mat3(model))
};

} // namespace vkt
