#pragma once

#include "BoundingVolume.h"

#include <glm/glm.hpp>
#include <array>

namespace vkt {

// View frustum built from a combined Vulkan VP matrix (depth range [0, 1]).
//
// Each of the six planes is stored as vec4(nx, ny, nz, d).
// A world-space point p is inside the frustum when:
//   dot(plane.xyz, p) + plane.w >= 0   for all six planes.
//
// Planes are normalised so the signed distance to a plane equals the dot result.
struct Frustum {
    std::array<glm::vec4, 6> planes;

    // Build a Frustum from a combined view-projection matrix.
    // Uses the Gribb-Hartmann method; works for both perspective and orthographic.
    // vp must use Vulkan depth convention (GLM_FORCE_DEPTH_ZERO_TO_ONE).
    static Frustum fromVP(const glm::mat4& vp);

    // Test a BoundingVolume against this frustum.
    //
    // transform   --  the object's model matrix (world-space position and orientation).
    // rigidBody   --  when true the transform has no non-uniform scale so radius does
    //              not need to be inflated (avoids sqrt on all three column lengths).
    //
    // Returns false only when the volume is guaranteed fully outside the frustum
    // (conservative: may return true for volumes just outside).
    bool test(const BoundingVolume& vol,
              const glm::mat4&      transform,
              bool                  rigidBody = true) const;

private:
    bool testSphere(glm::vec3 worldCenter, float worldRadius) const;
};

} // namespace vkt
