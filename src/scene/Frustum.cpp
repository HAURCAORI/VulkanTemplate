#include "Frustum.h"

#include <glm/glm.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace vkt {

Frustum Frustum::fromVP(const glm::mat4& vp) {
    // GLM matrices are column-major: vp[col][row].
    // Extract rows as vec4 for the Gribb-Hartmann plane equations.
    const auto row = [&](int r) {
        return glm::vec4(vp[0][r], vp[1][r], vp[2][r], vp[3][r]);
    };
    const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);

    Frustum f;
    f.planes[0] = r3 + r0; // left:   clip.x >= -clip.w
    f.planes[1] = r3 - r0; // right:  clip.x <=  clip.w
    f.planes[2] = r3 + r1; // bottom: clip.y >= -clip.w
    f.planes[3] = r3 - r1; // top:    clip.y <=  clip.w
    f.planes[4] = r2;       // near:   clip.z >=  0       (Vulkan depth [0, 1])
    f.planes[5] = r3 - r2;  // far:    clip.z <=  clip.w

    // Normalise xyz so dot(plane.xyz, p) is a true signed distance in world units.
    for (auto& p : f.planes) {
        const float len = glm::length(glm::vec3(p));
        assert(len > 1e-6f && "Frustum::fromVP: degenerate view-projection matrix (zero-length plane normal)");
        if (len > 1e-6f) p /= len;
    }

    return f;
}

bool Frustum::testSphere(glm::vec3 worldCenter, float worldRadius) const {
    for (const auto& p : planes)
        if (glm::dot(glm::vec3(p), worldCenter) + p.w < -worldRadius)
            return false; // entirely outside this half-space
    return true;
}

bool Frustum::test(const BoundingVolume& vol,
                   const glm::mat4&      transform,
                   bool                  rigidBody) const {
    switch (vol.shape) {
    case BoundingVolume::Shape::None:
        return true; // no volume  --  never cull

    case BoundingVolume::Shape::Sphere: {
        const glm::vec3 worldCenter =
            glm::vec3(transform * glm::vec4(vol.center, 1.0f));

        // For scaled objects inflate the radius by the largest column length so
        // the sphere still encloses the mesh after scaling.
        // Rigid bodies (rotation + translation only) skip this for performance.
        float worldRadius = vol.radius;
        if (!rigidBody) {
            const float sx = glm::length(glm::vec3(transform[0]));
            const float sy = glm::length(glm::vec3(transform[1]));
            const float sz = glm::length(glm::vec3(transform[2]));
            worldRadius *= std::max({sx, sy, sz});
        }

        return testSphere(worldCenter, worldRadius);
    }

    // Future shapes (AABB, OBB, ...) go here as additional case branches.
    }

    return true; // unreachable; keeps the compiler happy
}

} // namespace vkt
