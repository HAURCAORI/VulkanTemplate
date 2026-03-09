#pragma once

#include <glm/glm.hpp>

namespace vkt {

// Describes the spatial extent of an object for visibility culling.
//
// Currently supports sphere culling only.
// To add AABB or OBB culling:
//   1. Add a new Shape enumerator.
//   2. Add the corresponding data fields (e.g. glm::vec3 halfExtents for AABB).
//   3. Implement the test in Frustum::test().
//   4. Call setBoundingAABB() (or equivalent) from the object setup code.
//
// The volume is expressed in object space. Frustum::test() transforms it to
// world space using the object's model matrix before testing.
struct BoundingVolume {
    enum class Shape {
        None,    // no culling  --  object is always submitted for drawing
        Sphere,  // conservative sphere; never under-culls but may miss tight fits
    };

    Shape     shape  = Shape::None;
    glm::vec3 center {0.0f};  // object-space center of the volume
    float     radius = 0.0f;  // sphere: object-space radius
};

} // namespace vkt
