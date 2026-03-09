#include <doctest/doctest.h>

#include "scene/Frustum.h"
#include "scene/BoundingVolume.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Build a simple perspective VP matrix pointing down -Z.
// fov=90 deg, aspect=1, near=1, far=100 in Vulkan depth convention.
static glm::mat4 makeVP() {
    const glm::mat4 proj = []{
        glm::mat4 p = glm::perspective(glm::radians(90.0f), 1.0f, 1.0f, 100.0f);
        p[1][1] *= -1.0f; // Vulkan Y flip
        return p;
    }();
    const glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f,  0.0f),  // eye at origin
        glm::vec3(0.0f, 0.0f, -1.0f),  // looking down -Z
        glm::vec3(0.0f, 1.0f,  0.0f)
    );
    return proj * view;
}

// -- Frustum::fromVP ---------------------------------------------------------

TEST_CASE("Frustum -- fromVP produces 6 planes") {
    const vkt::Frustum f = vkt::Frustum::fromVP(makeVP());
    CHECK(f.planes.size() == 6);
    // Each plane normal must be non-zero after normalisation.
    for (const auto& p : f.planes) {
        const float len = glm::length(glm::vec3(p));
        CHECK(std::abs(len - 1.0f) < 1e-4f);
    }
}

// -- Sphere tests ------------------------------------------------------------

TEST_CASE("Frustum -- sphere directly in front is inside") {
    const vkt::Frustum f = vkt::Frustum::fromVP(makeVP());
    vkt::BoundingVolume vol;
    vol.shape  = vkt::BoundingVolume::Shape::Sphere;
    vol.center = {0.0f, 0.0f, 0.0f};
    vol.radius = 0.5f;

    // Object at (0, 0, -10): well within the frustum (near=1, far=100)
    const glm::mat4 model = glm::translate(glm::mat4{1.0f}, glm::vec3(0.0f, 0.0f, -10.0f));
    CHECK(f.test(vol, model, true));
}

TEST_CASE("Frustum -- sphere far behind the camera is outside") {
    const vkt::Frustum f = vkt::Frustum::fromVP(makeVP());
    vkt::BoundingVolume vol;
    vol.shape  = vkt::BoundingVolume::Shape::Sphere;
    vol.center = {0.0f, 0.0f, 0.0f};
    vol.radius = 0.1f;

    // Object at (0, 0, +50): behind the eye (eye looks down -Z).
    const glm::mat4 model = glm::translate(glm::mat4{1.0f}, glm::vec3(0.0f, 0.0f, 50.0f));
    CHECK(!f.test(vol, model, true));
}

TEST_CASE("Frustum -- sphere beyond far plane is outside") {
    const vkt::Frustum f = vkt::Frustum::fromVP(makeVP());
    vkt::BoundingVolume vol;
    vol.shape  = vkt::BoundingVolume::Shape::Sphere;
    vol.center = {0.0f, 0.0f, 0.0f};
    vol.radius = 0.1f;

    // Object at z=-200: beyond far plane (far=100).
    const glm::mat4 model = glm::translate(glm::mat4{1.0f}, glm::vec3(0.0f, 0.0f, -200.0f));
    CHECK(!f.test(vol, model, true));
}

TEST_CASE("Frustum -- sphere far off to the side is outside") {
    const vkt::Frustum f = vkt::Frustum::fromVP(makeVP());
    vkt::BoundingVolume vol;
    vol.shape  = vkt::BoundingVolume::Shape::Sphere;
    vol.center = {0.0f, 0.0f, 0.0f};
    vol.radius = 0.5f;

    // Object at (1000, 0, -10): far to the right, outside the 90-deg FOV.
    const glm::mat4 model = glm::translate(glm::mat4{1.0f}, glm::vec3(1000.0f, 0.0f, -10.0f));
    CHECK(!f.test(vol, model, true));
}

// -- Shape::None (no culling) -----------------------------------------------

TEST_CASE("Frustum -- Shape::None is never culled") {
    const vkt::Frustum f = vkt::Frustum::fromVP(makeVP());
    vkt::BoundingVolume vol; // default: Shape::None

    // Even at a location that would be outside the frustum.
    const glm::mat4 model = glm::translate(glm::mat4{1.0f}, glm::vec3(9999.0f, 0.0f, 0.0f));
    CHECK(f.test(vol, model, true));
}

// -- Scaled (non-rigid-body) sphere ------------------------------------------

TEST_CASE("Frustum -- scaled object inflates radius correctly") {
    const vkt::Frustum f = vkt::Frustum::fromVP(makeVP());
    vkt::BoundingVolume vol;
    vol.shape  = vkt::BoundingVolume::Shape::Sphere;
    vol.center = {0.0f, 0.0f, 0.0f};
    vol.radius = 0.1f; // tiny object-space radius

    // Scale by 500 in X only -- non-uniform scale, so rigidBody=false required.
    // World-space radius becomes 50 (500 * 0.1), making it visible even off-center.
    const glm::mat4 model =
        glm::translate(glm::mat4{1.0f}, glm::vec3(30.0f, 0.0f, -10.0f)) *
        glm::scale(glm::mat4{1.0f}, glm::vec3(500.0f, 1.0f, 1.0f));

    // rigidBody=false: radius should be inflated to cover the scaled mesh.
    CHECK(f.test(vol, model, false));
    // rigidBody=true (wrong for scaled object): radius stays 0.1f -- may be culled.
    // We don't assert the rigid-body result here because it's intentionally incorrect;
    // just confirm it doesn't crash.
    (void)f.test(vol, model, true);
}
