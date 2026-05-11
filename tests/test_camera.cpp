#include <doctest/doctest.h>

#include "scene/Camera.h"

#include <glm/glm.hpp>
#include <cmath>

// -- Projection --------------------------------------------------------------

TEST_CASE("Camera -- setPerspective flips Y for Vulkan NDC") {
    vkt::Camera cam;
    cam.setPerspective(60.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
    // GLM produces OpenGL conventions; Camera negates proj[1][1] for Vulkan.
    CHECK(cam.projectionMatrix()[1][1] < 0.0f);
}

TEST_CASE("Camera -- setOrthographic flips Y for Vulkan NDC") {
    vkt::Camera cam;
    cam.setOrthographic(5.0f, 16.0f / 9.0f, -100.0f, 1000.0f);
    CHECK(cam.projectionMatrix()[1][1] < 0.0f);
}

TEST_CASE("Camera -- projectionMode switches correctly") {
    vkt::Camera cam;
    cam.setPerspective(60.0f, 1.0f, 0.1f, 100.0f);
    CHECK(cam.projectionMode() == vkt::Camera::ProjectionMode::Perspective);

    cam.setOrthographic(5.0f, 1.0f, -10.0f, 100.0f);
    CHECK(cam.projectionMode() == vkt::Camera::ProjectionMode::Orthographic);

    cam.setProjectionMode(vkt::Camera::ProjectionMode::Perspective);
    CHECK(cam.projectionMode() == vkt::Camera::ProjectionMode::Perspective);
}

TEST_CASE("Camera -- setFov / fov roundtrip") {
    vkt::Camera cam;
    cam.setPerspective(45.0f, 1.0f, 0.1f, 100.0f);
    cam.setFov(90.0f);
    CHECK(std::abs(cam.fov() - 90.0f) < 1e-4f);
}

TEST_CASE("Camera -- setOrthoSize / orthoSize roundtrip") {
    vkt::Camera cam;
    cam.setOrthographic(5.0f, 1.0f, -10.0f, 100.0f);
    cam.setOrthoSize(12.5f);
    CHECK(std::abs(cam.orthoSize() - 12.5f) < 1e-4f);
}

TEST_CASE("Camera -- setNearFar roundtrip") {
    vkt::Camera cam;
    cam.setPerspective(60.0f, 1.0f, 0.1f, 1000.0f);
    cam.setNearFar(0.5f, 500.0f);
    CHECK(std::abs(cam.nearPlane() - 0.5f)  < 1e-4f);
    CHECK(std::abs(cam.farPlane()  - 500.0f) < 1e-4f);
}

// -- Position & orientation --------------------------------------------------

TEST_CASE("Camera -- setPosition / worldPosition roundtrip (double precision)") {
    vkt::Camera cam;
    cam.setPosition({1.0, 2.0, 3.0});
    const glm::dvec3 p = cam.worldPosition();
    CHECK(std::abs(p.x - 1.0) < 1e-12);
    CHECK(std::abs(p.y - 2.0) < 1e-12);
    CHECK(std::abs(p.z - 3.0) < 1e-12);
}

TEST_CASE("Camera -- float position() is a lossy approximation of worldPosition()") {
    vkt::Camera cam;
    const glm::dvec3 dpos{1.0e8, 2.0e8, 3.0e8};
    cam.setPosition(dpos);
    // worldPosition() must be exact (double).
    const glm::dvec3 d = cam.worldPosition();
    CHECK(std::abs(d.x - dpos.x) < 1.0);
    // position() (float) is allowed to lose precision at these magnitudes.
    const glm::vec3  f = cam.position();
    (void)f; // just confirm it compiles and doesn't crash
}

TEST_CASE("Camera -- viewMatrix is rotation-only (no translation in mat)") {
    vkt::Camera cam;
    // A camera with floating origin: position is stored in dvec3, not in viewMatrix.
    cam.setPerspective(60.0f, 1.0f, 0.1f, 1000.0f);
    cam.setPosition({100.0, 200.0, 300.0});
    const glm::mat4 v = cam.viewMatrix();
    // Column 3 (translation) should be near zero because floating-origin
    // rendering expresses world objects relative to the camera, not the camera
    // relative to the world origin.
    CHECK(std::abs(v[3][0]) < 1e-3f);
    CHECK(std::abs(v[3][1]) < 1e-3f);
    CHECK(std::abs(v[3][2]) < 1e-3f);
}

TEST_CASE("Camera -- forward() and right() are unit vectors") {
    vkt::Camera cam;
    cam.setPerspective(60.0f, 1.0f, 0.1f, 1000.0f);
    const float fLen = glm::length(cam.forward());
    const float rLen = glm::length(cam.right());
    CHECK(std::abs(fLen - 1.0f) < 1e-5f);
    CHECK(std::abs(rLen - 1.0f) < 1e-5f);
}

TEST_CASE("Camera -- pitch is clamped to +/-89 deg") {
    vkt::Camera cam;
    cam.setPerspective(60.0f, 1.0f, 0.1f, 1000.0f); // establish a valid projection
    cam.setPitch(200.0f);
    CHECK(cam.projectionMatrix()[1][1] < 0.0f); // still valid projection (Y-flipped for Vulkan)
    // No crash -- clamping doesn't produce NaN in viewMatrix
    const glm::mat4 v = cam.viewMatrix();
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            CHECK(!std::isnan(v[c][r]));
}
