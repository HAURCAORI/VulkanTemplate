#include <doctest/doctest.h>

// ShadowPass is instantiated without init() for all tests below.
// Only updateCascades() is exercised -- pure GLM math, zero Vulkan calls.
// The GPU stress scenarios (resize, live rendering, validation layer checks)
// require a live Vulkan context and are verified manually by running the app
// with VKT_VALIDATION enabled.

#include "render/ShadowPass.h"
#include "scene/Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a standard camera perspective-projection (Y-flipped for Vulkan).
static glm::mat4 makePerspective(float fovDeg, float aspect,
                                 float nearZ, float farZ) {
    glm::mat4 p = glm::perspective(glm::radians(fovDeg), aspect, nearZ, farZ);
    p[1][1] *= -1.0f; // Vulkan NDC Y-flip
    return p;
}

// Build a plain view matrix pointing along -Z from the origin.
static glm::mat4 makeView() {
    return glm::lookAt(glm::vec3{0.0f, 5.0f, 10.0f},
                       glm::vec3{0.0f, 0.0f, 0.0f},
                       glm::vec3{0.0f, 1.0f, 0.0f});
}

static bool isFiniteMat4(const glm::mat4& m) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            if (!std::isfinite(m[c][r])) return false;
    return true;
}

// ---------------------------------------------------------------------------
// D2: Shadow cascade math tests
// ---------------------------------------------------------------------------

TEST_CASE("ShadowPass -- cascadeCount defaults to 1 without init()") {
    vkt::ShadowPass pass;
    CHECK(pass.cascadeCount() == 1u);
}

TEST_CASE("ShadowPass -- enabled() defaults to true") {
    vkt::ShadowPass pass;
    CHECK(pass.enabled());
}

TEST_CASE("ShadowPass -- setEnabled() toggles the flag") {
    vkt::ShadowPass pass;
    pass.setEnabled(false);
    CHECK(!pass.enabled());
    pass.setEnabled(true);
    CHECK(pass.enabled());
}

TEST_CASE("ShadowPass -- cascadeSplit[0] falls within [near, far]") {
    vkt::ShadowPass pass; // cascadeCount=1, default cfg

    const float near = 0.1f;
    const float far  = 500.0f;
    pass.updateCascades(glm::vec3{-0.5f, -1.0f, -0.5f},
                        makeView(),
                        makePerspective(60.0f, 16.0f / 9.0f, near, far),
                        near, far);

    const float split = pass.cascadeSplits()[0];
    CHECK(split > near);
    CHECK(split <= far);
}

TEST_CASE("ShadowPass -- cascade splits beyond cascadeCount are FLT_MAX") {
    vkt::ShadowPass pass; // cascadeCount=1

    pass.updateCascades(glm::vec3{0.0f, -1.0f, 0.0f},
                        makeView(),
                        makePerspective(60.0f, 1.0f, 0.1f, 100.0f),
                        0.1f, 100.0f);

    // Indices 1, 2, 3 must be FLT_MAX (not written by single-cascade path).
    for (uint32_t i = 1; i < vkt::RenderLimits::kMaxShadowCascades; ++i)
        CHECK(pass.cascadeSplits()[i] == std::numeric_limits<float>::max());
}

TEST_CASE("ShadowPass -- light space matrix is non-identity for standard inputs") {
    vkt::ShadowPass pass;

    pass.updateCascades(glm::vec3{-0.5f, -1.0f, -0.5f},
                        makeView(),
                        makePerspective(60.0f, 16.0f / 9.0f, 0.1f, 500.0f),
                        0.1f, 500.0f);

    const glm::mat4 mat = pass.cascadeMatrices()[0];
    const glm::mat4 identity{1.0f};
    bool differs = false;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            if (std::abs(mat[c][r] - identity[c][r]) > 1e-4f) differs = true;

    CHECK(differs);
}

TEST_CASE("ShadowPass -- light space matrix is finite (no NaN / Inf)") {
    vkt::ShadowPass pass;

    pass.updateCascades(glm::vec3{0.0f, -1.0f, 0.0f},
                        makeView(),
                        makePerspective(60.0f, 1.0f, 0.1f, 100.0f),
                        0.1f, 100.0f);

    CHECK(isFiniteMat4(pass.cascadeMatrices()[0]));
}

TEST_CASE("ShadowPass -- different light directions produce different matrices") {
    vkt::ShadowPass passA, passB;
    const glm::mat4 view = makeView();
    const glm::mat4 proj = makePerspective(60.0f, 1.0f, 0.1f, 100.0f);

    passA.updateCascades(glm::vec3{0.0f, -1.0f, 0.0f}, view, proj, 0.1f, 100.0f);
    passB.updateCascades(glm::vec3{1.0f, -1.0f, 0.0f}, view, proj, 0.1f, 100.0f);

    // Matrices must differ -- the light came from a different direction.
    float maxDiff = 0.0f;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            maxDiff = std::max(maxDiff,
                               std::abs(passA.cascadeMatrices()[0][c][r] -
                                        passB.cascadeMatrices()[0][c][r]));
    CHECK(maxDiff > 1e-3f);
}

TEST_CASE("ShadowPass -- wider frustum produces a larger cascade split") {
    // A camera seeing farther should give a larger cascade split distance.
    vkt::ShadowPass passNear, passFar;
    const glm::mat4 view = makeView();
    const glm::vec3 lightDir{-0.5f, -1.0f, -0.5f};

    passNear.updateCascades(lightDir, view,
                            makePerspective(60.0f, 1.0f, 0.1f, 50.0f),
                            0.1f, 50.0f);
    passFar.updateCascades(lightDir, view,
                           makePerspective(60.0f, 1.0f, 0.1f, 500.0f),
                           0.1f, 500.0f);

    // The far-camera split should be larger than the near-camera split.
    CHECK(passFar.cascadeSplits()[0] > passNear.cascadeSplits()[0]);
}

TEST_CASE("ShadowPass -- vertical light direction uses z-up fallback without crash") {
    vkt::ShadowPass pass;

    // lightDir=(0,-1,0) means the light points straight down.
    // The cross-product with (0,1,0) degenerates; the code should use (0,0,1) fallback.
    pass.updateCascades(glm::vec3{0.0f, -1.0f, 0.0f},
                        makeView(),
                        makePerspective(60.0f, 1.0f, 0.1f, 100.0f),
                        0.1f, 100.0f);

    CHECK(isFiniteMat4(pass.cascadeMatrices()[0]));
}

TEST_CASE("ShadowPass -- updateCascades is idempotent for same inputs") {
    vkt::ShadowPass pass;
    const glm::vec3 ld{-0.5f, -1.0f, -0.5f};
    const glm::mat4 view = makeView();
    const glm::mat4 proj = makePerspective(60.0f, 1.0f, 0.1f, 200.0f);

    pass.updateCascades(ld, view, proj, 0.1f, 200.0f);
    const glm::mat4 first = pass.cascadeMatrices()[0];

    pass.updateCascades(ld, view, proj, 0.1f, 200.0f);
    const glm::mat4 second = pass.cascadeMatrices()[0];

    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            CHECK(std::abs(first[c][r] - second[c][r]) < 1e-4f);
}

// ---------------------------------------------------------------------------
// D2: Camera projection under resize-like aspect ratio changes
// These tests verify that the Camera's projection matrix stays valid after the
// aspect ratio changes that occur during a window resize event.
// ---------------------------------------------------------------------------

TEST_CASE("Camera -- projection remains valid after aspect ratio changes (resize sim)") {
    vkt::Camera cam;
    cam.setPerspective(60.0f, 16.0f / 9.0f, 0.1f, 1000.0f);

    // Simulate repeated resize events with different aspect ratios.
    const float aspects[] = {1.0f, 16.0f/9.0f, 4.0f/3.0f, 21.0f/9.0f, 0.5625f};
    for (float aspect : aspects) {
        cam.setPerspective(cam.fov(), aspect, cam.nearPlane(), cam.farPlane());
        const glm::mat4 p = cam.projectionMatrix();

        // Y must be flipped for Vulkan (Camera always negates [1][1]).
        CHECK(p[1][1] < 0.0f);

        // No NaN or Inf.
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                CHECK(std::isfinite(p[c][r]));
    }
}

TEST_CASE("Camera -- orthographic projection valid after near/far changes") {
    vkt::Camera cam;
    cam.setOrthographic(5.0f, 1.0f, 0.01f, 5000.0f);

    const float pairs[][2] = {{0.01f, 100.0f}, {0.1f, 1000.0f}, {1.0f, 10000.0f}};
    for (const auto& p : pairs) {
        cam.setNearFar(p[0], p[1]);
        const glm::mat4 proj = cam.projectionMatrix();
        CHECK(proj[1][1] < 0.0f);
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                CHECK(std::isfinite(proj[c][r]));
    }
}

TEST_CASE("Camera -- viewMatrix stays finite after extreme position changes") {
    vkt::Camera cam;
    cam.setPerspective(60.0f, 1.0f, 0.1f, 100.0f);

    // Large world positions that simulate a floating-origin scene (solar-system scale).
    const glm::dvec3 positions[] = {
        {0.0, 0.0, 0.0},
        {1.495978707e11, 0.0, 0.0}, // ~1 AU
        {-3.0856e16, 2.0e15, 0.0},  // ~1 pc
    };
    for (const auto& pos : positions) {
        cam.setPosition(pos);
        const glm::mat4 v = cam.viewMatrix();
        // viewMatrix() is rotation-only (floating origin), so should always be finite.
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                CHECK(std::isfinite(v[c][r]));
    }
}
