#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vkt {

// FPS-style camera using Euler angles (yaw + pitch).
// Supports both perspective and orthographic projection.
// Projection flips the Y axis so +Y is up in Vulkan NDC (see updateProjection).
// Not thread-safe; all methods must be called from the render/input thread.
class Camera {
public:
    enum class ProjectionMode { Perspective, Orthographic };

    Camera() = default;

    // Set up perspective projection and switch to perspective mode.
    void setPerspective(float fovDegrees, float aspect, float nearPlane, float farPlane);

    // Set up orthographic projection and switch to orthographic mode.
    // orthoSize is the half-height of the view volume in world units.
    // Width is derived from orthoSize * aspect.
    void setOrthographic(float orthoSize, float aspect, float nearPlane, float farPlane);

    // World position in double precision. Float literals like {1.0f, 2.0f, 3.0f}
    // are accepted via implicit float->double widening  --  no vec3 overload needed.
    void setPosition(const glm::dvec3& pos);
    void setLookAt(const glm::dvec3& target, const glm::dvec3& up = {0.0, 1.0, 0.0});

    // Euler-angle control (degrees)
    void setYaw(float yaw);
    void setPitch(float pitch); // clamped to +/-89 deg

    void moveForward(float delta);
    void moveRight(float delta);
    void moveUp(float delta);
    void rotate(float dyaw, float dpitch); // delta in degrees

    // Projection parameter adjustments (take effect immediately)
    void setFov(float degrees);                    // perspective only
    void setOrthoSize(float size);                 // orthographic only; size = half-height
    void setAspect(float aspect);                  // both modes
    void setNearFar(float nearPlane, float farPlane); // both modes
    void setProjectionMode(ProjectionMode mode);

    glm::mat4      viewMatrix()       const;
    glm::mat4      projectionMatrix() const { return m_projection; }
    // Double-precision world position (use for simulation/physics code).
    glm::dvec3     worldPosition()    const { return m_position; }
    // Float approximation of world position (for display or legacy use).
    glm::vec3      position()         const { return glm::vec3(m_position); }
    glm::vec3      forward()          const;
    glm::vec3      right()            const;
    ProjectionMode projectionMode()   const { return m_projMode; }
    float          fov()              const { return m_fovDegrees; }
    float          orthoSize()        const { return m_orthoSize; }
    float          nearPlane()        const { return m_nearPlane; }
    float          farPlane()         const { return m_farPlane; }

private:
    void updateVectors();
    void updateProjection();

    // World position stored in double to avoid catastrophic cancellation at
    // large distances (floating origin: objects are rendered relative to this).
    glm::dvec3 m_position{0.0, 0.0, 3.0};
    glm::vec3 m_front{0.0f, 0.0f, -1.0f};
    glm::vec3 m_up{0.0f, 1.0f, 0.0f};
    glm::vec3 m_right{1.0f, 0.0f, 0.0f};

    float m_yaw   = -90.0f;
    float m_pitch = 0.0f;

    float          m_fovDegrees  = 60.0f;
    float          m_orthoSize   = 5.0f;   // half-height in world units
    float          m_aspect      = 16.0f / 9.0f;
    // Perspective and orthographic use independent near/far defaults.
    // Orthographic near is negative so geometry behind the camera eye position remains visible
    // (ortho has no depth perspective). -100 covers typical scene extents without over-clipping.
    float          m_perspNear   = 0.1f;
    float          m_perspFar    = 1000.0f;
    float          m_orthoNear   = -100.0f;
    float          m_orthoFar    = 1000.0f;
    float          m_nearPlane   = 0.1f;     // mirrors the active mode's near
    float          m_farPlane    = 1000.0f;  // mirrors the active mode's far
    ProjectionMode m_projMode    = ProjectionMode::Perspective;

    glm::mat4 m_projection{1.0f};
};

} // namespace vkt
