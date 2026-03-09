#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace vkt {

void Camera::setPerspective(float fovDegrees, float aspect, float nearPlane, float farPlane) {
    m_fovDegrees = fovDegrees;
    m_aspect     = aspect;
    m_perspNear  = nearPlane;
    m_perspFar   = farPlane;
    m_projMode   = ProjectionMode::Perspective;
    m_nearPlane  = m_perspNear;
    m_farPlane   = m_perspFar;
    updateProjection();
}

void Camera::setOrthographic(float orthoSize, float aspect, float nearPlane, float farPlane) {
    m_orthoSize = orthoSize;
    m_aspect    = aspect;
    m_orthoNear = nearPlane;
    m_orthoFar  = farPlane;
    m_projMode  = ProjectionMode::Orthographic;
    m_nearPlane = m_orthoNear;
    m_farPlane  = m_orthoFar;
    updateProjection();
}

void Camera::setProjectionMode(ProjectionMode mode) {
    if (m_projMode == mode) return;
    m_projMode  = mode;
    // Each mode keeps its own near/far so switching back and forth is lossless.
    if (mode == ProjectionMode::Perspective) {
        m_nearPlane = m_perspNear;
        m_farPlane  = m_perspFar;
    } else {
        m_nearPlane = m_orthoNear;
        m_farPlane  = m_orthoFar;
    }
    updateProjection();
}

void Camera::setNearFar(float nearPlane, float farPlane) {
    m_nearPlane = nearPlane;
    m_farPlane  = farPlane;
    // Also persist into the per-mode slots so mode switches don't reset the value.
    if (m_projMode == ProjectionMode::Perspective) {
        m_perspNear = nearPlane;
        m_perspFar  = farPlane;
    } else {
        m_orthoNear = nearPlane;
        m_orthoFar  = farPlane;
    }
    updateProjection();
}

void Camera::setFov(float degrees) {
    m_fovDegrees = degrees;
    updateProjection();
}

void Camera::setOrthoSize(float size) {
    m_orthoSize = size;
    updateProjection();
}

void Camera::setAspect(float aspect) {
    m_aspect = aspect;
    updateProjection();
}

void Camera::setPosition(const glm::dvec3& pos) {
    m_position = pos;
}

void Camera::setLookAt(const glm::dvec3& target, const glm::dvec3& up) {
    // Compute direction in double to avoid cancellation when target and
    // m_position are both large numbers that differ only in low-order bits.
    const glm::dvec3 toTarget = target - m_position;
    const double toTargetLen2 = glm::dot(toTarget, toTarget);
    if (toTargetLen2 <= 1e-24)
        return; // ignore degenerate look-at requests

    // Cast to float for orientation vectors  --  float precision is sufficient
    // for angles; only the position needs double.
    m_front = glm::normalize(glm::vec3(toTarget));

    glm::vec3 upVec = glm::vec3(up);
    if (glm::dot(upVec, upVec) <= 1e-12f)
        upVec = {0.0f, 1.0f, 0.0f};

    m_right = glm::cross(m_front, upVec);
    if (glm::dot(m_right, m_right) <= 1e-12f) {
        // If up is parallel to front, choose a safe fallback up axis.
        upVec = (std::abs(m_front.y) > 0.99f) ? glm::vec3{0.0f, 0.0f, 1.0f}
                                              : glm::vec3{0.0f, 1.0f, 0.0f};
        m_right = glm::cross(m_front, upVec);
    }
    m_right = glm::normalize(m_right);
    m_up    = glm::normalize(glm::cross(m_right, m_front));

    // Derive yaw/pitch from the new front vector
    m_pitch = glm::degrees(std::asin(m_front.y));
    m_yaw   = glm::degrees(std::atan2(m_front.z, m_front.x));
}

void Camera::setYaw(float yaw) {
    m_yaw = yaw;
    updateVectors();
}

void Camera::setPitch(float pitch) {
    m_pitch = std::clamp(pitch, -89.0f, 89.0f);
    updateVectors();
}

// Movement accumulates in double so position stays precise over large distances.
void Camera::moveForward(float delta) { m_position += glm::dvec3(m_front) * double(delta); }
void Camera::moveRight(float delta)   { m_position += glm::dvec3(m_right) * double(delta); }
void Camera::moveUp(float delta)      { m_position += glm::dvec3(m_up)    * double(delta); }

void Camera::rotate(float dyaw, float dpitch) {
    m_yaw   += dyaw;
    m_pitch  = std::clamp(m_pitch + dpitch, -89.0f, 89.0f);
    updateVectors();
}

glm::mat4 Camera::viewMatrix() const {
    // Floating origin: the view matrix is built as if the camera sits at (0,0,0).
    // Object positions are expressed camera-relative before upload to the GPU,
    // so the translation component of the view matrix is always zero.
    // This keeps all GPU coordinates well within float32 range regardless of
    // where the camera is in the double-precision world.
    return glm::lookAt(glm::vec3(0.0f), glm::vec3(m_front), m_up);
}

glm::vec3 Camera::forward() const { return m_front; }
glm::vec3 Camera::right()   const { return m_right; }

void Camera::updateProjection() {
    if (m_projMode == ProjectionMode::Perspective) {
        m_projection = glm::perspective(
            glm::radians(m_fovDegrees), m_aspect, m_nearPlane, m_farPlane);
    } else {
        // Orthographic: half-height = m_orthoSize, width derived from aspect.
        const float hw = m_orthoSize * m_aspect; // half-width
        m_projection = glm::ortho(-hw, hw, -m_orthoSize, m_orthoSize,
                                  m_nearPlane, m_farPlane);
    }
    // GLM uses OpenGL clip convention (+Y up in NDC, depth -1..1). Vulkan uses
    // +Y down and depth 0..1 (GLM_FORCE_DEPTH_ZERO_TO_ONE handles depth).
    // Negating [1][1] flips the Y axis so geometry is not rendered upside-down.
    m_projection[1][1] *= -1.0f;
}

void Camera::updateVectors() {
    glm::vec3 front;
    front.x = std::cos(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch));
    front.y = std::sin(glm::radians(m_pitch));
    front.z = std::sin(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch));
    m_front = glm::normalize(front);
    m_right = glm::normalize(glm::cross(m_front, glm::vec3(0.0f, 1.0f, 0.0f)));
    m_up    = glm::normalize(glm::cross(m_right, m_front));
}

} // namespace vkt
