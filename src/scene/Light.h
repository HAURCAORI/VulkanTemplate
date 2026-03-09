#pragma once

#include "config/RenderLimits.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <mutex>

namespace vkt {

inline constexpr int MAX_LIGHTS = RenderLimits::kMaxLights;

// -- GPU-side light data --------------------------------------------------------
// Mirror of the GLSL GPULight struct; must stay in sync with the shader.
// std140 layout: 4 x vec4 = 64 bytes, 16-byte aligned.
struct alignas(16) GPULight {
    glm::vec4 position  {0.0f, 0.0f, 0.0f, 0.0f}; // xyz=pos or dir, w=type (0=dir,1=point,2=spot)
    glm::vec4 color     {1.0f, 1.0f, 1.0f, 1.0f}; // rgb=color, a=intensity
    glm::vec4 direction {0.0f,-1.0f, 0.0f, 10.0f}; // xyz=direction, w=range
    glm::vec4 spotAngles{0.0f, 0.0f, 0.0f, 0.0f}; // x=cos(innerAngle), y=cos(outerAngle), zw=unused
};

// -- C++ light base class -------------------------------------------------------
// CPU-side light parameters. Virtual dispatch only happens via toGPUData()
// once per frame per light, not per fragment.
class Light {
public:
    enum class Type { Directional = 0, Point = 1, Spot = 2 };

    struct CommonState {
        glm::vec3 color{1.0f, 1.0f, 1.0f};
        float     intensity{1.0f};
        bool      enabled{true};
    };

    virtual ~Light() = default;
    virtual Type     type()      const = 0;
    virtual GPULight toGPUData() const = 0;

    CommonState commonState() const {
        std::scoped_lock lock(m_mutex);
        return {m_color, m_intensity, m_enabled};
    }

    void setCommonState(const CommonState& s) {
        std::scoped_lock lock(m_mutex);
        m_color     = s.color;
        m_intensity = s.intensity;
        m_enabled   = s.enabled;
    }

protected:
    mutable std::mutex m_mutex;
    glm::vec3 m_color{1.0f, 1.0f, 1.0f};
    float     m_intensity{1.0f};
    bool      m_enabled{true};
};

// -- Directional light ----------------------------------------------------------
// Infinite-distance light (sun). No attenuation. Direction is world-space.
class DirectionalLight : public Light {
public:
    explicit DirectionalLight(glm::vec3 dir = {-0.5f, -1.0f, -0.5f},
                              glm::vec3 col = {1.0f, 1.0f, 1.0f},
                              float     ity = 1.0f) {
        m_direction = glm::normalize(dir);
        m_color     = col;
        m_intensity = ity;
    }

    Type type() const override { return Type::Directional; }

    void setDirection(const glm::vec3& dir) {
        std::scoped_lock lock(m_mutex);
        m_direction = glm::normalize(dir);
    }

    glm::vec3 direction() const {
        std::scoped_lock lock(m_mutex);
        return m_direction;
    }

    GPULight toGPUData() const override {
        std::scoped_lock lock(m_mutex);
        GPULight g{};
        g.position  = glm::vec4(glm::normalize(m_direction), 0.0f); // w=0 directional
        g.color     = glm::vec4(m_color, m_intensity);
        g.direction = glm::vec4(glm::normalize(m_direction), 0.0f);
        return g;
    }

private:
    glm::vec3 m_direction{0.0f, -1.0f, 0.0f};
};

// -- Point light ----------------------------------------------------------------
// Omnidirectional light with smooth inverse-square falloff within a range.
class PointLight : public Light {
public:
    explicit PointLight(glm::vec3 pos  = {0.0f, 1.0f, 0.0f},
                        glm::vec3 col  = {1.0f, 1.0f, 1.0f},
                        float     ity  = 1.0f,
                        float     rng  = 10.0f) {
        m_position  = pos;
        m_color     = col;
        m_intensity = ity;
        m_range     = rng;
    }

    Type type() const override { return Type::Point; }

    void setPosition(const glm::vec3& pos) {
        std::scoped_lock lock(m_mutex);
        m_position = pos;
    }

    glm::vec3 position() const {
        std::scoped_lock lock(m_mutex);
        return m_position;
    }

    void setRange(float rng) {
        std::scoped_lock lock(m_mutex);
        m_range = rng;
    }

    float range() const {
        std::scoped_lock lock(m_mutex);
        return m_range;
    }

    GPULight toGPUData() const override {
        std::scoped_lock lock(m_mutex);
        GPULight g{};
        g.position  = glm::vec4(m_position, 1.0f); // w=1 point
        g.color     = glm::vec4(m_color, m_intensity);
        g.direction = glm::vec4(0.0f, 0.0f, 0.0f, m_range); // w stores range
        return g;
    }

private:
    glm::vec3 m_position{0.0f, 1.0f, 0.0f};
    float     m_range   {10.0f};
};

// -- Spot light -----------------------------------------------------------------
// Cone-shaped light. innerAngle is the full-bright cone, outerAngle is the falloff edge.
class SpotLight : public Light {
public:
    explicit SpotLight(glm::vec3 pos   = {0.0f, 3.0f, 0.0f},
                       glm::vec3 dir   = {0.0f,-1.0f, 0.0f},
                       glm::vec3 col   = {1.0f, 1.0f, 1.0f},
                       float     ity   = 2.0f,
                       float     rng   = 15.0f,
                       float     inner = 15.0f,
                       float     outer = 30.0f) {
        m_position   = pos;
        m_direction  = glm::normalize(dir);
        m_color      = col;
        m_intensity  = ity;
        m_range      = rng;
        m_innerAngle = inner;
        m_outerAngle = outer;
    }

    Type type() const override { return Type::Spot; }

    void setPosition(const glm::vec3& pos) {
        std::scoped_lock lock(m_mutex);
        m_position = pos;
    }

    void setDirection(const glm::vec3& dir) {
        std::scoped_lock lock(m_mutex);
        m_direction = glm::normalize(dir);
    }

    void setRange(float rng) {
        std::scoped_lock lock(m_mutex);
        m_range = rng;
    }

    void setAngles(float inner, float outer) {
        std::scoped_lock lock(m_mutex);
        m_innerAngle = inner;
        m_outerAngle = outer;
    }

    GPULight toGPUData() const override {
        std::scoped_lock lock(m_mutex);
        GPULight g{};
        g.position   = glm::vec4(m_position, 2.0f); // w=2 spot
        g.color      = glm::vec4(m_color, m_intensity);
        g.direction  = glm::vec4(glm::normalize(m_direction), m_range);
        g.spotAngles = glm::vec4(
            std::cos(glm::radians(m_innerAngle)),
            std::cos(glm::radians(m_outerAngle)),
            0.0f, 0.0f
        );
        return g;
    }

private:
    glm::vec3 m_position   {0.0f, 3.0f, 0.0f};
    glm::vec3 m_direction  {0.0f,-1.0f, 0.0f};
    float     m_range      {15.0f};
    float     m_innerAngle {15.0f};
    float     m_outerAngle {30.0f};
};

// -- LightManager --------------------------------------------------------------
// Thread-safe collection of up to MAX_LIGHTS lights.
// pack() snapshots the shared_ptr list under the manager lock (one heap copy per
// call), then calls toGPUData() outside the lock to prevent lock-order inversion.
// The returned GPULight array is stack-allocated (MAX_LIGHTS entries, fixed size).
class LightManager {
public:
    void add(std::shared_ptr<Light> light) {
        std::scoped_lock lock(m_mutex);
        if (m_lights.size() < static_cast<size_t>(MAX_LIGHTS))
            m_lights.push_back(std::move(light));
    }

    void remove(size_t index) {
        std::scoped_lock lock(m_mutex);
        if (index < m_lights.size())
            m_lights.erase(m_lights.begin() + static_cast<ptrdiff_t>(index));
    }

    void clear() {
        std::scoped_lock lock(m_mutex);
        m_lights.clear();
    }

    int count() const {
        std::scoped_lock lock(m_mutex);
        return static_cast<int>(m_lights.size());
    }

    std::vector<std::shared_ptr<Light>> snapshot() const {
        std::scoped_lock lock(m_mutex);
        return m_lights;
    }

    // Pack enabled lights into a flat GPU array and return the count atomically.
    // Snapshots the list under the manager lock, then calls toGPUData() outside
    // it so the manager lock is never held while acquiring a per-light mutex
    // (prevents lock-order inversion if lights are mutated concurrently).
    std::array<GPULight, MAX_LIGHTS> pack(int* outCount = nullptr) const {
        std::vector<std::shared_ptr<Light>> lights;
        {
            std::scoped_lock lock(m_mutex);
            lights = m_lights;
        }
        std::array<GPULight, MAX_LIGHTS> result{};
        int count = 0;
        for (auto& l : lights) {
            if (!l) continue;
            const auto state = l->commonState();
            if (!state.enabled) continue;
            result[count++] = l->toGPUData();
        }
        if (outCount) *outCount = count;
        return result;
    }

    // Number of enabled lights.
    int enabledCount() const {
        std::vector<std::shared_ptr<Light>> lights;
        {
            std::scoped_lock lock(m_mutex);
            lights = m_lights;
        }
        int n = 0;
        for (auto& l : lights)
            if (l && l->commonState().enabled) ++n;
        return n;
    }

private:
    std::vector<std::shared_ptr<Light>> m_lights;
    mutable std::mutex m_mutex;
};

} // namespace vkt