#include <doctest/doctest.h>

#include "scene/UpdateLoop.h"
#include "scene/RenderWorld.h"

#include <glm/glm.hpp>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

// ---------------------------------------------------------------------------
// Concrete subclasses used across test cases
// ---------------------------------------------------------------------------

// Records the number of onStep() calls and the last t/dt values.
struct CountingLoop : vkt::UpdateLoop {
    explicit CountingLoop(const Config& cfg = {}) : vkt::UpdateLoop(cfg) {}
    int    stepCount = 0;
    double lastT     = -1.0;
    double lastDt    = -1.0;
    void onStep(double t, double dt) override {
        lastT  = t;
        lastDt = dt;
        ++stepCount;
    }
};

// Throws std::runtime_error("test exception") on the Nth call to onStep().
struct ThrowingLoop : vkt::UpdateLoop {
    explicit ThrowingLoop(int throwOnStep, const Config& cfg = {})
        : vkt::UpdateLoop(cfg), m_throwOn(throwOnStep) {}
    int m_throwOn;
    int m_count = 0;
    void onStep(double /*t*/, double /*dt*/) override {
        if (++m_count >= m_throwOn)
            throw std::runtime_error("test exception");
    }
};

// Calls setTransform(id, matrix) once per step.
struct SetTransformLoop : vkt::UpdateLoop {
    explicit SetTransformLoop(const Config& cfg = {}) : vkt::UpdateLoop(cfg) {}
    vkt::ObjectId targetId = vkt::INVALID_HANDLE;
    glm::mat4     matrix{2.0f};
    void onStep(double /*t*/, double /*dt*/) override {
        if (targetId != vkt::INVALID_HANDLE)
            setTransform(targetId, matrix);
    }
};

// Calls setTransform() `callsPerStep` times per step (for backpressure tests).
struct FloodingLoop : vkt::UpdateLoop {
    explicit FloodingLoop(int callsPerStep, const Config& cfg = {})
        : vkt::UpdateLoop(cfg), m_callsPerStep(callsPerStep) {}
    int           m_callsPerStep;
    vkt::ObjectId targetId = vkt::INVALID_HANDLE;
    glm::mat4     matrix{1.0f};
    void onStep(double /*t*/, double /*dt*/) override {
        for (int i = 0; i < m_callsPerStep; ++i)
            setTransform(targetId, matrix);
    }
};

// World fixture shared by render-world-requiring tests.
struct LoopWorldFixture {
    vkt::RenderWorld  world;
    vkt::MeshHandle   mh = vkt::INVALID_HANDLE;
    vkt::TextureHandle th = vkt::INVALID_HANDLE;
    vkt::ObjectId     id = vkt::INVALID_HANDLE;

    LoopWorldFixture() {
        mh = world.registerMesh("cube", vkt::Mesh{});
        th = world.registerTexture("stone",
             reinterpret_cast<VkDescriptorSet>(uintptr_t{1}));
        world.registerTexture("default",
             reinterpret_cast<VkDescriptorSet>(uintptr_t{2}));
        id = world.spawn(mh, th);
    }
    ~LoopWorldFixture() { world.destroy(); }
};

// ---------------------------------------------------------------------------
// D1: Non-threaded UpdateLoop tests
// ---------------------------------------------------------------------------

TEST_CASE("UpdateLoop (non-threaded) -- fixed step fires correct number of times") {
    // fixedStep=0.016 sec; advance(0.1) should fire 6 steps
    // (0.1 / 0.016 = 6.25 -> 6 whole steps)
    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep = 0.016;
    CountingLoop loop(cfg);

    vkt::RenderWorld world;
    loop.advance(0.1, world);
    world.destroy();

    CHECK(loop.stepCount == 6);
}

TEST_CASE("UpdateLoop (non-threaded) -- variable step mode calls once per advance") {
    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep = 0.0; // variable
    CountingLoop loop(cfg);

    vkt::RenderWorld world;
    loop.advance(0.033, world);
    loop.advance(0.033, world);
    world.destroy();

    CHECK(loop.stepCount == 2);
    CHECK(std::abs(loop.lastDt - 0.033) < 1e-9);
}

TEST_CASE("UpdateLoop (non-threaded) -- maxStepsPerFrame caps calls per advance") {
    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep        = 0.001; // 1 ms -- would give 100 steps for dt=0.1
    cfg.maxStepsPerFrame = 5;
    CountingLoop loop(cfg);

    vkt::RenderWorld world;
    loop.advance(0.1, world);
    world.destroy();

    // Must not exceed the cap.
    CHECK(loop.stepCount <= static_cast<int>(cfg.maxStepsPerFrame));
}

TEST_CASE("UpdateLoop (non-threaded) -- simulationTime advances per step") {
    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep = 0.1;
    CountingLoop loop(cfg);

    vkt::RenderWorld world;
    loop.advance(0.5, world); // 5 steps exactly
    world.destroy();

    CHECK(loop.stepCount == 5);
    CHECK(std::abs(loop.simulationTime() - 0.5) < 1e-9);
}

TEST_CASE("UpdateLoop (non-threaded) -- simulationTime in variable mode") {
    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep = 0.0;
    CountingLoop loop(cfg);

    vkt::RenderWorld world;
    loop.advance(0.25, world);
    world.destroy();

    CHECK(std::abs(loop.simulationTime() - 0.25) < 1e-9);
}

TEST_CASE("UpdateLoop (non-threaded) -- onStep t matches accumulated time") {
    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep = 0.1;
    CountingLoop loop(cfg);

    vkt::RenderWorld world;
    // Use 0.35 rather than 0.3 to avoid float64 accumulator drift:
    // 0.35 - 0.1 - 0.1 - 0.1 = 0.05 exactly in float64, so all 3 steps fire.
    loop.advance(0.35, world); // 3 steps at t=0, 0.1, 0.2
    world.destroy();

    // Last step starts at t = 0.2 (third step; 0-indexed t: 0, 0.1, 0.2).
    CHECK(loop.stepCount == 3);
    CHECK(std::abs(loop.lastT - 0.2) < 1e-9);
    CHECK(std::abs(loop.lastDt - 0.1) < 1e-9);
}

TEST_CASE("UpdateLoop (non-threaded) -- exception from onStep propagates out of advance") {
    ThrowingLoop loop(1); // throws on the very first step
    vkt::RenderWorld world;
    CHECK_THROWS_AS(loop.advance(1.0, world), std::runtime_error);
    world.destroy();
}

TEST_CASE("UpdateLoop (non-threaded) -- world pointer cleared after exception") {
    // After an exception the internal m_world pointer must be reset to null.
    // If it is not, the next advance() may see a dangling pointer.
    ThrowingLoop loop(2, []() {
        vkt::UpdateLoop::Config c;
        c.fixedStep = 0.1;
        return c;
    }());

    {
        vkt::RenderWorld world1;
        // First advance: step 1 succeeds (count=1), step 2 throws (count=2).
        try { loop.advance(0.5, world1); } catch (...) {}
        world1.destroy();
    }
    // world1 is now destroyed.  If m_world were still pointing to it, the next
    // advance() call would silently write to freed memory.

    // A fresh world -- no crash means the stale pointer was cleared.
    vkt::RenderWorld world2;
    try { loop.advance(0.5, world2); } catch (...) {}
    world2.destroy();
    // Reaching here without crash/UB confirms exception safety.
}

TEST_CASE("UpdateLoop (non-threaded) -- setTransform writes to world during advance") {
    LoopWorldFixture f;
    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep = 0.1;
    SetTransformLoop loop(cfg);
    loop.targetId = f.id;
    loop.matrix   = glm::mat4{3.0f};

    loop.advance(0.1, f.world); // 1 step

    const glm::mat4 got = f.world.getTransform(f.id);
    // getTransform returns rotation+scale from the mat4 (column 3 is rebuilt from dvec3).
    // For a diagonal-3 matrix the upper-3x3 block should be 3*Identity.
    CHECK(std::abs(got[0][0] - 3.0f) < 1e-5f);
    CHECK(std::abs(got[1][1] - 3.0f) < 1e-5f);
    CHECK(std::abs(got[2][2] - 3.0f) < 1e-5f);
}

// ---------------------------------------------------------------------------
// D1: Threaded UpdateLoop tests
// ---------------------------------------------------------------------------

TEST_CASE("UpdateLoop (threaded) -- startThread and stopThread do not deadlock") {
    CountingLoop loop;
    loop.startThread();
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    loop.stopThread(); // must return (not hang)
    CHECK(loop.stepCount > 0);
}

TEST_CASE("UpdateLoop (threaded) -- multiple restart cycles are stable") {
    CountingLoop loop;
    for (int i = 0; i < 3; ++i) {
        loop.startThread();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        loop.stopThread();
    }
    // No deadlock, no crash -- just verify it reached here.
    CHECK(loop.stepCount > 0);
}

TEST_CASE("UpdateLoop (threaded) -- stopThread is no-op when not running") {
    CountingLoop loop;
    loop.stopThread(); // should not throw or hang
    loop.stopThread(); // idempotent
    CHECK(loop.stepCount == 0);
}

TEST_CASE("UpdateLoop (threaded) -- startThread while already running is no-op") {
    CountingLoop loop;
    loop.startThread();
    loop.startThread(); // second call is a no-op
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    loop.stopThread();
    // Should reach here without issue.
}

TEST_CASE("UpdateLoop (threaded) -- exception from onStep is rethrown by applyToWorld") {
    // ThrowingLoop set to throw on the 2nd step so we definitely get past the first.
    ThrowingLoop loop(2);
    loop.startThread();

    // Wait long enough for 2+ steps at 1/60 rate (~33ms each).
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    vkt::RenderWorld world;
    // applyToWorld() checks m_threadFailed and calls stopThread() which rethrows.
    CHECK_THROWS_AS(loop.applyToWorld(world), std::runtime_error);
    world.destroy();
}

TEST_CASE("UpdateLoop (threaded) -- applyToWorld delivers transform commands to world") {
    LoopWorldFixture f;

    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep = 1.0 / 60.0;
    SetTransformLoop loop(cfg);
    loop.targetId = f.id;
    loop.matrix   = glm::mat4{5.0f};

    loop.startThread();
    std::this_thread::sleep_for(std::chrono::milliseconds(80)); // ~4-5 steps
    loop.stopThread();
    loop.applyToWorld(f.world);

    const glm::mat4 got = f.world.getTransform(f.id);
    // Upper-3x3 block should be 5*Identity.
    CHECK(std::abs(got[0][0] - 5.0f) < 1e-4f);
    CHECK(std::abs(got[1][1] - 5.0f) < 1e-4f);
    CHECK(std::abs(got[2][2] - 5.0f) < 1e-4f);
}

TEST_CASE("UpdateLoop (threaded) -- simulationTime returns thread time while running") {
    CountingLoop loop;
    loop.startThread();
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    const double t = loop.simulationTime();
    loop.stopThread();
    // Must have advanced from zero.
    CHECK(t > 0.0);
}

TEST_CASE("UpdateLoop (threaded) -- droppedCommandCount increments when cap is hit") {
    LoopWorldFixture f;

    vkt::UpdateLoop::Config cfg;
    cfg.fixedStep      = 1.0 / 60.0;
    cfg.maxPendingCmds = 4; // extremely small: any step calling >4 times will drop
    FloodingLoop loop(10, cfg); // 10 setTransform calls per step, cap=4
    loop.targetId = f.id;
    loop.resetDroppedCommandCount();

    loop.startThread();
    std::this_thread::sleep_for(std::chrono::milliseconds(80)); // 4-5 steps
    loop.stopThread();

    CHECK(loop.droppedCommandCount() > 0);
}

TEST_CASE("UpdateLoop (threaded) -- simulationTime is continuous after stopThread + restart") {
    CountingLoop loop;

    loop.startThread();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    loop.stopThread();

    const double tAfterStop = loop.simulationTime();

    // Restart: simulationTime should continue from where it left off, not reset.
    loop.startThread();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    loop.stopThread();

    const double tAfterRestart = loop.simulationTime();
    CHECK(tAfterRestart > tAfterStop);
}

TEST_CASE("UpdateLoop (threaded) -- isThreaded reflects running state") {
    CountingLoop loop;
    CHECK(!loop.isThreaded());
    loop.startThread();
    CHECK(loop.isThreaded());
    loop.stopThread();
    CHECK(!loop.isThreaded());
}
