#pragma once

#include "RenderWorld.h"

#include <glm/glm.hpp>

#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vkt {

// Fixed-step update loop with double-precision time accumulation.
//
// Subclass and override onStep() to implement per-frame or fixed-rate logic.
// Call setTransform() / setWorldPosition() from onStep() -- routing between
// direct world access (non-threaded) and the staged buffer (threaded) is automatic.
//
// NON-THREADED (default)
// ---------------------
//   Call advance(frameDt, world) once per render frame.
//   onStep() runs synchronously on the calling thread.
//   setTransform() / setWorldPosition() write directly to world.
//
// THREADED
// --------
//   Call startThread() once after construction.
//   onStep() runs on the dedicated update thread at the fixed step rate.
//   setTransform() / setWorldPosition() accumulate into a per-step local buffer;
//   one lock is taken at the end of each step to flush to the shared write buffer.
//   Call applyToWorld(world) from the render thread each frame
//   (e.g. in Application::onBeforeDraw) to swap and apply the shared buffer.
//
//   After stopThread(), call applyToWorld() once more to flush the last step's
//   commands before the world falls out of use.
//
// TIME PRECISION
// --------------
//   All time values use double precision (seconds).  Suitable for long-running
//   scientific simulations where float accumulates visible drift within minutes.
class UpdateLoop {
public:
    struct Config {
        // Step size in seconds.  0 = variable (one onStep() per advance() with raw frameDt).
        // For numerical integration use a small fixed value (e.g. 1e-3 for 1 kHz).
        double   fixedStep        = 1.0 / 60.0;
        // Maximum onStep() calls per advance() invocation (non-threaded only).
        // Prevents a spiral of death when the render frame is slow.
        uint32_t maxStepsPerFrame = 8;
        // Maximum staged commands (setTransform + setWorldPosition combined) per step
        // in threaded mode.  Excess commands are silently dropped to bound memory growth
        // under render-thread stalls.  0 = unlimited (use only if object count is bounded
        // by the scene and you trust callers not to flood the queue).
        uint32_t maxPendingCmds   = 65536;
    };

    explicit UpdateLoop(const Config& cfg = {});
    virtual ~UpdateLoop();
    UpdateLoop(const UpdateLoop&)            = delete;
    UpdateLoop& operator=(const UpdateLoop&) = delete;

    // -- Non-threaded: drive from the render thread ----------------------------
    // Accumulates frameDt and calls onStep() one or more times.
    // Throws std::logic_error if the update thread is currently running.
    void advance(double frameDt, RenderWorld& world);

    // -- Threaded: independent update thread -----------------------------------
    // Calls onStep() at the fixed step rate independently of the render rate.
    void startThread();
    // Signals the update thread to stop and joins it.  Blocks until exit.
    // Remaining staged commands are moved to the read buffer -- call applyToWorld()
    // once after this to flush the last step into the world.
    void stopThread();
    // Returns true while threaded mode is active.  Note: may return true for a brief
    // window after the update thread has exited due to an onStep() exception; the flag
    // is cleared by the next applyToWorld() or explicit stopThread() call.
    bool isThreaded() const { return m_threaded.load(std::memory_order_acquire); }

    // Swap staged buffer and apply to world.  Call from the render thread each frame.
    // May also be called once after stopThread() to flush the final step.
    void applyToWorld(RenderWorld& world);

    // Total simulation time elapsed, double precision seconds.
    double simulationTime() const;

    // -- Diagnostics -----------------------------------------------------------
    // Commands dropped since construction or last resetDroppedCommandCount().
    // Incremented when setTransform/setWorldPosition hit the per-step cap, and
    // when flushStepBuffer() finds m_writeBuffer already at the total cap.
    uint64_t droppedCommandCount() const {
        return m_droppedCmds.load(std::memory_order_relaxed);
    }
    void resetDroppedCommandCount() {
        m_droppedCmds.store(0, std::memory_order_relaxed);
    }

protected:
    // Override: implement one simulation step.
    // t  = simulation time at the START of this step (double precision, seconds).
    // dt = step duration (= fixedStep, or frameDt in variable mode).
    //
    // Call setTransform() / setWorldPosition() from here.
    // Both are safe in threaded and non-threaded modes.
    virtual void onStep(double t, double dt) = 0;

    // Write an object transform.  Routes to world directly (non-threaded) or
    // to a thread-local staging buffer (threaded).  Safe only from onStep().
    void setTransform(ObjectId id, const glm::mat4& xf);
    void setWorldPosition(ObjectId id, const glm::dvec3& pos);

private:
    void threadFunc();
    void flushStepBuffer(); // called from update thread: move m_stepBuffer -> m_writeBuffer

    struct Cmd {
        ObjectId   id         = INVALID_HANDLE;
        bool       isWorldPos = false;
        glm::mat4  matrix{1.0f};
        glm::dvec3 pos{};
    };

    Config m_cfg;

    // Non-threaded state
    double       m_time        = 0.0;
    double       m_accumulator = 0.0;
    RenderWorld* m_world       = nullptr; // valid only during advance()

    // Threaded state -- per-step local buffer (written by update thread, no lock)
    std::vector<Cmd> m_stepBuffer;

    // Shared between update thread (writes via flushStepBuffer) and render thread (swap)
    std::vector<Cmd> m_writeBuffer;
    std::vector<Cmd> m_readBuffer;
    std::mutex       m_bufferMutex; // guards m_writeBuffer only; held for one flush per step

    std::thread              m_thread;
    std::atomic<bool>        m_stopFlag{false};
    std::atomic<bool>        m_threaded{false};     // true while threaded mode is active;
                                                    // remains true briefly after thread exits
                                                    // due to exception (until applyToWorld or
                                                    // stopThread finalises it via m_threadFailed)
    std::atomic<bool>        m_threadFailed{false}; // set (release) by threadFunc on exception;
                                                    // applyToWorld() loads (acquire) and rethrows
    mutable std::mutex       m_timeMutex;
    double                   m_threadTime = 0.0;   // guarded by m_timeMutex
    std::exception_ptr       m_threadException{};   // set by threadFunc on onStep() throw
    std::atomic<uint64_t>    m_droppedCmds{0};      // monotone drop counter (relaxed)
};

} // namespace vkt
