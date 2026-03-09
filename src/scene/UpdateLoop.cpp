#include "UpdateLoop.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vkt {

namespace {
using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;
} // namespace

UpdateLoop::UpdateLoop(const Config& cfg) : m_cfg(cfg) {
    m_stepBuffer.reserve(256);
    m_writeBuffer.reserve(256);
    m_readBuffer.reserve(256);
}

UpdateLoop::~UpdateLoop() {
    // stopThread() may rethrow an onStep() exception.  Swallow it: destructors
    // must not throw.  In normal usage the render loop already received this
    // exception via applyToWorld() (which checks m_threadFailed each frame), so
    // swallowing here is only a last-resort safety net, not a silent failure path.
    try { stopThread(); } catch (...) {}
}

// -- Non-threaded ------------------------------------------------------------

void UpdateLoop::advance(double frameDt, RenderWorld& world) {
    if (m_threaded.load(std::memory_order_acquire))
        throw std::logic_error("[UpdateLoop] advance() called while update thread is running");

    m_world = &world;

    // Wrap stepping in try/catch so m_world is always cleared even if onStep() throws.
    // Without this, a stale non-null m_world would cause setTransform() to silently
    // write to a dangling reference on the next advance() call.
    try {
        if (m_cfg.fixedStep <= 0.0) {
            onStep(m_time, frameDt);
            m_time += frameDt;
        } else {
            m_accumulator += frameDt;
            uint32_t steps = 0;
            while (m_accumulator >= m_cfg.fixedStep && steps < m_cfg.maxStepsPerFrame) {
                onStep(m_time, m_cfg.fixedStep);
                m_time        += m_cfg.fixedStep;
                m_accumulator -= m_cfg.fixedStep;
                ++steps;
            }
            // Spiral-of-death guard: if the frame cap was hit the simulation is
            // temporarily behind real time.  fmod preserves the fractional sub-step
            // phase so the accumulator stays bounded [0, fixedStep), bounding the
            // temporal drift to at most maxStepsPerFrame * fixedStep seconds.
            if (steps == m_cfg.maxStepsPerFrame)
                m_accumulator = std::fmod(m_accumulator, m_cfg.fixedStep);
        }
    } catch (...) {
        m_world = nullptr;
        throw;
    }

    m_world = nullptr;
}

// -- Threaded ----------------------------------------------------------------

void UpdateLoop::startThread() {
    if (m_threaded.load(std::memory_order_acquire)) return;
    m_stopFlag        = false;
    m_threadException = nullptr;
    m_threadFailed.store(false, std::memory_order_relaxed);
    m_threadTime      = m_time; // carry over sim time so simulationTime() is continuous
    // m_threaded must be set BEFORE the thread is constructed.  The new thread can reach
    // onStep() -> setTransform() before this function's next line executes; if m_threaded
    // were still false at that point, setTransform() would silently drop the command.
    m_threaded.store(true, std::memory_order_release);
    try {
        m_thread = std::thread(&UpdateLoop::threadFunc, this);
    } catch (...) {
        // Thread construction failed (OS resource exhaustion, etc.).
        // Roll back m_threaded so the object stays usable in non-threaded mode.
        m_threaded.store(false, std::memory_order_release);
        throw;
    }
}

void UpdateLoop::stopThread() {
    if (!m_threaded.load(std::memory_order_acquire)) return;
    m_stopFlag = true;
    m_thread.join();
    // Set flags AFTER join: the thread is confirmed dead, no more setTransform calls.
    m_threadFailed.store(false, std::memory_order_relaxed); // clear before m_threaded=false
    m_threaded.store(false, std::memory_order_release);

    // Carry the thread's simulation clock back into m_time so that simulationTime()
    // is continuous after switching back to non-threaded mode.
    // No lock needed: the thread is confirmed dead, m_threadTime is exclusively ours.
    m_time = m_threadTime;

    // The update thread flushed its last step into m_writeBuffer before exiting.
    // applyToWorld() swaps write <-> read, clears write, then iterates read.
    // So commands must be in m_writeBuffer for applyToWorld() to deliver them.
    // Absorb any residual m_readBuffer commands into m_writeBuffer as well (they
    // could exist if applyToWorld() was not called between the last two steps).
    // Guard with m_bufferMutex: applyToWorld() may still race until this returns.
    {
        std::lock_guard<std::mutex> lock(m_bufferMutex);
        m_writeBuffer.insert(m_writeBuffer.end(), m_readBuffer.begin(), m_readBuffer.end());
        m_readBuffer.clear();
    }

    // Re-throw any exception that threadFunc() captured from onStep().
    // Done AFTER buffer consolidation so the caller still receives the last
    // step's committed commands before the exception propagates.
    if (m_threadException)
        std::rethrow_exception(std::exchange(m_threadException, nullptr));
}

void UpdateLoop::applyToWorld(RenderWorld& world) {
    // Swap write <-> read under a brief lock (one mutex acquisition per frame).
    // The update thread holds m_bufferMutex only once per step (flushStepBuffer),
    // so contention is minimal.  The loop below is O(total commands in the buffer):
    // keep setTransform/setWorldPosition call frequency proportional to object count.
    {
        std::lock_guard<std::mutex> lock(m_bufferMutex);
        std::swap(m_writeBuffer, m_readBuffer);
        m_writeBuffer.clear();
    }
    for (const auto& cmd : m_readBuffer) {
        if (cmd.isWorldPos)
            world.setWorldPosition(cmd.id, cmd.pos);
        else
            world.setTransform(cmd.id, cmd.matrix);
    }
    m_readBuffer.clear();

    // Check for thread failure AFTER applying committed commands so the world
    // receives every successfully-completed step before the exception propagates.
    // At this point readBuffer is empty and writeBuffer was just cleared by the
    // swap, so stopThread()'s buffer consolidation is a no-op; it only joins the
    // dead thread and rethrows.  Acquire load pairs with the release store in
    // threadFunc() to guarantee the exception_ptr write is visible here.
    if (m_threadFailed.load(std::memory_order_acquire))
        stopThread(); // joins dead thread, clears m_threadFailed, rethrows
}

void UpdateLoop::flushStepBuffer() {
    // Called from the update thread at the end of each step.
    // One lock acquisition per step regardless of how many setTransform calls were made.
    // m_stepBuffer is exclusively owned by the update thread, so the clear happens
    // outside the lock to minimise the critical section.
    {
        std::lock_guard<std::mutex> lock(m_bufferMutex);
        if (m_cfg.maxPendingCmds > 0) {
            // Enforce the TOTAL backlog cap: m_writeBuffer can grow across N stalled
            // render frames (each delivering up to maxPendingCmds commands), so the
            // per-step check in setTransform/setWorldPosition is not sufficient alone.
            const size_t cap       = static_cast<size_t>(m_cfg.maxPendingCmds);
            const size_t have      = m_writeBuffer.size();
            const size_t available = (have < cap) ? (cap - have) : 0u;
            const size_t toAppend  = std::min(available, m_stepBuffer.size());
            m_writeBuffer.insert(m_writeBuffer.end(),
                                 m_stepBuffer.begin(),
                                 m_stepBuffer.begin() + static_cast<ptrdiff_t>(toAppend));
            if (toAppend < m_stepBuffer.size())
                m_droppedCmds.fetch_add(m_stepBuffer.size() - toAppend,
                                        std::memory_order_relaxed);
        } else {
            m_writeBuffer.insert(m_writeBuffer.end(), m_stepBuffer.begin(), m_stepBuffer.end());
        }
    }
    m_stepBuffer.clear();
}

void UpdateLoop::threadFunc() {
    const double step = (m_cfg.fixedStep > 0.0) ? m_cfg.fixedStep : (1.0 / 60.0);

    auto nextWakeup = Clock::now();
    while (!m_stopFlag) {
        double t;
        {
            std::lock_guard<std::mutex> lock(m_timeMutex);
            t = m_threadTime;
        }

        try {
            onStep(t, step);   // user writes to m_stepBuffer (no lock)
        } catch (...) {
            // Capture the exception for rethrow in stopThread().
            // Discard the partial step buffer so half-committed commands aren't applied.
            m_stepBuffer.clear();
            m_threadException = std::current_exception();
            // Release store: ensures the exception_ptr write above is visible to
            // any thread that subsequently loads m_threadFailed with acquire ordering.
            m_threadFailed.store(true, std::memory_order_release);
            return; // exit loop; applyToWorld() or stopThread() will join and rethrow
        }
        flushStepBuffer();     // one lock to move m_stepBuffer -> m_writeBuffer

        {
            std::lock_guard<std::mutex> lock(m_timeMutex);
            m_threadTime += step;
        }

        // Sleep until the next step deadline.  If onStep() took longer than
        // one step, nextWakeup is in the past and sleep_until returns immediately
        // so the thread runs as fast as it can to catch up.
        nextWakeup += std::chrono::duration_cast<Clock::duration>(Sec(step));
        std::this_thread::sleep_until(nextWakeup);
    }
}

// -- Shared write helpers ----------------------------------------------------

void UpdateLoop::setTransform(ObjectId id, const glm::mat4& xf) {
    if (m_threaded.load(std::memory_order_acquire)) {
        // No lock: m_stepBuffer is exclusively owned by the update thread during onStep().
        // Enforce the per-step cap to prevent unbounded growth under render stalls.
        if (m_cfg.maxPendingCmds > 0 &&
            m_stepBuffer.size() >= static_cast<size_t>(m_cfg.maxPendingCmds)) {
            m_droppedCmds.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Cmd& cmd      = m_stepBuffer.emplace_back();
        cmd.id         = id;
        cmd.isWorldPos = false;
        cmd.matrix     = xf;
    } else {
        if (m_world) m_world->setTransform(id, xf);
    }
}

void UpdateLoop::setWorldPosition(ObjectId id, const glm::dvec3& pos) {
    if (m_threaded.load(std::memory_order_acquire)) {
        if (m_cfg.maxPendingCmds > 0 &&
            m_stepBuffer.size() >= static_cast<size_t>(m_cfg.maxPendingCmds)) {
            m_droppedCmds.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Cmd& cmd      = m_stepBuffer.emplace_back();
        cmd.id         = id;
        cmd.isWorldPos = true;
        cmd.pos        = pos;
    } else {
        if (m_world) m_world->setWorldPosition(id, pos);
    }
}

double UpdateLoop::simulationTime() const {
    if (m_threaded.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(m_timeMutex);
        return m_threadTime;
    }
    return m_time;
}

} // namespace vkt
