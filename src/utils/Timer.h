#pragma once

// Lightweight high-resolution timer built on std::chrono.
// Header-only, no OS-specific code.
//
// Usage:
//   vkt::Timer t;
//   doWork();
//   std::printf("%.2f ms\n", t.elapsedMs());
//
//   {
//       vkt::TimerScope s("loadAssets");
//       loadAssets();
//   } // automatically prints "[Timer] loadAssets: X.XX ms"

#include <chrono>
#include <cstdio>
#include <string>

namespace vkt {

class Timer {
public:
    Timer() { reset(); }

    void reset() { m_start = Clock::now(); }

    // Elapsed time since construction or last reset()
    float elapsedMs() const {
        return std::chrono::duration<float, std::milli>(Clock::now() - m_start).count();
    }

    float elapsedSeconds() const {
        return std::chrono::duration<float>(Clock::now() - m_start).count();
    }

    // Capture a lap time without resetting the base clock
    float lapMs() {
        float elapsed = elapsedMs();
        m_lastLap = elapsed;
        return elapsed;
    }

    float lastLapMs() const { return m_lastLap; }

private:
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point m_start;
    float             m_lastLap{0.0f};
};

// RAII scope timer: prints the elapsed time when it goes out of scope.
// Defined outside Timer so that Timer is complete when m_timer is declared.
struct TimerScope {
    explicit TimerScope(const char* name) : m_name(name) {}
    ~TimerScope() {
        std::printf("[Timer] %s: %.3f ms\n", m_name, m_timer.elapsedMs());
    }
    TimerScope(const TimerScope&)            = delete;
    TimerScope& operator=(const TimerScope&) = delete;

private:
    const char* m_name;
    Timer       m_timer;
};

} // namespace vkt
