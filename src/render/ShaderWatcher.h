#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace vkt {

// Background shader file watcher and recompiler (Track F1-F3).
//
// Polls modification times of shader source files on a worker thread.
// When a change is detected and the debounce window has passed, invokes
// glslc to recompile the changed shaders.  Results are communicated to the
// main thread via consumeReady() + lastResult().
//
// Setup:
//   watcher.init(glslcPath, sourceDir, outputDir, {"default.vert", "default.frag"});
//   // Each frame:
//   if (watcher.consumeReady()) rebuildPipeline();
//
// Source discovery:
//   - Set VKT_SHADER_SOURCE_DIR env var to the shaders/ source directory, or
//   - ShaderWatcher will walk up from the exe directory looking for it.
class ShaderWatcher {
public:
    enum class Status { Idle, Watching, Compiling, Ready, Failed };

    struct CompileResult {
        bool        success = false;
        std::string errors;                            // stderr from glslc on failure
        std::string timestamp;                         // human-readable local time string
    };

    ShaderWatcher()  = default;
    ~ShaderWatcher() = default;
    ShaderWatcher(const ShaderWatcher&)            = delete;
    ShaderWatcher& operator=(const ShaderWatcher&) = delete;

    // Start watching the given shader names in sourceDir; compile to outputDir.
    // glslcPath: full path to glslc executable, or "glslc" to search PATH.
    // Returns false and disables watching if glslcPath is not usable or
    // the source directory does not exist.
    bool init(std::string glslcPath,
              std::string sourceDir,
              std::string outputDir,
              std::vector<std::string> shaderNames);

    // Stop watching, join the worker thread. Safe to call without init().
    void destroy();

    // Returns false if init() failed or watching is disabled.
    bool active() const { return m_active; }

    // Current watcher status (thread-safe atomic read).
    Status status() const { return m_status.load(); }

    // Returns true exactly once after a successful compile that has not
    // yet been acknowledged. Clears the flag. Call on the main thread each frame.
    bool consumeReady();

    // Returns the most recent compile result.  Thread-safe (mutex-guarded).
    CompileResult lastResult() const;

    // Paths that are watched (set at init time).
    const std::vector<std::string>& watchedNames() const { return m_shaderNames; }

    // Trigger an immediate recompile regardless of whether files changed.
    // Safe to call from the main thread.
    void requestRecompile();

    // -- Source/glslc discovery helpers ---------------------------------------

    // Walk up from exeDir and nearby paths to find a shaders/ source directory
    // containing .vert or .frag files.  Returns empty string on failure.
    static std::string findSourceDir(const std::string& exeDir);

    // Locate the glslc executable via VULKAN_SDK env var or PATH.
    // Returns "glslc" as a fallback (caller will discover failure at compile time).
    static std::string findGlslc();

private:
    void watcherLoop();
    void compileAll_and_notify(); // calls compileAll(), updates atomics, prints status
    bool compileAll(std::string& errOut);

    std::string              m_glslcPath;
    std::string              m_sourceDir;
    std::string              m_outputDir;
    std::vector<std::string> m_shaderNames;

    bool                     m_active  = false;
    std::thread              m_thread;
    std::atomic<bool>        m_running{false};
    std::atomic<bool>        m_forceRecompile{false};
    std::atomic<Status>      m_status{Status::Idle};
    std::atomic<bool>        m_hasReady{false};

    mutable std::mutex       m_resultMutex;
    CompileResult            m_lastResult;

    // Per-shader last known modification time.
    std::vector<std::filesystem::file_time_type> m_lastMtimes;

    // Time of the most recent source-file change (used for debounce).
    std::optional<std::chrono::steady_clock::time_point> m_lastChangeTime;

    static constexpr std::chrono::milliseconds kPollInterval{300};
    static constexpr std::chrono::milliseconds kDebounce{400};
};

} // namespace vkt
