#include "render/ShaderWatcher.h"
#include "utils/Platform.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <sstream>

// Cross-platform popen/pclose shim.
#if defined(_MSC_VER) || defined(_WIN32)
#  include <cstdio>
#  define VKT_POPEN  _popen
#  define VKT_PCLOSE _pclose
#else
#  define VKT_POPEN  popen
#  define VKT_PCLOSE pclose
#endif

namespace vkt {

namespace fs = std::filesystem;

// -- Helpers ------------------------------------------------------------------

static std::string localTimeString() {
    const auto now    = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    char buf[32] = {};
#if defined(_MSC_VER)
    struct tm tm_info;
    localtime_s(&tm_info, &tt);
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm_info);
#else
    struct tm* tm_info = std::localtime(&tt);
    if (tm_info) std::strftime(buf, sizeof(buf), "%H:%M:%S", tm_info);
#endif
    return buf;
}

// Run a shell command via popen and capture its combined stdout+stderr.
// Returns the process exit code (0 = success).
static int runCapture(const std::string& cmd, std::string& output) {
    output.clear();
    FILE* pipe = VKT_POPEN(cmd.c_str(), "r");
    if (!pipe) {
        output = "[ShaderWatcher] Failed to launch process: " + cmd + "\n";
        return -1;
    }
    char buf[256];
    while (std::fgets(buf, sizeof(buf), pipe))
        output += buf;
    return VKT_PCLOSE(pipe);
}

// Quote a path for use in a shell command line.
static std::string quotePath(const fs::path& p) {
    return "\"" + p.string() + "\"";
}

// -- ShaderWatcher::init / destroy -------------------------------------------

bool ShaderWatcher::init(std::string glslcPath,
                         std::string sourceDir,
                         std::string outputDir,
                         std::vector<std::string> shaderNames) {
    m_glslcPath   = std::move(glslcPath);
    m_sourceDir   = std::move(sourceDir);
    m_outputDir   = std::move(outputDir);
    m_shaderNames = std::move(shaderNames);

    if (m_sourceDir.empty() || !fs::exists(m_sourceDir)) {
        std::printf("[ShaderWatcher] Source directory not found: \"%s\".\n"
                    "  Set VKT_SHADER_SOURCE_DIR to the shaders/ source directory.\n",
                    m_sourceDir.c_str());
        return false;
    }

    // Seed modification times so the first poll does not immediately trigger.
    m_lastMtimes.resize(m_shaderNames.size());
    for (size_t i = 0; i < m_shaderNames.size(); ++i) {
        fs::path src = fs::path(m_sourceDir) / m_shaderNames[i];
        if (fs::exists(src))
            m_lastMtimes[i] = fs::last_write_time(src);
    }

    m_active  = true;
    m_running = true;
    m_status  = Status::Watching;
    m_thread  = std::thread(&ShaderWatcher::watcherLoop, this);

    std::printf("[ShaderWatcher] Watching %zu shader(s) in \"%s\".\n",
                m_shaderNames.size(), m_sourceDir.c_str());
    return true;
}

void ShaderWatcher::destroy() {
    m_running = false;
    if (m_thread.joinable())
        m_thread.join();
    m_active = false;
    m_status = Status::Idle;
}

// -- Public API ---------------------------------------------------------------

bool ShaderWatcher::consumeReady() {
    return m_hasReady.exchange(false);
}

ShaderWatcher::CompileResult ShaderWatcher::lastResult() const {
    std::lock_guard<std::mutex> lock(m_resultMutex);
    return m_lastResult;
}

void ShaderWatcher::requestRecompile() {
    m_forceRecompile = true;
}

// -- Watcher loop -------------------------------------------------------------

void ShaderWatcher::watcherLoop() {
    while (m_running.load()) {
        // Forced recompile requested by the user (e.g. via UI button).
        if (m_forceRecompile.exchange(false)) {
            m_lastChangeTime.reset();
            compileAll_and_notify();
            std::this_thread::sleep_for(kPollInterval);
            continue;
        }

        // Poll file modification times.
        bool anyChanged = false;
        for (size_t i = 0; i < m_shaderNames.size(); ++i) {
            const fs::path src = fs::path(m_sourceDir) / m_shaderNames[i];
            if (!fs::exists(src)) continue;

            std::error_code ec;
            const auto mtime = fs::last_write_time(src, ec);
            if (ec) continue;

            if (mtime != m_lastMtimes[i]) {
                m_lastMtimes[i]  = mtime;
                m_lastChangeTime = std::chrono::steady_clock::now();
                anyChanged       = true;
            }
        }

        // Debounce: compile only after the file has been stable for kDebounce.
        if (m_lastChangeTime.has_value()) {
            const auto elapsed = std::chrono::steady_clock::now() - *m_lastChangeTime;
            if (elapsed >= kDebounce) {
                m_lastChangeTime.reset();
                compileAll_and_notify();
            }
        }
        (void)anyChanged; // silence unused warning
        std::this_thread::sleep_for(kPollInterval);
    }
}

// -- Compilation --------------------------------------------------------------

// Proxy that compiles and updates status/result atomics.
// Extracted so it can be called from both the loop and force-recompile paths.
void ShaderWatcher::compileAll_and_notify() {
    m_status = Status::Compiling;

    std::string errors;
    const bool  ok = compileAll(errors);

    CompileResult result;
    result.success   = ok;
    result.errors    = errors;
    result.timestamp = localTimeString();
    {
        std::lock_guard<std::mutex> lock(m_resultMutex);
        m_lastResult = result;
    }

    if (ok) {
        m_hasReady = true;
        m_status   = Status::Ready;
        std::printf("[ShaderWatcher] Compile succeeded at %s.\n", result.timestamp.c_str());
    } else {
        m_status = Status::Failed;
        std::printf("[ShaderWatcher] Compile FAILED at %s:\n%s\n",
                    result.timestamp.c_str(), errors.c_str());
    }
}

bool ShaderWatcher::compileAll(std::string& errOut) {
    errOut.clear();
    bool allOk = true;

    for (const auto& name : m_shaderNames) {
        const fs::path src = fs::path(m_sourceDir) / name;
        const fs::path out = fs::path(m_outputDir) / (name + ".spv");

        if (!fs::exists(src)) {
            errOut += "[ShaderWatcher] Source not found: " + src.string() + "\n";
            allOk = false;
            continue;
        }

        // Build: "glslcPath" src -o out 2>&1
        const std::string cmd =
            quotePath(m_glslcPath) + " " +
            quotePath(src)         + " -o " +
            quotePath(out)         + " 2>&1";

        std::string cmdOut;
        const int   ret = runCapture(cmd, cmdOut);

        if (!cmdOut.empty())
            errOut += name + ":\n" + cmdOut;

        if (ret != 0)
            allOk = false;
    }

    return allOk;
}

// -- Discovery helpers --------------------------------------------------------

std::string ShaderWatcher::findSourceDir(const std::string& exeDir) {
    // 1. Environment variable override.
    const std::string envDir = platform::readEnvVar("VKT_SHADER_SOURCE_DIR");
    if (!envDir.empty() && fs::exists(fs::path(envDir) / "default.vert"))
        return fs::path(envDir).string();

    // 2. Walk up from the exe directory (up to 7 levels) looking for
    //    a "shaders/" subdirectory that contains default.vert.
    fs::path p = fs::path(exeDir);
    for (int depth = 0; depth < 7; ++depth) {
        const fs::path candidate = p / "shaders";
        if (fs::exists(candidate / "default.vert"))
            return candidate.string();
        const fs::path parent = p.parent_path();
        if (parent == p) break; // filesystem root
        p = parent;
    }

    return {};
}

std::string ShaderWatcher::findGlslc() {
    namespace fs = std::filesystem;

    // 1. VULKAN_SDK environment variable.
    const std::string sdk = platform::readEnvVar("VULKAN_SDK");
    if (!sdk.empty()) {
#if defined(_WIN32)
        const fs::path candidate = fs::path(sdk) / "Bin" / "glslc.exe";
#else
        const fs::path candidate = fs::path(sdk) / "bin" / "glslc";
#endif
        if (fs::exists(candidate))
            return candidate.string();
    }

    // 2. Fallback: rely on PATH.  If glslc is not on PATH the compile step
    //    will fail and surface a clear error message.
    return "glslc";
}

} // namespace vkt
