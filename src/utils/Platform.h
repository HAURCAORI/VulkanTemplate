#pragma once

#include <filesystem>
#include <string>
#include <cstdlib>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <unistd.h>
#endif

namespace vkt::platform {

// Returns the value of an environment variable, or empty string if not set.
inline std::string readEnvVar(const char* name) {
#if defined(_MSC_VER)
    char*  buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || buf == nullptr)
        return {};
    std::string out(buf);
    std::free(buf);
    return out;
#else
    const char* val = std::getenv(name);
    return val ? val : "";
#endif
}

// Returns the directory containing the running executable.
inline std::filesystem::path getExeDir() {
#if defined(_WIN32)
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
#elif defined(__linux__)
    char buf[4096];
    const ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        return std::filesystem::path(buf).parent_path();
    }
    return std::filesystem::current_path();
#else
    return std::filesystem::current_path();
#endif
}

} // namespace vkt::platform
