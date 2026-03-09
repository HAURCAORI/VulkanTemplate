#pragma once

#include <volk.h>
#include <string>
#include <vector>

namespace vkt {

// RAII wrapper around a VkShaderModule loaded from a SPIR-V (.spv) file.
// The module may be destroyed immediately after pipeline creation; the Vulkan
// spec guarantees the pipeline retains a copy of the bytecode.
class Shader {
public:
    Shader() = default;
    ~Shader();
    Shader(const Shader&)            = delete;
    Shader& operator=(const Shader&) = delete;

    void load(VkDevice device, const std::string& path); // path: .spv file
    void destroy();

    VkShaderModule module() const { return m_module; }

    // Convenience: build a VkPipelineShaderStageCreateInfo for this shader
    VkPipelineShaderStageCreateInfo stageInfo(VkShaderStageFlagBits stage,
                                              const char* entryPoint = "main") const;

private:
    VkDevice       m_device = VK_NULL_HANDLE;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

} // namespace vkt
