#include "Shader.h"

#include <fstream>
#include <stdexcept>

namespace vkt {

static std::vector<uint32_t> readSpv(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("[Shader] Cannot open file: " + path);

    const size_t byteSize = static_cast<size_t>(file.tellg());
    // SPIR-V words are 4 bytes each; a misaligned file is corrupt or not SPIR-V.
    if (byteSize % 4 != 0)
        throw std::runtime_error("[Shader] Invalid SPIR-V (size not 4-byte aligned): " + path);

    // Store as uint32_t so pCode can be assigned without a reinterpret_cast and
    // alignment is guaranteed (vector<char> is only 1-byte aligned).
    std::vector<uint32_t> buf(byteSize / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(byteSize));
    return buf;
}

void Shader::load(VkDevice device, const std::string& path) {
    m_device = device;
    auto code = readSpv(path);

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size() * sizeof(uint32_t); // codeSize is in bytes
    info.pCode    = code.data();

    if (vkCreateShaderModule(device, &info, nullptr, &m_module) != VK_SUCCESS)
        throw std::runtime_error("[Shader] Failed to create shader module from: " + path);
}

void Shader::destroy() {
    if (m_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, m_module, nullptr);
        m_module = VK_NULL_HANDLE;
    }
}

Shader::~Shader() { destroy(); }

VkPipelineShaderStageCreateInfo Shader::stageInfo(VkShaderStageFlagBits stage,
                                                   const char* entryPoint) const {
    VkPipelineShaderStageCreateInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage  = stage;
    info.module = m_module;
    info.pName  = entryPoint;
    return info;
}

} // namespace vkt
