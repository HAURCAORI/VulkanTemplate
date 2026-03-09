#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include "resources/Buffer.h"

#include <vector>
#include <array>

namespace vkt {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 color{1.0f, 1.0f, 1.0f, 1.0f};

    static VkVertexInputBindingDescription bindingDesc();
    static std::array<VkVertexInputAttributeDescription, 4> attributeDescs();
};

// Owns device-local vertex and index buffers.  Move-only via the Buffer members.
// Staging buffer lifetime: the caller-provided stagingVertexBuf / stagingIndexBuf
// must remain alive until the upload command buffer has finished executing on the GPU.
class Mesh {
public:
    Mesh() = default;
    ~Mesh() { destroy(); }
    Mesh(const Mesh&)            = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&& o) noexcept;
    Mesh& operator=(Mesh&& o) noexcept;

    // Explicit release.  Must be called before the VmaAllocator is destroyed
    // when the Mesh is a class member that outlives the Allocator member.
    void destroy();

    // Records vkCmdCopyBuffer calls into cmd; caller submits and synchronizes.
    void upload(VmaAllocator allocator, VkCommandBuffer cmd,
                const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices,
                Buffer& stagingVertexBuf, Buffer& stagingIndexBuf);

    // Bind vertex and index buffers without issuing a draw call.
    // Use before vkCmdDrawIndexedIndirect (GPU-driven indirect draw).
    void bindBuffers(VkCommandBuffer cmd) const;

    // Bind buffers and issue a single-instance draw call.
    void draw(VkCommandBuffer cmd) const;

    // Bind buffers and issue an instanced draw call.
    // firstInstance maps to gl_InstanceIndex in the shader.
    void drawInstanced(VkCommandBuffer cmd,
                       uint32_t instanceCount, uint32_t firstInstance) const;

    uint32_t indexCount()  const { return m_indexCount; }
    uint32_t vertexCount() const { return m_vertexCount; }

    // -- Built-in geometry helpers ---------------------------------------------
    static std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeCube();
    static std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeQuad();
    static std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeUVSphere(
        uint32_t rings = 16, uint32_t sectors = 32);

private:
    Buffer   m_vertexBuffer;
    Buffer   m_indexBuffer;
    uint32_t m_vertexCount = 0;
    uint32_t m_indexCount  = 0;
};

} // namespace vkt
