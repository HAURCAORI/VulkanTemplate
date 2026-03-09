#include "Mesh.h"

#include <stdexcept>
#include <cmath>
#include <numbers>
#include <cassert>

namespace vkt {

// -- Vertex descriptors ---------------------------------------------------------

VkVertexInputBindingDescription Vertex::bindingDesc() {
    VkVertexInputBindingDescription b{};
    b.binding   = 0;
    b.stride    = sizeof(Vertex);
    b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return b;
}

std::array<VkVertexInputAttributeDescription, 4> Vertex::attributeDescs() {
    std::array<VkVertexInputAttributeDescription, 4> a{};
    a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex, position)};
    a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex, normal)};
    a[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(Vertex, texCoord)};
    a[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, color)};
    return a;
}

// -- Move semantics -------------------------------------------------------------

Mesh::Mesh(Mesh&& o) noexcept
    : m_vertexBuffer(std::move(o.m_vertexBuffer))
    , m_indexBuffer(std::move(o.m_indexBuffer))
    , m_vertexCount(o.m_vertexCount)
    , m_indexCount(o.m_indexCount)
{
    o.m_vertexCount = 0;
    o.m_indexCount  = 0;
}

Mesh& Mesh::operator=(Mesh&& o) noexcept {
    if (this != &o) {
        destroy();
        m_vertexBuffer  = std::move(o.m_vertexBuffer);
        m_indexBuffer   = std::move(o.m_indexBuffer);
        m_vertexCount   = o.m_vertexCount;
        m_indexCount    = o.m_indexCount;
        o.m_vertexCount = 0;
        o.m_indexCount  = 0;
    }
    return *this;
}

// -- Destroy --------------------------------------------------------------------

void Mesh::destroy() {
    m_vertexBuffer.destroy();
    m_indexBuffer.destroy();
    m_vertexCount = 0;
    m_indexCount  = 0;
}

// -- Upload ---------------------------------------------------------------------

void Mesh::upload(VmaAllocator allocator, VkCommandBuffer cmd,
                  const std::vector<Vertex>& vertices,
                  const std::vector<uint32_t>& indices,
                  Buffer& stagingVertexBuf, Buffer& stagingIndexBuf) {
    // cmd must be a valid, recording command buffer.
    // stagingVertexBuf and stagingIndexBuf must outlive this function AND remain
    // alive until the caller submits cmd and the GPU signals completion (fence/idle).
    // Queue family assumption: cmd must belong to the same queue family that will
    // read the vertex/index buffers (usually graphics). If a separate transfer queue
    // is used, an ownership transfer barrier is required after submission.
    assert(cmd != VK_NULL_HANDLE && "Mesh::upload requires a valid recording command buffer");
    assert(allocator != VK_NULL_HANDLE && "Mesh::upload requires a valid VmaAllocator");

    m_vertexCount = static_cast<uint32_t>(vertices.size());
    m_indexCount  = static_cast<uint32_t>(indices.size());

    VkDeviceSize vertexSize = sizeof(Vertex) * vertices.size();
    VkDeviceSize indexSize  = sizeof(uint32_t) * indices.size();

    // GPU-side buffers
    m_vertexBuffer.createGpuOnly(allocator, vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_indexBuffer.createGpuOnly(allocator, indexSize,  VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    // Staging buffers (caller keeps alive until cmd is submitted + completed)
    stagingVertexBuf.createCpuVisible(allocator, vertexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    stagingVertexBuf.uploadData(vertices.data(), vertexSize);

    stagingIndexBuf.createCpuVisible(allocator, indexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    stagingIndexBuf.uploadData(indices.data(), indexSize);

    VkBufferCopy vertexCopy{0, 0, vertexSize};
    VkBufferCopy indexCopy{0, 0, indexSize};
    vkCmdCopyBuffer(cmd, stagingVertexBuf.handle(), m_vertexBuffer.handle(), 1, &vertexCopy);
    vkCmdCopyBuffer(cmd, stagingIndexBuf.handle(),  m_indexBuffer.handle(),  1, &indexCopy);
}

// -- Draw -----------------------------------------------------------------------

void Mesh::bindBuffers(VkCommandBuffer cmd) const {
    VkDeviceSize off = 0;
    VkBuffer     vb  = m_vertexBuffer.handle();
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.handle(), 0, VK_INDEX_TYPE_UINT32);
}

void Mesh::draw(VkCommandBuffer cmd) const {
    VkBuffer     vb  = m_vertexBuffer.handle();
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.handle(), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
}

void Mesh::drawInstanced(VkCommandBuffer cmd,
                         uint32_t instanceCount, uint32_t firstInstance) const {
    VkBuffer     vb  = m_vertexBuffer.handle();
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.handle(), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, m_indexCount, instanceCount, 0, 0, firstInstance);
}

// -- Built-in geometry ----------------------------------------------------------

std::pair<std::vector<Vertex>, std::vector<uint32_t>> Mesh::makeCube() {
    std::vector<Vertex> verts = {
        // Front (+Z)
        {{-0.5f,-0.5f, 0.5f},{0,0,1},{0,1}}, {{ 0.5f,-0.5f, 0.5f},{0,0,1},{1,1}},
        {{ 0.5f, 0.5f, 0.5f},{0,0,1},{1,0}}, {{-0.5f, 0.5f, 0.5f},{0,0,1},{0,0}},
        // Back (-Z)
        {{ 0.5f,-0.5f,-0.5f},{0,0,-1},{0,1}}, {{-0.5f,-0.5f,-0.5f},{0,0,-1},{1,1}},
        {{-0.5f, 0.5f,-0.5f},{0,0,-1},{1,0}}, {{ 0.5f, 0.5f,-0.5f},{0,0,-1},{0,0}},
        // Left (-X)
        {{-0.5f,-0.5f,-0.5f},{-1,0,0},{0,1}}, {{-0.5f,-0.5f, 0.5f},{-1,0,0},{1,1}},
        {{-0.5f, 0.5f, 0.5f},{-1,0,0},{1,0}}, {{-0.5f, 0.5f,-0.5f},{-1,0,0},{0,0}},
        // Right (+X)
        {{ 0.5f,-0.5f, 0.5f},{1,0,0},{0,1}}, {{ 0.5f,-0.5f,-0.5f},{1,0,0},{1,1}},
        {{ 0.5f, 0.5f,-0.5f},{1,0,0},{1,0}}, {{ 0.5f, 0.5f, 0.5f},{1,0,0},{0,0}},
        // Top (+Y)
        {{-0.5f, 0.5f, 0.5f},{0,1,0},{0,1}}, {{ 0.5f, 0.5f, 0.5f},{0,1,0},{1,1}},
        {{ 0.5f, 0.5f,-0.5f},{0,1,0},{1,0}}, {{-0.5f, 0.5f,-0.5f},{0,1,0},{0,0}},
        // Bottom (-Y)
        {{-0.5f,-0.5f,-0.5f},{0,-1,0},{0,1}}, {{ 0.5f,-0.5f,-0.5f},{0,-1,0},{1,1}},
        {{ 0.5f,-0.5f, 0.5f},{0,-1,0},{1,0}}, {{-0.5f,-0.5f, 0.5f},{0,-1,0},{0,0}},
    };

    std::vector<uint32_t> indices;
    for (uint32_t f = 0; f < 6; ++f) {
        uint32_t b = f * 4;
        indices.insert(indices.end(), {b,b+1,b+2, b,b+2,b+3});
    }
    return {verts, indices};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> Mesh::makeQuad() {
    std::vector<Vertex> verts = {
        {{-0.5f,-0.5f, 0.0f},{0,0,1},{0,1}},
        {{ 0.5f,-0.5f, 0.0f},{0,0,1},{1,1}},
        {{ 0.5f, 0.5f, 0.0f},{0,0,1},{1,0}},
        {{-0.5f, 0.5f, 0.0f},{0,0,1},{0,0}},
    };
    std::vector<uint32_t> indices{0,1,2,0,2,3};
    return {verts, indices};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> Mesh::makeUVSphere(
    uint32_t rings, uint32_t sectors)
{
    if (rings   < 2) throw std::invalid_argument("[Mesh] makeUVSphere: rings must be >= 2");
    if (sectors < 3) throw std::invalid_argument("[Mesh] makeUVSphere: sectors must be >= 3");

    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    verts.reserve((rings + 1) * (sectors + 1));
    constexpr float pi = std::numbers::pi_v<float>;

    for (uint32_t r = 0; r <= rings; ++r) {
        float phi = pi * r / rings; // 0..PI
        for (uint32_t s = 0; s <= sectors; ++s) {
            float theta = 2.0f * pi * s / sectors; // 0..2PI
            Vertex v;
            v.normal   = {std::sin(phi) * std::cos(theta),
                          std::cos(phi),
                          std::sin(phi) * std::sin(theta)};
            v.position = v.normal * 0.5f;
            v.texCoord = {static_cast<float>(s) / sectors,
                          static_cast<float>(r) / rings};
            verts.push_back(v);
        }
    }

    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < sectors; ++s) {
            uint32_t a = r * (sectors + 1) + s;
            uint32_t b = a + (sectors + 1);
            indices.insert(indices.end(), {a, b, a+1, b, b+1, a+1});
        }
    }
    return {verts, indices};
}

} // namespace vkt


