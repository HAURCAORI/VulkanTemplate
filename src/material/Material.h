#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace vkt {

// GPU-side material data (std430, 48 bytes).
// Written into the material SSBO at set=0, binding=3.
// The fragment shader indexes by InstanceData::materialId.
//
// baseColor, roughness, metallic, and emissive are consumed directly by the
// fragment shader. Texture bindings are still provided via set 1 descriptors;
// the reserved slots at the end of the struct are left for a future indexed /
// bindless material path.
struct GPUMaterial {
    glm::vec4 baseColor {1.0f, 1.0f, 1.0f, 1.0f};  // 16 bytes offset  0: rgba tint
    float     roughness = 0.5f;   //  4 bytes  offset 16: [0,1]
    float     metallic  = 0.0f;   //  4 bytes  offset 20: [0,1]
    float     emissive  = 0.0f;   //  4 bytes  offset 24: emissive glow multiplier
    uint32_t  flags     = 0;      //  4 bytes  offset 28: bit0=unlit, bit1=alphaTest, bit2=PBR, bit3=hasNormalMap
    uint32_t  albedoTexIdx = 0;   //  4 bytes  offset 32: bindless index for albedo texture (Track G)
    uint32_t  normalTexIdx = 0;   //  4 bytes  offset 36: bindless index for normal map (Track G)
    uint32_t  _pad[2]   = {};     //  8 bytes  offset 40: reserved
};
static_assert(sizeof(GPUMaterial) == 48,
    "GPUMaterial size mismatch -- update GLSL structs in default.frag and default_bindless.frag");

// Bit flags for GPUMaterial::flags.
namespace MaterialFlags {
    inline constexpr uint32_t Unlit        = 1u << 0;  // skip lighting; output albedo only
    inline constexpr uint32_t AlphaTest    = 1u << 1;  // discard fragments below alpha threshold
    inline constexpr uint32_t UsePBR       = 1u << 2;  // Cook-Torrance BRDF instead of Blinn-Phong
    inline constexpr uint32_t HasNormalMap = 1u << 3;  // normal map at set=1, binding=1
}

// Owns a persistently-mapped GPU SSBO of GPUMaterial entries.
// Index 0 is always the default material (opaque white, 0.5 roughness).
//
// Pattern:
//   uint32_t redId = m_materials.add("red", {{1,0,0,1}});
//   m_materials.get(redId).emissive = 0.5f;   // edit in-place
//   m_materials.flush();                       // push changes to GPU before draw
class MaterialManager {
public:
    // Handle to the default material; always valid after init().
    static constexpr uint32_t kDefaultId = 0;

    MaterialManager() = default;
    ~MaterialManager() { destroy(); }
    MaterialManager(const MaterialManager&)            = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;

    // Create the GPU SSBO and register the default material (index 0).
    // maxMaterials: total SSBO capacity; grows GPU memory linearly.
    void init(VmaAllocator allocator, uint32_t maxMaterials = 256);
    void destroy();

    // Add a named material and return its handle.
    // Duplicate names are rejected (returns the existing handle).
    // Returns kDefaultId when the manager is full.
    uint32_t add(std::string name, const GPUMaterial& mat = {});

    // Add an anonymous material (no name lookup).
    uint32_t add(const GPUMaterial& mat = {});

    // Get or update a material by handle.
    // update() marks dirty; call flush() before the next draw.
    void     update(uint32_t id, const GPUMaterial& mat);
    GPUMaterial&       get(uint32_t id);
    const GPUMaterial& get(uint32_t id) const;

    // Name-based lookup; returns kDefaultId if the name is not registered.
    uint32_t find(std::string_view name) const;

    // Flush all CPU-side edits to GPU memory (no-op if not dirty).
    // Call once per frame before command-buffer submission.
    void flush();

    // GPU buffer handle -- bind to set=0, binding=3 as a storage buffer.
    VkBuffer     buffer()     const { return m_buffer; }
    VkDeviceSize bufferSize() const {
        return static_cast<VkDeviceSize>(m_capacity) * sizeof(GPUMaterial);
    }
    uint32_t count()    const { return m_count; }
    uint32_t capacity() const { return m_capacity; }

private:
    VmaAllocator  m_allocator  = nullptr;
    VkBuffer      m_buffer     = VK_NULL_HANDLE;
    VmaAllocation m_allocation = nullptr;
    GPUMaterial*  m_mapped     = nullptr;   // persistently-mapped CPU view
    uint32_t      m_capacity   = 0;
    uint32_t      m_count      = 0;
    bool          m_dirty      = false;

    std::unordered_map<std::string, uint32_t> m_nameIndex;
};

} // namespace vkt
