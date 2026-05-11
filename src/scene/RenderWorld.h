#pragma once

#include <volk.h>
#include <glm/glm.hpp>

#include "BoundingVolume.h"
#include "CullObject.h"
#include "InstanceData.h"
#include "Mesh.h"

#include <string>
#include <string_view>
#include <vector>
#include <limits>
#include <unordered_map>
#include <functional>

namespace vkt {

// Batch descriptor produced by buildBatches(); consumed by draw() and drawShadow().
struct DrawBatch {
    const Mesh*     mesh;          // LOD-selected mesh to draw
    VkDescriptorSet textureSet;    // material descriptor (set 1) -- NULL for depth-only draws
    uint32_t        firstInstance; // offset into instance SSBO
    uint32_t        count;         // number of instances in this batch
};

// Batch descriptor produced by buildCullData(); consumed by drawIndirect() and
// used to fill VkDrawIndexedIndirectCommand entries in the IndirectBuffer.
struct CullBatch {
    const Mesh*     mesh;          // LOD-selected mesh (same selection as drawBatches)
    VkDescriptorSet textureSet;    // material descriptor (set 1)
    uint32_t        batchId;       // index into the VkDrawIndexedIndirectCommand array
    uint32_t        firstInstance; // base offset in InstanceData SSBO for this batch
    uint32_t        maxInstances;  // total objects in this batch (capacity for compute)
};

using MeshHandle    = uint32_t;
using TextureHandle = uint32_t;
using ObjectId      = uint32_t;

// Sentinel: returned by find* when a name is not registered.
inline constexpr uint32_t INVALID_HANDLE = ~uint32_t{0};

// RenderWorld owns all mesh GPU resources and manages a dynamic object list.
// Textures are referenced by their VkDescriptorSet only  --  the caller owns the
// underlying Image/Sampler GPU resources and must outlive the world.
//
// Simulation pattern (no GPU allocation per object):
//
//   // One-time setup:
//   MeshHandle sphere = world.registerMesh("sphere", std::move(myMesh));
//   TextureHandle tex = world.registerTexture("stone", stoneSet);
//
//   // Each physics spawn (O(1), no GPU cost):
//   ObjectId id = world.spawn(sphere, tex, transform);
//
//   // Each simulation tick (zero GPU cost):
//   world.setTransform(id, newTransform);
//
//   // On removal (O(1), slot recycled):
//   world.despawn(id);
//
// Thread safety: not thread-safe. Drive from one thread per frame.
// For a separate physics thread, swap a transform buffer each frame boundary
// and call setTransform() from the render thread after the swap.
class RenderWorld {
public:
    RenderWorld() = default;
    RenderWorld(const RenderWorld&)            = delete;
    RenderWorld& operator=(const RenderWorld&) = delete;

    // -- Resource registration -------------------------------------------------
    // Meshes are owned; registering transfers GPU resource ownership to the world.
    // Textures are referenced only; the caller must keep Image/Sampler alive.
    MeshHandle    registerMesh(std::string name, Mesh mesh);
    TextureHandle registerTexture(std::string name, VkDescriptorSet set);

    // Register a mesh with multiple LOD levels.
    // meshes[0] = highest detail (used when dist < switchDists[0]).
    // meshes[i] = used when dist >= switchDists[i-1] and dist < switchDists[i].
    // meshes.back() = lowest detail (used when dist >= switchDists.back()).
    // Requires: !meshes.empty() && switchDists.size() == meshes.size() - 1.
    MeshHandle    registerLODMesh(std::string name,
                                  std::vector<Mesh>  meshes,
                                  std::vector<float> switchDists);

    MeshHandle    findMesh(std::string_view name) const;
    TextureHandle findTexture(std::string_view name) const;

    Mesh* mesh(MeshHandle h);

    // -- Object lifecycle (CPU-only, no GPU allocation/free per call) ----------
    //
    // rigidBody = true  (default): object has only rotation + translation.
    //   draw() uses mat4(mat3(model)) as normalMatrix  --  cheap extraction, no inversion.
    //   Correct for all physics rigid bodies.
    //
    // rigidBody = false: object may have non-uniform scale.
    //   draw() uses inverseTranspose(mat3(model))  --  correct but ~4x more expensive.
    //   Use only when you actually apply non-uniform scaling to the transform.
    // Returns INVALID_HANDLE when mesh is invalid.
    // If texture is invalid, spawn falls back to the "default" texture when registered.
    // materialId: index into the MaterialManager SSBO; 0 = default white material.
    ObjectId spawn(MeshHandle mesh, TextureHandle texture,
                   const glm::mat4& transform = glm::mat4{1.0f},
                   bool     rigidBody  = true,
                   uint32_t materialId = 0);
    void     despawn(ObjectId id);   // O(1): marks slot free; slot recycled by next spawn
    bool     alive(ObjectId id) const;

    // Change the material of a live object without despawning.
    void     setMaterialId(ObjectId id, uint32_t materialId);
    uint32_t getMaterialId(ObjectId id) const;

    // -- Transform (writes a mat4 in CPU memory, written to the instance SSBO) --
    // setTransform stores the rotation+scale part of t; the translation column
    // is extracted and stored as a double-precision world position (see setWorldPosition).
    void      setTransform(ObjectId id, const glm::mat4& t);

    // WARNING: the translation column of the returned matrix is a lossy float
    // reconstruction of the internal dvec3 world position.  For large-world scenes
    // (positions beyond ~1e6 units) this will lose precision silently.
    // Use getWorldPosition() to retrieve the exact double-precision position.
    glm::mat4 getTransform(ObjectId id) const;

    // -- Double-precision world position (floating origin) ---------------------
    // For large-scale scenes (e.g. space simulation), set the object's position
    // in double precision here.  Combine with setTransform() for rotation/scale.
    // draw() subtracts the camera's world position in double before casting to
    // float, keeping GPU coordinates well within float32 range.
    void       setWorldPosition(ObjectId id, const glm::dvec3& pos);
    glm::dvec3 getWorldPosition(ObjectId id) const;

    // -- Bounding volume (object space) ----------------------------------------
    // Used for frustum culling in draw(). Set after spawn().
    // Objects with no bounding volume (the default) are never culled.
    //
    // Sphere culling  --  conservative but fast:
    //   center: object-space center (usually {0,0,0} for symmetric meshes).
    //   radius: must enclose all vertices; use the circumscribed radius to avoid
    //           objects popping out when the sphere exits the frustum before the mesh.
    //   For rigid bodies the world-space radius equals the object-space radius.
    //   For scaled objects the radius is inflated automatically in draw().
    //
    // Example:
    //   ObjectId ball = world.spawn(sphereMesh, tex, transform);
    //   world.setBoundingSphere(ball, {0,0,0}, 0.5f);
    void setBoundingSphere(ObjectId id, glm::vec3 center, float radius);

    // Set an arbitrary BoundingVolume directly (for future AABB / OBB support).
    void setBoundingVolume(ObjectId id, const BoundingVolume& vol);

    // -- Rendering -------------------------------------------------------------

    // Build instance batches: cull + LOD select + group by (mesh, texture).
    // Writes per-instance data into instanceOut (max maxInstances entries).
    // Returns batch descriptors for use with draw() and drawShadow().
    // Call once per frame; pass the returned list to both draw() and drawShadow().
    std::vector<DrawBatch> buildBatches(
        InstanceData*      instanceOut,
        uint32_t           maxInstances,
        const glm::mat4&   viewProj,
        const glm::dvec3&  cameraWorldPos = {0.0, 0.0, 0.0}) const;

    // Build instance batches without frustum culling (all alive objects).
    // Intended for shadow-map caster rendering where camera-frustum culling can
    // incorrectly drop off-screen casters that still shadow visible pixels.
    std::vector<DrawBatch> buildBatchesNoCull(
        InstanceData*      instanceOut,
        uint32_t           maxInstances,
        const glm::dvec3&  cameraWorldPos = {0.0, 0.0, 0.0}) const;

    // Issue geometry draw calls from a pre-built batch list.
    // Binds material texture (set 1) per batch; skips redundant binds.
    // Assumes set 0 (UBO + SSBO) is already bound.
    void draw(VkCommandBuffer cmd, VkPipelineLayout layout,
              const std::vector<DrawBatch>& batches) const;

    // Issue depth-only draw calls for shadow map rendering.
    // Does NOT bind any material descriptor set (depth shader doesn't sample textures).
    // Assumes set 0 (SSBO) is already bound with the shadow pipeline.
    void drawShadow(VkCommandBuffer cmd,
                    const std::vector<DrawBatch>& batches) const;

    // -- GPU compute cull path -------------------------------------------------
    //
    // buildCullData() replaces buildBatches() for the GPU-driven indirect draw path.
    // It performs two linear passes over the object list:
    //   Pass 1: count objects per (mesh*, textureSet) batch pair, assign firstInstance.
    //   Pass 2: fill the CullObject SSBO with model matrix, normalMatrix, bounding
    //           sphere, batchId, materialId, firstInstance and maxInstances per object.
    //
    // The GPU compute cull shader (cull.comp) then:
    //   - Tests each CullObject's bounding sphere against the camera frustum.
    //   - Writes visible instances to the InstanceData SSBO via atomicAdd.
    //   - Updates VkDrawIndexedIndirectCommand.instanceCount per batch.
    //
    // cullObjectsOut -- CullObject SSBO mapped pointer (one per alive object).
    // maxObjects     -- capacity guard.
    // batchListOut   -- batch metadata needed to fill the indirect buffer and draw.
    // cameraWorldPos -- for floating-origin model matrix computation.
    //
    // Returns the number of CullObject entries written to cullObjectsOut.
    uint32_t buildCullData(CullObject*             cullObjectsOut,
                           uint32_t                maxObjects,
                           uint32_t                maxBatches,
                           std::vector<CullBatch>& batchListOut,
                           const glm::dvec3&       cameraWorldPos = {0.0, 0.0, 0.0}) const;

    // Issue one vkCmdDrawIndexedIndirect per batch.
    // The GPU compute pass must have completed before this is called (insert a
    // COMPUTE_WRITE -> VERTEX_SHADER + DRAW_INDIRECT barrier between them).
    // indirectBuffer must contain valid VkDrawIndexedIndirectCommand data indexed
    // by CullBatch::batchId (filled by compute shader + CPU for static fields).
    // skipTextureBinds: when true (bindless path), per-batch set=1 binds are skipped;
    //   the caller must have already bound the bindless descriptor set to set=1.
    void drawIndirect(VkCommandBuffer cmd, VkPipelineLayout layout,
                      const std::vector<CullBatch>& batches,
                      VkBuffer indirectBuffer,
                      bool skipTextureBinds = false) const;

    // Compatibility wrapper: builds batches then calls draw().
    // Call inside an active render pass with the pipeline already bound.
    // Writes per-instance data into instanceBuffer (one InstanceData per visible
    // object), then issues one vkCmdDrawIndexed per (mesh, texture) batch.
    // Binds set 1 (material) per batch; skips redundant binds within a batch.
    // Culls objects whose bounding volume is outside the view frustum.
    // Selects the appropriate LOD level per object based on camera distance.
    //
    // instanceBuffer -- persistently-mapped SSBO (set=0, binding=1), one per frame.
    // maxInstances   -- capacity guard; extra objects are silently dropped.
    // viewProj       = camera.projectionMatrix() * camera.viewMatrix()
    // cameraWorldPos = camera.worldPosition()  (double-precision floating origin)
    //   Each object's position is expressed as (worldPosition - cameraWorldPos),
    //   keeping GPU coordinates near zero regardless of scene scale.
    void draw(VkCommandBuffer cmd, VkPipelineLayout layout,
              const glm::mat4& viewProj,
              InstanceData*    instanceBuffer,
              uint32_t         maxInstances,
              const glm::dvec3& cameraWorldPos = {0.0, 0.0, 0.0}) const;

    // -- Statistics ------------------------------------------------------------
    size_t objectCount()  const;
    size_t meshCount()    const { return m_meshes.size(); }
    size_t textureCount() const { return m_textures.size(); }

    // Destroys all owned mesh GPU resources. Call before the VmaAllocator is
    // destroyed and before any Texture resources registered with this world.
    void destroy();

private:
    // Extra LOD level: switch to this mesh when camera distance >= switchDist.
    struct LODLevel {
        float switchDist = std::numeric_limits<float>::max();
        Mesh  mesh;
    };

    struct MeshSlot {
        std::string         name;
        Mesh                mesh;     // LOD 0 (highest detail, always valid)
        std::vector<LODLevel> lods;  // LOD 1, 2, ... (empty = no extra LODs)
    };
    struct TextureSlot {
        std::string     name;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };
    struct ObjectSlot {
        MeshHandle    meshHandle    = INVALID_HANDLE;
        TextureHandle textureHandle = INVALID_HANDLE;
        // Rotation + scale only (no translation column).
        // Translation is stored separately in worldPosition as double.
        glm::mat4     transform     {1.0f};
        bool          alive         = false;
        // When true, normalMatrix = mat4(mat3(transform))  --  no inversion needed.
        // Correct for rigid bodies (rotation only, det(R)=1).
        bool          rigidBody     = true;
        BoundingVolume bounds; // culling volume (object space); Shape::None = never cull
        // Double-precision world position (floating origin).
        // draw() subtracts the camera world position before casting to float.
        glm::dvec3    worldPosition {0.0, 0.0, 0.0};
        // Index into the MaterialManager SSBO (set=0, binding=3).
        // Passed through to InstanceData::materialId each frame.
        uint32_t      materialId    = 0;
    };

    // LOD selection: returns the Mesh* to use for a given camera distance.
    // Shared by buildBatches(), buildBatchesNoCull(), and buildCullData().
    static const Mesh* selectLOD(const MeshSlot& slot, float dist) noexcept;

    // Transparent hasher for std::unordered_map: enables find(string_view) without
    // allocating a temporary std::string (C++20 heterogeneous lookup).
    struct StringHash {
        using is_transparent = void;
        size_t operator()(std::string_view sv) const noexcept { return std::hash<std::string_view>{}(sv); }
        size_t operator()(const std::string& s)  const noexcept { return std::hash<std::string_view>{}(s);  }
    };

    // Hasher for (Mesh*, VkDescriptorSet) batch keys.
    // Valid on 64-bit targets where VkDescriptorSet is a typed pointer.
    struct BatchKeyHash {
        size_t operator()(const std::pair<const Mesh*, VkDescriptorSet>& k) const noexcept {
            const size_t h1 = reinterpret_cast<size_t>(k.first);
            const size_t h2 = reinterpret_cast<size_t>(k.second);
            return h1 ^ (h2 * 2654435761u + 0x9e3779b9u + (h1 << 6) + (h1 >> 2));
        }
    };

    // Flat entry used by buildBatches / buildBatchesNoCull scratch storage:
    // one (key, instance-data) pair per visible object, sorted before linearisation.
    struct BatchEntry {
        std::pair<const Mesh*, VkDescriptorSet> key;
        InstanceData                            inst;
    };

    std::vector<MeshSlot>    m_meshes;
    std::vector<TextureSlot> m_textures;
    std::vector<ObjectSlot>  m_objects;
    std::vector<ObjectId>    m_freeList;  // recycled object slots
    size_t                   m_activeCount = 0; // live count; avoids O(n) scan

    // O(1) name -> handle lookup. Kept in sync with m_meshes / m_textures.
    // Registration is append-only so handles never change after insertion.
    // StringHash + std::equal_to<> enable heterogeneous lookup with string_view
    // (avoids heap allocation on every findMesh / findTexture call).
    std::unordered_map<std::string, MeshHandle,    StringHash, std::equal_to<>> m_meshIndex;
    std::unordered_map<std::string, TextureHandle, StringHash, std::equal_to<>> m_textureIndex;

    // Scratch storage reused across build*() calls to avoid per-frame heap allocation.
    // Cleared at the start of each call; capacity grows to the high-water mark over time.
    mutable std::vector<size_t>     m_indexScratch;      // buildCullData: accepted object indices
    mutable std::vector<BatchEntry> m_batchEntryScratch; // buildBatches*: flat (key, inst) pairs
};

} // namespace vkt
