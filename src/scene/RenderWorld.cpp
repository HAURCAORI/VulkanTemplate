#include "RenderWorld.h"
#include "Frustum.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>

namespace {
// Extract the translation column from a model matrix and return it as dvec3.
glm::dvec3 extractTranslation(const glm::mat4& m) {
    return glm::dvec3(m[3][0], m[3][1], m[3][2]);
}
// Return the matrix with its translation column zeroed (rotation + scale only).
glm::mat4 stripTranslation(const glm::mat4& m) {
    glm::mat4 r = m;
    r[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return r;
}
} // anonymous namespace

namespace vkt {

// Static LOD selection helper.  Selects the Mesh* for the given camera distance.
// Iterates LOD levels (LOD 1, 2, ...) in ascending switchDist order.
// Falls back to slot.mesh (LOD 0) when dist is below the first switch distance.
/*static*/ const Mesh* RenderWorld::selectLOD(const MeshSlot& slot, float dist) noexcept {
    const Mesh* result = &slot.mesh;
    for (const auto& lod : slot.lods) {
        if (dist < lod.switchDist) break;
        result = &lod.mesh;
    }
    return result;
}

// -- Resource registration ------------------------------------------------------

MeshHandle RenderWorld::registerMesh(std::string name, Mesh mesh) {
    const MeshHandle h = static_cast<MeshHandle>(m_meshes.size());
    m_meshIndex[name] = h;
    m_meshes.push_back({std::move(name), std::move(mesh)});
    return h;
}

TextureHandle RenderWorld::registerTexture(std::string name, VkDescriptorSet set) {
    const TextureHandle h = static_cast<TextureHandle>(m_textures.size());
    m_textureIndex[name] = h;
    m_textures.push_back({std::move(name), set});
    return h;
}

MeshHandle RenderWorld::findMesh(std::string_view name) const {
    const auto it = m_meshIndex.find(name); // heterogeneous lookup: no string allocation
    return it != m_meshIndex.end() ? it->second : INVALID_HANDLE;
}

TextureHandle RenderWorld::findTexture(std::string_view name) const {
    const auto it = m_textureIndex.find(name); // heterogeneous lookup: no string allocation
    return it != m_textureIndex.end() ? it->second : INVALID_HANDLE;
}

Mesh* RenderWorld::mesh(MeshHandle h) {
    return h < m_meshes.size() ? &m_meshes[h].mesh : nullptr;
}

MeshHandle RenderWorld::registerLODMesh(std::string name,
                                        std::vector<Mesh>  meshes,
                                        std::vector<float> switchDists) {
    if (meshes.empty() || switchDists.size() != meshes.size() - 1)
        return INVALID_HANDLE;

    const MeshHandle h = static_cast<MeshHandle>(m_meshes.size());
    m_meshIndex[name] = h;

    MeshSlot slot;
    slot.name = std::move(name); // m_meshIndex[name] already copied it into the map key
    slot.mesh = std::move(meshes[0]);
    for (size_t i = 1; i < meshes.size(); ++i) {
        LODLevel lod;
        lod.switchDist = switchDists[i - 1];
        lod.mesh       = std::move(meshes[i]);
        slot.lods.push_back(std::move(lod));
    }
    m_meshes.push_back(std::move(slot));
    return h;
}

// -- Object lifecycle -----------------------------------------------------------

ObjectId RenderWorld::spawn(MeshHandle meshH, TextureHandle texH,
                            const glm::mat4& transform, bool rigidBody,
                            uint32_t materialId) {
    if (meshH >= m_meshes.size())
        return INVALID_HANDLE;

    if (texH >= m_textures.size() || m_textures[texH].set == VK_NULL_HANDLE) {
        const TextureHandle defaultH = findTexture("default");
        assert(defaultH != INVALID_HANDLE && "RenderWorld: \"default\" texture must be registered before spawning objects with an invalid texture handle");
        if (defaultH == INVALID_HANDLE)
            return INVALID_HANDLE;
        texH = defaultH;
    }

    ++m_activeCount;

    // Separate the translation from the rotation+scale.  Translation goes into
    // worldPosition (double) so draw() can apply the floating-origin offset.
    ObjectSlot slot;
    slot.meshHandle    = meshH;
    slot.textureHandle = texH;
    slot.transform     = stripTranslation(transform);
    slot.alive         = true;
    slot.rigidBody     = rigidBody;
    slot.worldPosition = extractTranslation(transform);
    slot.materialId    = materialId;

    ObjectId id;
    if (!m_freeList.empty()) {
        id = m_freeList.back();
        m_freeList.pop_back();
        m_objects[id] = slot;
    } else {
        id = static_cast<ObjectId>(m_objects.size());
        m_objects.push_back(slot);
    }
    return id;
}

void RenderWorld::despawn(ObjectId id) {
    if (id < m_objects.size() && m_objects[id].alive) {
        m_objects[id].alive = false;
        m_freeList.push_back(id);
        --m_activeCount;
    }
}

bool RenderWorld::alive(ObjectId id) const {
    return id < m_objects.size() && m_objects[id].alive;
}

void RenderWorld::setMaterialId(ObjectId id, uint32_t materialId) {
    if (id < m_objects.size() && m_objects[id].alive) m_objects[id].materialId = materialId;
}

uint32_t RenderWorld::getMaterialId(ObjectId id) const {
    return id < m_objects.size() ? m_objects[id].materialId : 0;
}

// -- Bounding volume ------------------------------------------------------------

void RenderWorld::setBoundingSphere(ObjectId id, glm::vec3 center, float radius) {
    if (id < m_objects.size() && m_objects[id].alive)
        m_objects[id].bounds = {BoundingVolume::Shape::Sphere, center, radius};
}

void RenderWorld::setBoundingVolume(ObjectId id, const BoundingVolume& vol) {
    if (id < m_objects.size() && m_objects[id].alive)
        m_objects[id].bounds = vol;
}

// -- Transform ------------------------------------------------------------------

void RenderWorld::setTransform(ObjectId id, const glm::mat4& t) {
    if (id < m_objects.size() && m_objects[id].alive) {
        m_objects[id].worldPosition = extractTranslation(t);
        m_objects[id].transform     = stripTranslation(t);
    }
}

glm::mat4 RenderWorld::getTransform(ObjectId id) const {
    if (id >= m_objects.size()) return glm::mat4{1.0f};
    const auto& obj = m_objects[id];
    glm::mat4 t = obj.transform;
    t[3] = glm::vec4(glm::vec3(obj.worldPosition), 1.0f); // lossy re-pack for compat
    return t;
}

void RenderWorld::setWorldPosition(ObjectId id, const glm::dvec3& pos) {
    if (id < m_objects.size() && m_objects[id].alive) m_objects[id].worldPosition = pos;
}

glm::dvec3 RenderWorld::getWorldPosition(ObjectId id) const {
    return id < m_objects.size() ? m_objects[id].worldPosition : glm::dvec3{0.0};
}

// -- Rendering ------------------------------------------------------------------

std::vector<DrawBatch> RenderWorld::buildBatches(
    InstanceData*      instanceOut,
    uint32_t           maxInstances,
    const glm::mat4&   viewProj,
    const glm::dvec3&  cameraWorldPos) const
{
    if (!instanceOut || maxInstances == 0) return {};

    using BatchKey = std::pair<const Mesh*, VkDescriptorSet>;

    // -- Pass 1: cull, LOD-select, and collect (key, inst) into flat scratch --
    // Reuses m_batchEntryScratch capacity from the previous call (no per-frame alloc).
    m_batchEntryScratch.clear();

    const Frustum frustum = Frustum::fromVP(viewProj);
    size_t        seen    = 0;
    const size_t  total   = m_activeCount;

    for (const auto& obj : m_objects) {
        if (seen == total) break;
        if (!obj.alive || obj.meshHandle >= m_meshes.size()) continue;
        ++seen;

        if (obj.textureHandle >= m_textures.size()) continue;
        VkDescriptorSet set = m_textures[obj.textureHandle].set;
        if (set == VK_NULL_HANDLE) continue;

        const glm::vec3 relOffset = glm::vec3(obj.worldPosition - cameraWorldPos);
        const glm::mat4 model     = glm::translate(glm::mat4{1.0f}, relOffset) * obj.transform;

        if (!frustum.test(obj.bounds, model, obj.rigidBody)) continue;

        const float dist = glm::length(relOffset);
        const Mesh* mesh = selectLOD(m_meshes[obj.meshHandle], dist);

        InstanceData inst;
        inst.model        = model;
        inst.normalMatrix = obj.rigidBody
            ? glm::mat4(glm::mat3(obj.transform))
            : glm::mat4(glm::inverseTranspose(glm::mat3(obj.transform)));
        inst.materialId   = obj.materialId;

        m_batchEntryScratch.push_back({{mesh, set}, inst});
    }

    // Sort by (mesh*, textureSet) so identical keys are adjacent.
    // Deterministic order reduces descriptor bind churn across frames.
    // Use std::less (not operator<) for pointer comparison: std::less is guaranteed
    // to give a strict total order over arbitrary pointer values; operator< is not.
    std::sort(m_batchEntryScratch.begin(), m_batchEntryScratch.end(),
              [](const BatchEntry& a, const BatchEntry& b) {
                  if (a.key.first != b.key.first)
                      return std::less<const Mesh*>{}(a.key.first, b.key.first);
                  return std::less<VkDescriptorSet>{}(a.key.second, b.key.second);
              });

    // -- Pass 2: linear scan to flatten into SSBO and build batch list --------
    std::vector<DrawBatch> result;
    uint32_t offset = 0;

    for (size_t i = 0, n = m_batchEntryScratch.size(); i < n; ) {
        const BatchKey& key = m_batchEntryScratch[i].key;
        size_t j = i + 1;
        while (j < n && m_batchEntryScratch[j].key == key) ++j;

        const uint32_t count = static_cast<uint32_t>(j - i);
        if (offset + count > maxInstances) break; // capacity guard

        for (uint32_t k = 0; k < count; ++k)
            instanceOut[offset + k] = m_batchEntryScratch[i + k].inst;

        result.push_back({key.first, key.second, offset, count});
        offset += count;
        i = j;
    }

    return result;
}

std::vector<DrawBatch> RenderWorld::buildBatchesNoCull(
    InstanceData*      instanceOut,
    uint32_t           maxInstances,
    const glm::dvec3&  cameraWorldPos) const
{
    if (!instanceOut || maxInstances == 0) return {};

    using BatchKey = std::pair<const Mesh*, VkDescriptorSet>;

    m_batchEntryScratch.clear();

    size_t seen = 0;
    for (const auto& obj : m_objects) {
        if (seen == m_activeCount) break;
        if (!obj.alive || obj.meshHandle >= m_meshes.size()) continue;
        ++seen;

        if (obj.textureHandle >= m_textures.size()) continue;
        const VkDescriptorSet set = m_textures[obj.textureHandle].set;
        if (set == VK_NULL_HANDLE) continue;

        const glm::vec3 relOffset = glm::vec3(obj.worldPosition - cameraWorldPos);
        const glm::mat4 model     = glm::translate(glm::mat4{1.0f}, relOffset) * obj.transform;
        const float     dist      = glm::length(relOffset);
        const Mesh*     mesh      = selectLOD(m_meshes[obj.meshHandle], dist);

        InstanceData inst;
        inst.model        = model;
        inst.normalMatrix = obj.rigidBody
            ? glm::mat4(glm::mat3(obj.transform))
            : glm::mat4(glm::inverseTranspose(glm::mat3(obj.transform)));
        inst.materialId   = obj.materialId;

        m_batchEntryScratch.push_back({{mesh, set}, inst});
    }

    std::sort(m_batchEntryScratch.begin(), m_batchEntryScratch.end(),
              [](const BatchEntry& a, const BatchEntry& b) {
                  if (a.key.first != b.key.first)
                      return std::less<const Mesh*>{}(a.key.first, b.key.first);
                  return std::less<VkDescriptorSet>{}(a.key.second, b.key.second);
              });

    std::vector<DrawBatch> result;
    uint32_t offset = 0;
    for (size_t i = 0, n = m_batchEntryScratch.size(); i < n; ) {
        const BatchKey& key = m_batchEntryScratch[i].key;
        size_t j = i + 1;
        while (j < n && m_batchEntryScratch[j].key == key) ++j;

        const uint32_t count = static_cast<uint32_t>(j - i);
        if (offset + count > maxInstances) break;

        for (uint32_t k = 0; k < count; ++k)
            instanceOut[offset + k] = m_batchEntryScratch[i + k].inst;

        result.push_back({key.first, key.second, offset, count});
        offset += count;
        i = j;
    }
    return result;
}

void RenderWorld::draw(VkCommandBuffer cmd, VkPipelineLayout layout,
                       const std::vector<DrawBatch>& batches) const {
    VkDescriptorSet lastSet = VK_NULL_HANDLE;
    for (const auto& b : batches) {
        if (b.textureSet != lastSet) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    layout, 1, 1, &b.textureSet, 0, nullptr);
            lastSet = b.textureSet;
        }
        b.mesh->drawInstanced(cmd, b.count, b.firstInstance);
    }
}

void RenderWorld::drawShadow(VkCommandBuffer cmd,
                             const std::vector<DrawBatch>& batches) const {
    for (const auto& b : batches) {
        b.mesh->drawInstanced(cmd, b.count, b.firstInstance);
    }
}

// -- GPU compute cull path ------------------------------------------------------

uint32_t RenderWorld::buildCullData(CullObject*             cullObjectsOut,
                                    uint32_t                maxObjects,
                                    uint32_t                maxBatches,
                                    std::vector<CullBatch>& batchListOut,
                                    const glm::dvec3&       cameraWorldPos) const {
    if (!cullObjectsOut || maxObjects == 0 || maxBatches == 0) return 0;

    using BatchKey = std::pair<const Mesh*, VkDescriptorSet>;

    // Pass 1: accept up to maxObjects valid objects while limiting unique batches
    // to maxBatches; count accepted objects per (mesh*, textureSet) pair.
    // No frustum test here -- the GPU compute shader does it.
    std::unordered_map<BatchKey, uint32_t, BatchKeyHash> batchCounts; // (mesh*, set) -> object count
    m_indexScratch.clear(); // reuse capacity from previous call (avoids per-frame realloc)
    if (m_indexScratch.capacity() < maxObjects) m_indexScratch.reserve(maxObjects);
    {
        size_t seen = 0;
        for (size_t objIndex = 0; objIndex < m_objects.size(); ++objIndex) {
            const auto& obj = m_objects[objIndex];
            if (seen == m_activeCount) break;
            if (!obj.alive || obj.meshHandle >= m_meshes.size()) continue;
            ++seen;
            if (obj.textureHandle >= m_textures.size()) continue;
            VkDescriptorSet set = m_textures[obj.textureHandle].set;
            if (set == VK_NULL_HANDLE) continue;

            const float dist = glm::length(glm::vec3(obj.worldPosition - cameraWorldPos));
            const Mesh* mesh = selectLOD(m_meshes[obj.meshHandle], dist);
            const BatchKey key{mesh, set};

            auto it = batchCounts.find(key);
            if (it == batchCounts.end()) {
                if (batchCounts.size() >= maxBatches) continue;
                it = batchCounts.emplace(key, 0u).first;
            }
            ++(it->second);
            m_indexScratch.push_back(objIndex);
            if (m_indexScratch.size() >= maxObjects) break;
        }
    }

    // Build sorted batch list; assign batchId (= index) and firstInstance.
    // Sort keys for deterministic batch assignment and reduced descriptor bind churn.
    batchListOut.clear();
    std::vector<BatchKey> sortedKeys;
    sortedKeys.reserve(batchCounts.size());
    for (const auto& entry : batchCounts) sortedKeys.push_back(entry.first);
    std::sort(sortedKeys.begin(), sortedKeys.end(),
              [](const BatchKey& a, const BatchKey& b) {
                  if (a.first != b.first)
                      return std::less<const Mesh*>{}(a.first, b.first);
                  return std::less<VkDescriptorSet>{}(a.second, b.second);
              });

    std::unordered_map<BatchKey, uint32_t, BatchKeyHash> batchIdMap; // (mesh*, set) -> batchId
    batchIdMap.reserve(sortedKeys.size());
    uint32_t instanceOffset = 0;
    for (const BatchKey& key : sortedKeys) {
        const uint32_t bid   = static_cast<uint32_t>(batchListOut.size());
        const uint32_t count = batchCounts.at(key);
        batchIdMap[key] = bid;
        batchListOut.push_back({key.first, key.second, bid, instanceOffset, count});
        instanceOffset += count;
    }

    // Pass 2: fill CullObject array for the accepted objects only.
    uint32_t objectCount = 0;
    {
        for (size_t objIndex : m_indexScratch) {
            if (objectCount >= maxObjects) break;
            const auto& obj = m_objects[objIndex];
            const VkDescriptorSet set = m_textures[obj.textureHandle].set;

            // Floating-origin model matrix.
            const glm::vec3 relOffset = glm::vec3(obj.worldPosition - cameraWorldPos);
            const glm::mat4 model     = glm::translate(glm::mat4{1.0f}, relOffset) * obj.transform;

            // Normal matrix.
            const glm::mat4 normalMatrix = obj.rigidBody
                ? glm::mat4(glm::mat3(obj.transform))
                : glm::mat4(glm::inverseTranspose(glm::mat3(obj.transform)));

            // World-space bounding sphere (w=0 means no cull).
            glm::vec4 bsphere{0.0f};
            if (obj.bounds.shape == BoundingVolume::Shape::Sphere) {
                glm::vec3 worldCenter = glm::vec3(model * glm::vec4(obj.bounds.center, 1.0f));
                float worldRadius     = obj.bounds.radius;
                if (!obj.rigidBody) {
                    // Inflate by the maximum column scale for non-rigid objects.
                    worldRadius *= std::max({glm::length(glm::vec3(model[0])),
                                             glm::length(glm::vec3(model[1])),
                                             glm::length(glm::vec3(model[2]))});
                }
                bsphere = glm::vec4(worldCenter, worldRadius);
            }

            // Batch assignment.
            const float dist = glm::length(relOffset);
            const Mesh* mesh = selectLOD(m_meshes[obj.meshHandle], dist);
            const uint32_t batchId = batchIdMap.at({mesh, set});
            const CullBatch& batch = batchListOut[batchId];

            cullObjectsOut[objectCount++] = CullObject{
                model,
                normalMatrix,
                bsphere,
                batchId,
                obj.materialId,
                batch.firstInstance,
                batch.maxInstances
            };
        }
    }

    return objectCount;
}

void RenderWorld::drawIndirect(VkCommandBuffer cmd, VkPipelineLayout layout,
                                const std::vector<CullBatch>& batches,
                                VkBuffer indirectBuffer) const {
    VkDescriptorSet lastSet = VK_NULL_HANDLE;
    for (const auto& b : batches) {
        if (b.textureSet != lastSet) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    layout, 1, 1, &b.textureSet, 0, nullptr);
            lastSet = b.textureSet;
        }
        b.mesh->bindBuffers(cmd);
        const VkDeviceSize offset =
            static_cast<VkDeviceSize>(b.batchId) * sizeof(VkDrawIndexedIndirectCommand);
        vkCmdDrawIndexedIndirect(cmd, indirectBuffer, offset, 1,
                                 sizeof(VkDrawIndexedIndirectCommand));
    }
}

// Compatibility wrapper: builds batches then calls draw().
void RenderWorld::draw(VkCommandBuffer cmd, VkPipelineLayout layout,
                       const glm::mat4& viewProj,
                       InstanceData*    instanceBuffer,
                       uint32_t         maxInstances,
                       const glm::dvec3& cameraWorldPos) const {
    auto batches = buildBatches(instanceBuffer, maxInstances, viewProj, cameraWorldPos);
    draw(cmd, layout, batches);
}

// -- Statistics -----------------------------------------------------------------

size_t RenderWorld::objectCount() const {
    return m_activeCount; // O(1): maintained by spawn/despawn
}

// -- Shutdown -------------------------------------------------------------------

void RenderWorld::destroy() {
    for (auto& s : m_meshes) {
        s.mesh.destroy();
        for (auto& lod : s.lods) lod.mesh.destroy();
    }
    m_meshes.clear();
    m_textures.clear();
    m_objects.clear();
    m_freeList.clear();
    m_meshIndex.clear();
    m_textureIndex.clear();
    m_activeCount = 0;
}

} // namespace vkt
