#include <doctest/doctest.h>

#include "scene/RenderWorld.h"
#include "scene/CullObject.h"
#include "scene/Mesh.h"
#include "config/RenderLimits.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static VkDescriptorSet fakeSet(uintptr_t n) {
    return reinterpret_cast<VkDescriptorSet>(n);
}

// World with one mesh, one texture, and the "default" fallback texture.
struct CullFixture {
    vkt::RenderWorld   world;
    vkt::MeshHandle    mh  = vkt::INVALID_HANDLE;
    vkt::TextureHandle th  = vkt::INVALID_HANDLE;
    vkt::TextureHandle th2 = vkt::INVALID_HANDLE; // second texture for multi-batch tests

    CullFixture() {
        mh  = world.registerMesh("cube",   vkt::Mesh{});
        th  = world.registerTexture("stone", fakeSet(1));
        th2 = world.registerTexture("metal", fakeSet(2));
        world.registerTexture("default",     fakeSet(99));
    }
    ~CullFixture() { world.destroy(); }

    vkt::ObjectId spawnAt(vkt::TextureHandle tex,
                          const glm::vec3& pos = {0.0f, 0.0f, 0.0f}) {
        const glm::mat4 t = glm::mat4{1.0f};
        vkt::ObjectId id = world.spawn(mh, tex, t);
        if (id != vkt::INVALID_HANDLE)
            world.setWorldPosition(id, glm::dvec3{pos});
        return id;
    }
};

// Run buildCullData with large enough buffers and return the result vector.
static std::pair<std::vector<vkt::CullObject>, std::vector<vkt::CullBatch>>
runCullData(vkt::RenderWorld& world,
            uint32_t maxObjects = vkt::RenderLimits::kMaxInstances,
            uint32_t maxBatches = vkt::RenderLimits::kMaxBatches) {
    std::vector<vkt::CullObject> objs(maxObjects);
    std::vector<vkt::CullBatch>  batches;
    const uint32_t n = world.buildCullData(
        objs.data(), maxObjects, maxBatches, batches);
    objs.resize(n);
    return {objs, batches};
}

// ---------------------------------------------------------------------------
// D3: High object count and culling tests
// ---------------------------------------------------------------------------

TEST_CASE("buildCullData -- empty world returns 0") {
    CullFixture f;
    auto [objs, batches] = runCullData(f.world);
    CHECK(objs.empty());
    CHECK(batches.empty());
}

TEST_CASE("buildCullData -- single object produces 1 cull entry") {
    CullFixture f;
    f.spawnAt(f.th);

    auto [objs, batches] = runCullData(f.world);
    CHECK(objs.size() == 1u);
    CHECK(batches.size() == 1u);
}

TEST_CASE("buildCullData -- N objects same batch produce N cull entries") {
    CullFixture f;
    constexpr uint32_t N = 16;
    for (uint32_t i = 0; i < N; ++i)
        f.spawnAt(f.th, glm::vec3{static_cast<float>(i), 0.0f, 0.0f});

    auto [objs, batches] = runCullData(f.world);
    CHECK(objs.size() == N);
    // All share the same (mesh, tex) so there should be exactly 1 batch.
    CHECK(batches.size() == 1u);
}

TEST_CASE("buildCullData -- objects with different textures create separate batches") {
    CullFixture f;
    f.spawnAt(f.th);
    f.spawnAt(f.th2);

    auto [objs, batches] = runCullData(f.world);
    CHECK(objs.size() == 2u);
    CHECK(batches.size() == 2u);
}

TEST_CASE("buildCullData -- maxObjects cap truncates output") {
    CullFixture f;
    constexpr uint32_t total = 20;
    constexpr uint32_t cap   = 5;
    for (uint32_t i = 0; i < total; ++i)
        f.spawnAt(f.th);

    auto [objs, batches] = runCullData(f.world, cap, vkt::RenderLimits::kMaxBatches);
    CHECK(objs.size() <= cap);
}

TEST_CASE("buildCullData -- maxObjects=0 produces empty output") {
    CullFixture f;
    f.spawnAt(f.th);
    f.spawnAt(f.th);

    auto [objs, batches] = runCullData(f.world, 0, vkt::RenderLimits::kMaxBatches);
    CHECK(objs.empty());
}

TEST_CASE("buildCullData -- maxBatches cap truncates batch list") {
    CullFixture f;
    // Register multiple textures to create many distinct batches.
    constexpr uint32_t kTexCount = 8;
    std::array<vkt::TextureHandle, kTexCount> texHandles{};
    for (uint32_t i = 0; i < kTexCount; ++i) {
        texHandles[i] = f.world.registerTexture(
            "tex" + std::to_string(i), fakeSet(10 + i));
    }
    for (uint32_t i = 0; i < kTexCount; ++i)
        f.spawnAt(texHandles[i]);

    constexpr uint32_t batchCap = 3;
    auto [objs, batches] = runCullData(f.world,
                                       vkt::RenderLimits::kMaxInstances, batchCap);
    CHECK(batches.size() <= batchCap);
}

TEST_CASE("buildCullData -- bounding sphere set via setBoundingSphere appears in cull data") {
    CullFixture f;
    const vkt::ObjectId id = f.spawnAt(f.th);
    f.world.setBoundingSphere(id, glm::vec3{1.0f, 2.0f, 3.0f}, 5.0f);

    auto [objs, batches] = runCullData(f.world);
    REQUIRE(objs.size() == 1u);

    // CullObject.boundingSphere: xyz = center, w = radius.
    CHECK(std::abs(objs[0].boundingSphere.w - 5.0f) < 1e-5f);
}

TEST_CASE("buildCullData -- materialId from spawn propagates into CullObject") {
    CullFixture f;
    constexpr uint32_t matId = 7;
    const glm::mat4 t{1.0f};
    const vkt::ObjectId id = f.world.spawn(f.mh, f.th, t, true, matId);
    CHECK(id != vkt::INVALID_HANDLE);

    auto [objs, batches] = runCullData(f.world);
    REQUIRE(objs.size() == 1u);
    CHECK(objs[0].materialId == matId);
}

TEST_CASE("buildCullData -- batchId values are contiguous starting from 0") {
    CullFixture f;
    // Spawn objects across 3 distinct batches.
    const vkt::TextureHandle th3 = f.world.registerTexture("glass", fakeSet(50));
    f.spawnAt(f.th);
    f.spawnAt(f.th2);
    f.spawnAt(th3);

    auto [objs, batches] = runCullData(f.world);
    REQUIRE(batches.size() == 3u);

    // Collect batchIds from the batch list -- they must be 0, 1, 2 in some order.
    std::vector<uint32_t> ids;
    for (const auto& b : batches) ids.push_back(b.batchId);
    std::sort(ids.begin(), ids.end());
    for (uint32_t i = 0; i < static_cast<uint32_t>(ids.size()); ++i)
        CHECK(ids[i] == i);
}

TEST_CASE("buildCullData -- firstInstance + maxInstances covers full batch") {
    CullFixture f;
    constexpr uint32_t N = 10;
    for (uint32_t i = 0; i < N; ++i)
        f.spawnAt(f.th);

    auto [objs, batches] = runCullData(f.world);
    REQUIRE(batches.size() == 1u);

    const vkt::CullBatch& b = batches[0];
    // firstInstance should be 0; maxInstances should be N for a fresh world.
    CHECK(b.firstInstance == 0u);
    CHECK(b.maxInstances  == N);
}

TEST_CASE("buildCullData -- despawned objects are not included") {
    CullFixture f;
    const vkt::ObjectId a = f.spawnAt(f.th);
    const vkt::ObjectId b = f.spawnAt(f.th);
    f.world.despawn(a);

    auto [objs, batches] = runCullData(f.world);
    CHECK(objs.size() == 1u);
    (void)b; // ensure b is alive
}

TEST_CASE("buildCullData -- stress: kMaxInstances objects do not overflow") {
    // Spawn exactly kMaxInstances objects and verify no truncation occurs
    // (the SSBO is sized to kMaxInstances, so this is the boundary condition).
    vkt::RenderWorld world;
    const vkt::MeshHandle mh = world.registerMesh("m", vkt::Mesh{});
    const vkt::TextureHandle th = world.registerTexture("t", fakeSet(1));
    world.registerTexture("default", fakeSet(99));

    constexpr uint32_t N = vkt::RenderLimits::kMaxInstances;
    for (uint32_t i = 0; i < N; ++i)
        world.spawn(mh, th);

    CHECK(world.objectCount() == N);

    std::vector<vkt::CullObject> objs(N);
    std::vector<vkt::CullBatch>  batches;
    const uint32_t written = world.buildCullData(
        objs.data(), N, vkt::RenderLimits::kMaxBatches, batches);

    // All objects must be present -- no silent truncation at the exact cap.
    CHECK(written == N);
    CHECK(batches.size() == 1u);
    CHECK(batches[0].maxInstances == N);

    world.destroy();
}

TEST_CASE("buildCullData -- stress: kMaxInstances+1 objects are capped at N") {
    vkt::RenderWorld world;
    const vkt::MeshHandle mh = world.registerMesh("m", vkt::Mesh{});
    const vkt::TextureHandle th = world.registerTexture("t", fakeSet(1));
    world.registerTexture("default", fakeSet(99));

    constexpr uint32_t N = vkt::RenderLimits::kMaxInstances;
    for (uint32_t i = 0; i <= N; ++i) // one over the cap
        world.spawn(mh, th);

    std::vector<vkt::CullObject> objs(N); // capacity == cap, not N+1
    std::vector<vkt::CullBatch>  batches;
    const uint32_t written = world.buildCullData(
        objs.data(), N, vkt::RenderLimits::kMaxBatches, batches);

    // Must not exceed the provided capacity.
    CHECK(written <= N);

    world.destroy();
}

TEST_CASE("buildCullData -- kMaxBatches batches do not overflow") {
    vkt::RenderWorld world;
    const vkt::MeshHandle mh = world.registerMesh("m", vkt::Mesh{});
    world.registerTexture("default", fakeSet(99));

    // Register exactly kMaxBatches textures and spawn one object per texture.
    constexpr uint32_t B = vkt::RenderLimits::kMaxBatches;
    for (uint32_t i = 0; i < B; ++i) {
        const vkt::TextureHandle th =
            world.registerTexture("t" + std::to_string(i), fakeSet(100 + i));
        world.spawn(mh, th);
    }

    std::vector<vkt::CullObject> objs(B);
    std::vector<vkt::CullBatch>  batches;
    const uint32_t written = world.buildCullData(
        objs.data(), B, B, batches);

    CHECK(written <= B);
    CHECK(batches.size() <= B);

    world.destroy();
}

TEST_CASE("buildCullData -- model matrix encodes world position relative to camera") {
    CullFixture f;
    const vkt::ObjectId id = f.spawnAt(f.th, {0.0f, 0.0f, 0.0f});
    f.world.setWorldPosition(id, glm::dvec3{100.0, 0.0, 0.0});

    // Camera at world origin: model[3] should encode x=100.
    {
        auto [objs, batches] = runCullData(f.world);
        REQUIRE(objs.size() == 1u);
        CHECK(std::abs(objs[0].model[3][0] - 100.0f) < 1e-3f);
    }

    // Camera at world x=100: relative x should be ~0.
    {
        std::vector<vkt::CullObject> objs(8);
        std::vector<vkt::CullBatch>  batches;
        f.world.buildCullData(objs.data(), 8, 64, batches,
                               glm::dvec3{100.0, 0.0, 0.0});
        REQUIRE(!batches.empty());
        CHECK(std::abs(objs[0].model[3][0]) < 1e-3f);
    }
}
