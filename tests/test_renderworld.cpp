#include <doctest/doctest.h>

#include "scene/RenderWorld.h"
#include "scene/Mesh.h"

#include <glm/glm.hpp>
#include <cstdint>
#include <cmath>

// Helpers --------------------------------------------------------------------

// Cast an integer to a fake VkDescriptorSet for tests that need a non-null set.
// No actual Vulkan device is involved; these are opaque handle values only.
static VkDescriptorSet fakeSet(uintptr_t n) {
    return reinterpret_cast<VkDescriptorSet>(n);
}

// Fixture: world with one mesh and one texture pre-registered.
struct WorldFixture {
    vkt::RenderWorld  world;
    vkt::MeshHandle   mh = vkt::INVALID_HANDLE;
    vkt::TextureHandle th = vkt::INVALID_HANDLE;

    WorldFixture() {
        mh = world.registerMesh("cube", vkt::Mesh{});
        th = world.registerTexture("stone", fakeSet(1));
    }
    ~WorldFixture() { world.destroy(); }
};

// -- Registration & lookup ---------------------------------------------------

TEST_CASE("RenderWorld -- registerMesh and findMesh") {
    vkt::RenderWorld world;
    const vkt::MeshHandle h = world.registerMesh("sphere", vkt::Mesh{});

    CHECK(h != vkt::INVALID_HANDLE);
    CHECK(world.findMesh("sphere") == h);
    CHECK(world.findMesh("missing") == vkt::INVALID_HANDLE);
    CHECK(world.meshCount() == 1);

    world.destroy();
}

TEST_CASE("RenderWorld -- registerTexture and findTexture") {
    vkt::RenderWorld world;
    const vkt::TextureHandle h = world.registerTexture("rock", fakeSet(42));

    CHECK(h != vkt::INVALID_HANDLE);
    CHECK(world.findTexture("rock") == h);
    CHECK(world.findTexture("missing") == vkt::INVALID_HANDLE);

    world.destroy();
}

TEST_CASE("RenderWorld -- multiple registrations get distinct handles") {
    vkt::RenderWorld world;
    const vkt::MeshHandle h0 = world.registerMesh("a", vkt::Mesh{});
    const vkt::MeshHandle h1 = world.registerMesh("b", vkt::Mesh{});
    const vkt::MeshHandle h2 = world.registerMesh("c", vkt::Mesh{});

    CHECK(h0 != h1);
    CHECK(h1 != h2);
    CHECK(world.findMesh("a") == h0);
    CHECK(world.findMesh("b") == h1);
    CHECK(world.findMesh("c") == h2);

    world.destroy();
}

// -- Object lifecycle --------------------------------------------------------

TEST_CASE("RenderWorld -- spawn alive despawn objectCount") {
    WorldFixture f;
    CHECK(f.world.objectCount() == 0);

    const vkt::ObjectId id = f.world.spawn(f.mh, f.th);
    CHECK(id != vkt::INVALID_HANDLE);
    CHECK(f.world.alive(id));
    CHECK(f.world.objectCount() == 1);

    f.world.despawn(id);
    CHECK(!f.world.alive(id));
    CHECK(f.world.objectCount() == 0);
}

TEST_CASE("RenderWorld -- despawned slot is recycled by next spawn") {
    WorldFixture f;
    const vkt::ObjectId first = f.world.spawn(f.mh, f.th);
    f.world.despawn(first);
    const vkt::ObjectId second = f.world.spawn(f.mh, f.th);
    // Free-list pop: second spawn should reuse the same slot index.
    CHECK(second == first);
}

TEST_CASE("RenderWorld -- spawn with invalid mesh returns INVALID_HANDLE") {
    vkt::RenderWorld world;
    world.registerTexture("t", fakeSet(1));
    const vkt::TextureHandle th = world.findTexture("t");

    const vkt::ObjectId id = world.spawn(vkt::INVALID_HANDLE, th);
    CHECK(id == vkt::INVALID_HANDLE);
    CHECK(world.objectCount() == 0);

    world.destroy();
}

TEST_CASE("RenderWorld -- spawn with invalid texture falls back to default") {
    vkt::RenderWorld world;
    const vkt::MeshHandle mh = world.registerMesh("m", vkt::Mesh{});
    // Register the "default" fallback texture.
    world.registerTexture("default", fakeSet(99));

    // Passing INVALID_HANDLE for texture should fall back to "default" and succeed.
    const vkt::ObjectId id = world.spawn(mh, vkt::INVALID_HANDLE);
    CHECK(id != vkt::INVALID_HANDLE);

    world.destroy();
}

TEST_CASE("RenderWorld -- spawn without default texture and invalid handle returns INVALID") {
    vkt::RenderWorld world;
    const vkt::MeshHandle mh = world.registerMesh("m", vkt::Mesh{});
    // No "default" texture registered; spawn should fail gracefully.
    const vkt::ObjectId id = world.spawn(mh, vkt::INVALID_HANDLE);
    CHECK(id == vkt::INVALID_HANDLE);

    world.destroy();
}

TEST_CASE("RenderWorld -- alive() is false for out-of-range id") {
    vkt::RenderWorld world;
    CHECK(!world.alive(0));
    CHECK(!world.alive(vkt::INVALID_HANDLE));
    world.destroy();
}

TEST_CASE("RenderWorld -- materialId is stored on spawn and can be updated") {
    WorldFixture f;
    f.world.registerTexture("default", fakeSet(99));
    const vkt::ObjectId id = f.world.spawn(f.mh, f.th, glm::mat4{1.0f}, true, 7);
    CHECK(id != vkt::INVALID_HANDLE);
    CHECK(f.world.getMaterialId(id) == 7u);

    f.world.setMaterialId(id, 11);
    CHECK(f.world.getMaterialId(id) == 11u);
}

// -- Transform & world position ----------------------------------------------

TEST_CASE("RenderWorld -- setWorldPosition / getWorldPosition roundtrip") {
    WorldFixture f;
    const vkt::ObjectId id = f.world.spawn(f.mh, f.th);

    // Use an astronomically large value to confirm double precision is preserved.
    const glm::dvec3 pos{1.495978707e11, -3.0856e16, 0.0}; // ~1 AU and ~1 pc
    f.world.setWorldPosition(id, pos);

    const glm::dvec3 got = f.world.getWorldPosition(id);
    CHECK(std::abs(got.x - pos.x) < 1.0);  // better than float precision (~1e3 m error)
    CHECK(std::abs(got.y - pos.y) < 1.0);
}

TEST_CASE("RenderWorld -- setTransform rotation/scale survives roundtrip (lossy translation)") {
    WorldFixture f;
    const vkt::ObjectId id = f.world.spawn(f.mh, f.th);

    const glm::mat4 t = glm::mat4{1.0f}; // identity
    f.world.setTransform(id, t);

    // getTransform() re-packs worldPosition as float -- confirmed lossy for large coords.
    // For identity / zero translation the roundtrip should be exact.
    const glm::mat4 got = f.world.getTransform(id);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            CHECK(std::abs(got[c][r] - t[c][r]) < 1e-6f);
}

TEST_CASE("RenderWorld -- statistics: meshCount textureCount objectCount") {
    vkt::RenderWorld world;
    CHECK(world.meshCount()    == 0);
    CHECK(world.textureCount() == 0);
    CHECK(world.objectCount()  == 0);

    world.registerMesh("m1", vkt::Mesh{});
    world.registerMesh("m2", vkt::Mesh{});
    world.registerTexture("t1", fakeSet(1));
    CHECK(world.meshCount()    == 2);
    CHECK(world.textureCount() == 1);

    world.destroy();
}
