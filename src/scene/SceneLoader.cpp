#include "SceneLoader.h"
#include "pipeline/Descriptors.h"

#include <nlohmann/json.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <unordered_map>

namespace vkt {

using json = nlohmann::json;

namespace {

std::string typeName(const json& v) {
    if (v.is_null()) return "null";
    if (v.is_boolean()) return "bool";
    if (v.is_number()) return "number";
    if (v.is_string()) return "string";
    if (v.is_array()) return "array";
    if (v.is_object()) return "object";
    return "unknown";
}

const json& requireField(const json& obj, const char* key, const char* where) {
    if (!obj.is_object() || !obj.contains(key)) {
        throw std::runtime_error(std::string("[SceneLoader] Missing '") + key + "' in " + where);
    }
    return obj[key];
}

std::string requireString(const json& obj, const char* key, const char* where) {
    const auto& v = requireField(obj, key, where);
    if (!v.is_string()) {
        throw std::runtime_error(std::string("[SceneLoader] '") + key + "' must be string in " + where +
                                 ", got " + typeName(v));
    }
    return v.get<std::string>();
}

glm::vec3 readVec3(const json& obj, const char* key,
                  glm::vec3 def = {0.0f, 0.0f, 0.0f}) {
    if (!obj.contains(key)) return def;
    const auto& v = obj[key];
    if (!v.is_array() || v.size() != 3 ||
        !v[0].is_number() || !v[1].is_number() || !v[2].is_number()) {
        throw std::runtime_error(std::string("[SceneLoader] '") + key +
                                 "' must be [x,y,z] numbers");
    }
    return {v[0].get<float>(), v[1].get<float>(), v[2].get<float>()};
}

glm::vec4 readVec4(const json& obj, const char* key,
                  glm::vec4 def = {1.0f, 1.0f, 1.0f, 1.0f}) {
    if (!obj.contains(key)) return def;
    const auto& v = obj[key];
    if (!v.is_array() || v.size() != 4 ||
        !v[0].is_number() || !v[1].is_number() || !v[2].is_number() || !v[3].is_number()) {
        throw std::runtime_error(std::string("[SceneLoader] '") + key +
                                 "' must be [x,y,z,w] numbers");
    }
    return {v[0].get<float>(), v[1].get<float>(), v[2].get<float>(), v[3].get<float>()};
}

glm::vec3 readScale(const json& obj) {
    if (!obj.contains("scale")) return {1.0f, 1.0f, 1.0f};
    const auto& s = obj["scale"];
    if (s.is_number()) {
        return glm::vec3(s.get<float>());
    }
    if (s.is_array() && s.size() == 3 &&
        s[0].is_number() && s[1].is_number() && s[2].is_number()) {
        return {s[0].get<float>(), s[1].get<float>(), s[2].get<float>()};
    }
    throw std::runtime_error("[SceneLoader] 'scale' must be a number or [x,y,z] numbers");
}

// Build a world-space transform from position / rotation (Euler degrees, Y->X->Z) / scale.
glm::mat4 buildTransform(const json& j) {
    const glm::vec3 pos = readVec3(j, "position");
    const glm::vec3 rot = readVec3(j, "rotation");
    const glm::vec3 scl = readScale(j);

    glm::mat4 t = glm::translate(glm::mat4{1.0f}, pos);
    t = glm::rotate(t, glm::radians(rot.y), {0.0f, 1.0f, 0.0f});
    t = glm::rotate(t, glm::radians(rot.x), {1.0f, 0.0f, 0.0f});
    t = glm::rotate(t, glm::radians(rot.z), {0.0f, 0.0f, 1.0f});
    t = glm::scale(t, scl);
    return t;
}

Texture::ColorSpace readColorSpace(const json& obj, const char* where) {
    const std::string colorSpace = obj.value("colorSpace", "srgb");
    if (colorSpace == "srgb" || colorSpace == "sRGB")
        return Texture::ColorSpace::Srgb;
    if (colorSpace == "linear")
        return Texture::ColorSpace::Linear;
    throw std::runtime_error("[SceneLoader] Unsupported colorSpace '" + colorSpace +
                             "' in " + where);
}

} // namespace

void SceneLoader::load(const SceneLoadContext&                ctx,
                       RenderWorld&                           world,
                       const std::string&                     path,
                       std::vector<Buffer>&                   outStaging,
                       std::vector<std::unique_ptr<Texture>>& outTextures) {
    // ctx.uploadCmd must be a valid recording command buffer on the graphics queue.
    // outStaging accumulates all staging buffers; the caller must keep outStaging
    // alive until after vkQueueSubmit completes (fence or vkQueueWaitIdle) before
    // destroying it. Textures in outTextures must outlive the RenderWorld.
    std::ifstream file(path);
    if (!file.good())
        throw std::runtime_error("[SceneLoader] Cannot open: " + path);

    // allow_exceptions=true (default), ignore_comments=true for // and /* */
    const json j = json::parse(file, nullptr, true, true);

    struct TextureBinding {
        VkImageView   view        = VK_NULL_HANDLE;
        VkSampler     sampler     = VK_NULL_HANDLE;
        TextureHandle handle      = INVALID_HANDLE;
        uint32_t      bindlessIdx = 0;  // index into bindlessSet array (Track G)
    };

    struct MaterialBinding {
        uint32_t      materialId    = MaterialManager::kDefaultId;
        TextureHandle textureHandle = INVALID_HANDLE;
        std::string   albedoName    = "default";
        std::string   normalName    = "default_normal";
    };

    std::unordered_map<std::string, TextureBinding> texturesByName;
    std::unordered_map<std::string, TextureHandle> materialTexturePairs;
    std::unordered_map<std::string, MaterialBinding> materialsByName;

    if (ctx.defaultAlbedoView != VK_NULL_HANDLE && ctx.defaultAlbedoSampler != VK_NULL_HANDLE) {
        TextureBinding binding;
        binding.view        = ctx.defaultAlbedoView;
        binding.sampler     = ctx.defaultAlbedoSampler;
        binding.handle      = world.findTexture("default");
        binding.bindlessIdx = ctx.defaultAlbedoBindlessIdx;
        texturesByName.emplace("default", binding);
    }

    auto resolveTextureBinding =
        [&](const std::string& name, bool allowMissingDefault, const char* where) -> TextureBinding {
            if (name.empty()) {
                return allowMissingDefault ? TextureBinding{} : texturesByName.at("default");
            }
            if (auto it = texturesByName.find(name); it != texturesByName.end())
                return it->second;
            if (allowMissingDefault)
                return {};
            throw std::runtime_error("[SceneLoader] Unknown texture '" + name + "' in " + where);
        };

    auto resolveMaterialTexture =
        [&](const std::string& materialName,
            const TextureBinding& albedo,
            const TextureBinding& normal,
            const std::string& albedoName,
            const std::string& normalName) -> TextureHandle {
            if (albedo.handle == INVALID_HANDLE) {
                throw std::runtime_error("[SceneLoader] Material '" + materialName +
                                         "' has no valid albedo texture binding");
            }

            const bool usesDefaultPair =
                albedo.handle == world.findTexture("default") &&
                (normal.view == VK_NULL_HANDLE || normal.view == ctx.defaultNormalView);
            if (usesDefaultPair)
                return albedo.handle;

            const std::string pairKey = albedoName + "|" + normalName;
            if (auto it = materialTexturePairs.find(pairKey); it != materialTexturePairs.end())
                return it->second;

            if (!ctx.descriptorPool || ctx.materialLayout == VK_NULL_HANDLE)
                throw std::runtime_error("[SceneLoader] Material texture pair requires descriptor context");

            const VkDescriptorSet set = ctx.descriptorPool->allocate(ctx.materialLayout);
            DescriptorWriter writer;
            writer.writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              albedo.view, albedo.sampler,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writer.writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              normal.view != VK_NULL_HANDLE ? normal.view : ctx.defaultNormalView,
                              normal.sampler != VK_NULL_HANDLE ? normal.sampler : ctx.defaultNormalSampler,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writer.update(ctx.device, set);

            const std::string worldName = "__material_pair__" + materialName;
            const TextureHandle handle = world.registerTexture(worldName, set);
            materialTexturePairs.emplace(pairKey, handle);
            return handle;
        };

    // -- Meshes ----------------------------------------------------------------
    if (j.contains("meshes")) {
        if (!j["meshes"].is_array())
            throw std::runtime_error("[SceneLoader] 'meshes' must be an array");

        for (size_t i = 0; i < j["meshes"].size(); ++i) {
            const auto& m = j["meshes"][i];
            if (!m.is_object())
                throw std::runtime_error("[SceneLoader] mesh entry must be an object");

            const std::string where = "meshes[" + std::to_string(i) + "]";
            const std::string name = requireString(m, "name", where.c_str());
            if (world.findMesh(name) != INVALID_HANDLE) continue;

            const std::string type = m.value("type", "sphere");
            if (type != "sphere" && type != "cube" && type != "quad") {
                throw std::runtime_error("[SceneLoader] Unsupported mesh type '" + type + "' in " + where);
            }

            auto [verts, indices] = [&]() -> std::pair<std::vector<Vertex>, std::vector<uint32_t>> {
                if (type == "cube") return Mesh::makeCube();
                if (type == "quad") return Mesh::makeQuad();
                const uint32_t rings   = m.value("rings",   16u);
                const uint32_t sectors = m.value("sectors", 32u);
                if (rings < 2u) {
                    throw std::runtime_error("[SceneLoader] 'rings' must be >= 2 in " + where);
                }
                if (sectors < 3u) {
                    throw std::runtime_error("[SceneLoader] 'sectors' must be >= 3 in " + where);
                }
                return Mesh::makeUVSphere(rings, sectors);
            }();

            Buffer vs, is;
            Mesh mesh;
            mesh.upload(ctx.allocator, ctx.uploadCmd, verts, indices, vs, is);
            outStaging.push_back(std::move(vs));
            outStaging.push_back(std::move(is));
            world.registerMesh(name, std::move(mesh));
        }
    }

    // -- Textures --------------------------------------------------------------
    if (j.contains("textures")) {
        if (!j["textures"].is_array())
            throw std::runtime_error("[SceneLoader] 'textures' must be an array");

        for (size_t i = 0; i < j["textures"].size(); ++i) {
            const auto& t = j["textures"][i];
            if (!t.is_object())
                throw std::runtime_error("[SceneLoader] texture entry must be an object");

            const std::string where = "textures[" + std::to_string(i) + "]";
            const std::string name = requireString(t, "name", where.c_str());
            const std::string texPath = t.value("path", "");
            if (world.findTexture(name) != INVALID_HANDLE) continue;
            if (texPath.empty()) continue; // caller-registered texture, e.g. "default"

            auto tex = std::make_unique<Texture>();
            Buffer stg;
            const auto colorSpace = readColorSpace(t, where.c_str());
            tex->loadFromFile(ctx.allocator, ctx.device, ctx.physDevice,
                              ctx.uploadCmd, stg, texPath, colorSpace);
            outStaging.push_back(std::move(stg));

            const VkDescriptorSet set = ctx.descriptorPool->allocate(ctx.materialLayout);
            {
                DescriptorWriter writer;
                writer.writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  tex->view(), tex->sampler(),
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                if (ctx.defaultNormalView != VK_NULL_HANDLE) {
                    writer.writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      ctx.defaultNormalView, ctx.defaultNormalSampler,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }
                writer.update(ctx.device, set);
            }

            const TextureHandle handle = world.registerTexture(name, set);
            TextureBinding binding;
            binding.view    = tex->view();
            binding.sampler = tex->sampler();
            binding.handle  = handle;
            // G2: register into bindless array if available.
            if (ctx.bindlessSet) {
                binding.bindlessIdx =
                    ctx.bindlessSet->registerTexture(ctx.device, tex->view(), tex->sampler());
            }
            texturesByName.emplace(name, binding);
            outTextures.push_back(std::move(tex));
        }
    }

    // -- Materials -------------------------------------------------------------
    if (j.contains("materials")) {
        if (!j["materials"].is_array())
            throw std::runtime_error("[SceneLoader] 'materials' must be an array");
        if (!ctx.materials)
            throw std::runtime_error("[SceneLoader] MaterialManager context is required for 'materials'");

        for (size_t i = 0; i < j["materials"].size(); ++i) {
            const auto& m = j["materials"][i];
            if (!m.is_object())
                throw std::runtime_error("[SceneLoader] material entry must be an object");

            const std::string where = "materials[" + std::to_string(i) + "]";
            const std::string name = requireString(m, "name", where.c_str());

            GPUMaterial gpuMat{};
            gpuMat.baseColor = readVec4(m, "baseColor", gpuMat.baseColor);
            gpuMat.roughness = m.value("roughness", gpuMat.roughness);
            gpuMat.metallic  = m.value("metallic",  gpuMat.metallic);
            gpuMat.emissive  = m.value("emissive",  gpuMat.emissive);

            if (m.value("unlit", false))     gpuMat.flags |= MaterialFlags::Unlit;
            if (m.value("alphaTest", false)) gpuMat.flags |= MaterialFlags::AlphaTest;
            if (m.value("usePBR", false))    gpuMat.flags |= MaterialFlags::UsePBR;

            std::string albedoName = m.value("albedoTexture", m.value("texture", std::string("default")));
            std::string normalName = m.value("normalTexture", std::string{});

            const TextureBinding albedo = resolveTextureBinding(albedoName, false, where.c_str());
            TextureBinding normal;
            if (!normalName.empty()) {
                normal = resolveTextureBinding(normalName, true, where.c_str());
                if (normal.view == VK_NULL_HANDLE) {
                    throw std::runtime_error("[SceneLoader] Unknown normal texture '" + normalName +
                                             "' in " + where);
                }
                gpuMat.flags |= MaterialFlags::HasNormalMap;
            } else {
                normalName = "default_normal";
                normal.view = ctx.defaultNormalView;
                normal.sampler = ctx.defaultNormalSampler;
            }

            const TextureHandle materialTexHandle =
                resolveMaterialTexture(name, albedo, normal, albedoName, normalName);

            // G2: write bindless texture indices if bindless set is available.
            if (ctx.bindlessSet) {
                gpuMat.albedoTexIdx = albedo.bindlessIdx;
                // Use the normal texture's bindless index if it was loaded from file.
                // Fall back to the default normal index otherwise.
                const bool useDefaultNormal =
                    (normal.view == VK_NULL_HANDLE || normal.view == ctx.defaultNormalView);
                gpuMat.normalTexIdx =
                    useDefaultNormal ? ctx.defaultNormalBindlessIdx : normal.bindlessIdx;
            }

            MaterialBinding binding;
            binding.materialId    = ctx.materials->add(name, gpuMat);
            binding.textureHandle = materialTexHandle;
            binding.albedoName    = albedoName;
            binding.normalName    = normalName;
            materialsByName[name] = binding;
        }
    }

    // -- Objects ---------------------------------------------------------------
    if (j.contains("objects")) {
        if (!j["objects"].is_array())
            throw std::runtime_error("[SceneLoader] 'objects' must be an array");

        for (size_t i = 0; i < j["objects"].size(); ++i) {
            const auto& o = j["objects"][i];
            if (!o.is_object())
                throw std::runtime_error("[SceneLoader] object entry must be an object");

            const std::string meshName = o.value("mesh", "");
            const std::string texName  = o.value("texture", "default");
            const std::string materialName = o.value("material", "");

            const MeshHandle mh = world.findMesh(meshName);
            TextureHandle    th = world.findTexture(texName);
            if (th == INVALID_HANDLE) th = world.findTexture("default");
            uint32_t         materialId = MaterialManager::kDefaultId;

            if (!materialName.empty()) {
                const auto it = materialsByName.find(materialName);
                if (it == materialsByName.end()) {
                    std::fprintf(stderr, "[SceneLoader] Unknown material '%s', using defaults\n",
                                 materialName.c_str());
                } else {
                    materialId = it->second.materialId;
                    th         = it->second.textureHandle;
                }
            }

            if (mh == INVALID_HANDLE) {
                std::fprintf(stderr, "[SceneLoader] Unknown mesh '%s', skipping\n",
                             meshName.c_str());
                continue;
            }

            const bool      rigidBody   = o.value("rigidBody", true);
            const float     boundRadius = o.value("boundRadius", 0.0f);
            const glm::vec3 boundCenter = readVec3(o, "boundCenter");

            const ObjectId id = world.spawn(mh, th, buildTransform(o), rigidBody, materialId);
            if (id == INVALID_HANDLE) {
                std::fprintf(stderr, "[SceneLoader] Failed to spawn object at index %zu, skipping\n", i);
                continue;
            }
            if (boundRadius > 0.0f)
                world.setBoundingSphere(id, boundCenter, boundRadius);
        }
    }
}

} // namespace vkt
