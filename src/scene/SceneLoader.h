#pragma once

#include "RenderWorld.h"
#include "material/Material.h"
#include "pipeline/BindlessTextureSet.h"
#include "pipeline/Descriptors.h"
#include "resources/Buffer.h"
#include "resources/Texture.h"

#include <string>
#include <vector>
#include <memory>

namespace vkt {

// Dependencies passed to SceneLoader::load; all pointers must remain valid
// for the duration of the call.
struct SceneLoadContext {
    VmaAllocator          allocator;
    VkDevice              device;
    VkPhysicalDevice      physDevice;
    VkCommandBuffer       uploadCmd;    // GPU upload commands recorded here
    DescriptorPool*       descriptorPool;
    VkDescriptorSetLayout materialLayout;
    MaterialManager*      materials = nullptr;
    // Default white albedo texture for legacy objects and materials without an explicit texture.
    VkImageView           defaultAlbedoView    = VK_NULL_HANDLE;
    VkSampler             defaultAlbedoSampler = VK_NULL_HANDLE;
    // Default flat normal map for material sets that have no real normal texture.
    // Set to VK_NULL_HANDLE to skip writing binding 1 (backward compatibility).
    VkImageView           defaultNormalView    = VK_NULL_HANDLE;
    VkSampler             defaultNormalSampler = VK_NULL_HANDLE;
    // Optional bindless texture set (Track G). When non-null, loaded textures are
    // also registered into the bindless array and GPUMaterial tex indices are filled.
    // Null = use per-material descriptor sets only (non-bindless fallback path).
    BindlessTextureSet*   bindlessSet              = nullptr;
    uint32_t              defaultAlbedoBindlessIdx = 0;  // bindless index of default white
    uint32_t              defaultNormalBindlessIdx = 0;  // bindless index of default normal
};

// Populates a RenderWorld from a JSON scene description file.
//
// JSON schema:
//   {
//     "meshes": [
//       {"name": "sphere", "type": "sphere", "rings": 32, "sectors": 64},
//       {"name": "cube",   "type": "cube"},
//       {"name": "ground", "type": "quad"}
//     ],
//     "textures": [
//       {"name": "stone", "path": "assets/textures/stone.png"},
//       {"name": "stone_n", "path": "assets/textures/stone_n.png", "colorSpace": "linear"}
//     ],
//     "materials": [
//       {"name": "stone_pbr", "albedoTexture": "stone", "normalTexture": "stone_n",
//        "roughness": 0.7, "metallic": 0.0, "usePBR": true}
//     ],
//     "objects": [
//       {"mesh": "sphere", "material": "stone_pbr",
//        "position": [0, 0.5, 0], "rotation": [-90, 0, 0], "scale": [1, 1, 1]}
//     ]
//   }
//
// Built-in mesh types: "sphere" (optional "rings"/"sectors"), "cube", "quad".
// Object rotation is Euler degrees applied Y then X then Z.
// Object "scale" may be a single number (uniform) or a [x, y, z] array.
// Texture entries default to sRGB; set "colorSpace": "linear" for data textures
// such as normal maps. Material entries are optional; objects without a material
// still use materialId 0 and the legacy "texture" field.
//
// All GPU uploads are recorded into ctx.uploadCmd; the caller must submit and
// wait-idle before using the scene.  Staging buffers in outStaging and Texture
// objects in outTextures must both remain alive until the upload completes.
class SceneLoader {
public:
    static void load(const SceneLoadContext&                ctx,
                     RenderWorld&                           world,
                     const std::string&                     path,
                     std::vector<Buffer>&                   outStaging,
                     std::vector<std::unique_ptr<Texture>>& outTextures);
};

} // namespace vkt
