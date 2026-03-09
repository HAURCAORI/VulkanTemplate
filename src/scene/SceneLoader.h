#pragma once

#include "RenderWorld.h"
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
//       {"name": "stone", "path": "assets/textures/stone.png"}
//     ],
//     "objects": [
//       {"mesh": "sphere", "texture": "stone",
//        "position": [0, 0.5, 0], "rotation": [-90, 0, 0], "scale": [1, 1, 1]}
//     ]
//   }
//
// Built-in mesh types: "sphere" (optional "rings"/"sectors"), "cube", "quad".
// Object rotation is Euler degrees applied Y then X then Z.
// Object "scale" may be a single number (uniform) or a [x, y, z] array.
// Texture entries without a "path" key are skipped (use "default" pre-registered
// by the caller).
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
