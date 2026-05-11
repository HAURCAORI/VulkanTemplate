#pragma once

#include "Image.h"
#include "Sampler.h"
#include "Buffer.h"

#include <string>
#include <mutex>

namespace vkt {

// GPU texture loaded from a file (PNG / JPG / BMP / TGA via stb_image).
// Always decoded as RGBA8; the GPU format is VK_FORMAT_R8G8B8A8_SRGB for color
// textures and VK_FORMAT_R8G8B8A8_UNORM for linear data such as normal maps.
// Automatically generates a full mip chain when the device supports linear
// blitting for the format; falls back to a single mip level otherwise.
//
// Staging buffer lifetime: the caller must keep outStaging alive until the
// upload command buffer has finished executing on the GPU.
//
// Usage:
//   Buffer staging;
//   Texture tex;
//   tex.loadFromFile(allocator, device, physDevice, uploadCmd, staging, "albedo.png");
//   // submit uploadCmd, wait idle (or fence), then:
//   staging.destroy();
//
//   DescriptorWriter{}.writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
//       tex.view(), tex.sampler(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
//       .update(device, materialSet);
class Texture {
public:
    enum class ColorSpace {
        Srgb,
        Linear,
    };

    Texture() = default;
    ~Texture();
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    // Load an RGBA8 image from disk, record the upload + mip generation into cmd.
    // outStaging must remain alive until the command buffer has been submitted and
    // the GPU has finished executing it (e.g. after vkQueueWaitIdle).
    void loadFromFile(VmaAllocator allocator, VkDevice device, VkPhysicalDevice physDevice,
                      VkCommandBuffer cmd, Buffer& outStaging,
                      const std::string& path,
                      ColorSpace colorSpace = ColorSpace::Srgb);

    void destroy();

    bool        valid()     const { return m_image.handle() != VK_NULL_HANDLE; }
    VkImageView view()      const { return m_image.view(); }
    VkSampler   sampler()   const { return m_sampler.handle(); }
    uint32_t    width()     const { return m_image.width(); }
    uint32_t    height()    const { return m_image.height(); }
    uint32_t    mipLevels() const { return m_image.mipLevels(); }

private:
    Image   m_image;
    Sampler m_sampler;
    mutable std::mutex m_mutex;
};

} // namespace vkt
