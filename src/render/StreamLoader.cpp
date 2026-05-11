#include "render/StreamLoader.h"

// stb_image.h -- do NOT define STB_IMAGE_IMPLEMENTATION here.
// The implementation is already compiled in Texture.cpp (exactly one TU).
#include <stb_image.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace vkt {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

StreamLoader::~StreamLoader() { destroy(); }

void StreamLoader::init(VkDevice device, VkPhysicalDevice physDevice,
                        VmaAllocator allocator,
                        VkQueue uploadQueue, uint32_t queueFamilyIndex,
                        VkImageView placeholderView, VkSampler placeholderSampler,
                        const Config& cfg) {
    assert(device != VK_NULL_HANDLE);
    assert(m_device == VK_NULL_HANDLE && "StreamLoader::init called twice");

    m_device            = device;
    m_physDevice        = physDevice;
    m_allocator         = allocator;
    m_uploadQueue       = uploadQueue;
    m_queueFamily       = queueFamilyIndex;
    m_placeholderView   = placeholderView;
    m_placeholderSampler = placeholderSampler;
    m_cfg               = cfg;

    const uint32_t threads = (cfg.workerThreads > 0) ? cfg.workerThreads : 1u;
    m_pool = std::make_unique<ThreadPool>(threads);
}

void StreamLoader::destroy() {
    if (m_device == VK_NULL_HANDLE) return;

    // Stop accepting new work; join all worker threads (blocks until decode done).
    m_pool.reset();

    // Discard decoded results that haven't been uploaded yet.
    {
        std::scoped_lock lock(m_decodeMutex);
        m_decodeReady.clear();
    }

    // Retire all in-flight uploads -- wait for their GPU fences.
    for (auto& job : m_uploading) {
        if (job.fence != VK_NULL_HANDLE) {
            vkWaitForFences(m_device, 1, &job.fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(m_device, job.fence, nullptr);
        }
        if (job.pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(m_device, job.pool, nullptr);
        job.staging.destroy();
        // job.image is destroyed by its unique_ptr destructor.
    }
    m_uploading.clear();

    // Clear all job metadata; Image + Sampler unique_ptrs are destroyed here.
    {
        std::scoped_lock lock(m_jobsMutex);
        m_jobs.clear();
    }

    m_device = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Request / cancel
// ---------------------------------------------------------------------------

StreamHandle StreamLoader::request(const std::string& path, ColorSpace colorSpace) {
    assert(m_device != VK_NULL_HANDLE && "StreamLoader not initialized");

    // Backpressure: reject new requests when the active queue is full.
    if (m_activeCount.load(std::memory_order_relaxed) >= m_cfg.maxQueueDepth)
        return kInvalidStream;

    const StreamHandle h = m_nextHandle.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock(m_jobsMutex);
        JobMeta& meta  = m_jobs[h];
        meta.status    = StreamStatus::Queued;
        meta.colorSpace = colorSpace;
    }
    m_activeCount.fetch_add(1, std::memory_order_relaxed);
    m_cntQueued.fetch_add(1, std::memory_order_relaxed);

    // Capture by value -- the worker thread outlives this call.
    m_pool->submit([this, h, path, colorSpace] {
        // Transition Queued -> Loading (cancel check #1).
        {
            std::scoped_lock lock(m_jobsMutex);
            auto it = m_jobs.find(h);
            if (it == m_jobs.end() || it->second.status == StreamStatus::Cancelled) {
                // Cancelled before the worker even started.
                m_activeCount.fetch_sub(1, std::memory_order_relaxed);
                m_cntQueued.fetch_sub(1, std::memory_order_relaxed);
                m_cntCancelled.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            it->second.status = StreamStatus::Loading;
        }
        m_cntQueued.fetch_sub(1, std::memory_order_relaxed);
        m_cntLoading.fetch_add(1, std::memory_order_relaxed);

        // Stage 1: file I/O + pixel decode.
        DecodeResult dec;
        dec.handle = h;

        int w = 0, hPx = 0, srcCh = 0;
        stbi_uc* pixels = stbi_load(path.c_str(), &w, &hPx, &srcCh, STBI_rgb_alpha);
        if (!pixels) {
            dec.failed = true;
            dec.error  = stbi_failure_reason();
        } else {
            const VkFormat fmt = (colorSpace == ColorSpace::Srgb)
                ? VK_FORMAT_R8G8B8A8_SRGB
                : VK_FORMAT_R8G8B8A8_UNORM;

            // Query mipmap blit support on the worker thread.
            // vkGetPhysicalDeviceFormatProperties is externally synchronized per
            // VkPhysicalDevice but is read-only; concurrent calls are safe.
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(m_physDevice, fmt, &props);
            const bool supportsLinearBlit =
                (props.optimalTilingFeatures &
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;

            dec.width     = w;
            dec.height    = hPx;
            dec.format    = fmt;
            dec.mipLevels = supportsLinearBlit
                ? Image::calcMipLevels(static_cast<uint32_t>(w),
                                       static_cast<uint32_t>(hPx))
                : 1u;

            const VkDeviceSize imageSize =
                static_cast<VkDeviceSize>(w) * hPx * 4;
            dec.pixels.resize(static_cast<size_t>(imageSize));
            std::memcpy(dec.pixels.data(), pixels, static_cast<size_t>(imageSize));
            stbi_image_free(pixels);
        }

        // Cancel check #2: was the job cancelled during decode?
        {
            std::scoped_lock lock(m_jobsMutex);
            auto it = m_jobs.find(h);
            if (it == m_jobs.end() || it->second.status == StreamStatus::Cancelled) {
                m_cntLoading.fetch_sub(1, std::memory_order_relaxed);
                m_cntCancelled.fetch_add(1, std::memory_order_relaxed);
                m_activeCount.fetch_sub(1, std::memory_order_relaxed);
                return;
            }
        }
        m_cntLoading.fetch_sub(1, std::memory_order_relaxed);

        // Push to the decode-ready queue for main-thread Stage 2 processing.
        {
            std::scoped_lock lock(m_decodeMutex);
            m_decodeReady.push_back(std::move(dec));
        }
    });

    return h;
}

void StreamLoader::cancel(StreamHandle handle) {
    std::scoped_lock lock(m_jobsMutex);
    auto it = m_jobs.find(handle);
    if (it == m_jobs.end()) return;

    // Only Queued and Loading jobs can be cancelled; Uploading is committed to the GPU.
    const StreamStatus s = it->second.status;
    if (s == StreamStatus::Queued || s == StreamStatus::Loading)
        it->second.status = StreamStatus::Cancelled;
    // Counter updates happen in the worker thread (or tick()) when it observes this.
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

StreamStatus StreamLoader::status(StreamHandle handle) const {
    std::scoped_lock lock(m_jobsMutex);
    auto it = m_jobs.find(handle);
    if (it == m_jobs.end()) return StreamStatus::Failed;
    return it->second.status;
}

StreamResult StreamLoader::result(StreamHandle handle) const {
    std::scoped_lock lock(m_jobsMutex);
    auto it = m_jobs.find(handle);
    if (it == m_jobs.end())
        return {StreamStatus::Failed, m_placeholderView, m_placeholderSampler, true};

    const JobMeta& meta = it->second;
    if (meta.status == StreamStatus::Ready && meta.image && meta.sampler)
        return {StreamStatus::Ready, meta.image->view(), meta.sampler->handle(), false};

    return {meta.status, m_placeholderView, m_placeholderSampler, true};
}

StreamLoader::Counters StreamLoader::counters() const {
    Counters c;
    c.queued    = m_cntQueued   .load(std::memory_order_relaxed);
    c.loading   = m_cntLoading  .load(std::memory_order_relaxed);
    c.uploading = m_cntUploading.load(std::memory_order_relaxed);
    c.completed = m_cntCompleted.load(std::memory_order_relaxed);
    c.failed    = m_cntFailed   .load(std::memory_order_relaxed);
    c.cancelled = m_cntCancelled.load(std::memory_order_relaxed);
    return c;
}

// ---------------------------------------------------------------------------
// Tick (main thread)
// ---------------------------------------------------------------------------

void StreamLoader::tick() {
    // Stage 2: drain decoded results and submit GPU uploads.
    std::vector<DecodeResult> ready;
    {
        std::scoped_lock lock(m_decodeMutex);
        ready.swap(m_decodeReady);
    }

    for (auto& dec : ready) {
        // Check cancelled or failed before allocating any GPU resources.
        {
            std::scoped_lock lock(m_jobsMutex);
            auto it = m_jobs.find(dec.handle);
            if (it == m_jobs.end() || it->second.status == StreamStatus::Cancelled) {
                m_cntCancelled.fetch_add(1, std::memory_order_relaxed);
                m_activeCount.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }
        }
        if (dec.failed) {
            std::fprintf(stderr, "[StreamLoader] Failed to decode stream %u: %s\n",
                         dec.handle, dec.error.c_str());
            {
                std::scoped_lock lock(m_jobsMutex);
                auto it = m_jobs.find(dec.handle);
                if (it != m_jobs.end())
                    it->second.status = StreamStatus::Failed;
            }
            m_cntFailed.fetch_add(1, std::memory_order_relaxed);
            m_activeCount.fetch_sub(1, std::memory_order_relaxed);
            continue;
        }
        submitUpload(dec);
    }

    // Stage 3: poll in-flight upload fences; retire completed uploads.
    size_t i = 0;
    while (i < m_uploading.size()) {
        UploadJob& job = m_uploading[i];
        if (vkGetFenceStatus(m_device, job.fence) == VK_SUCCESS) {
            retireUpload(job);
            // Swap-erase for O(1) removal (order doesn't matter).
            if (i + 1 < m_uploading.size())
                m_uploading[i] = std::move(m_uploading.back());
            m_uploading.pop_back();
        } else {
            ++i;
        }
    }
}

void StreamLoader::submitUpload(DecodeResult& dec) {
    // Build a CPU-visible staging buffer and copy pixel data into it.
    Buffer staging;
    const VkDeviceSize dataSize =
        static_cast<VkDeviceSize>(dec.width) * dec.height * 4;
    staging.createCpuVisible(m_allocator, dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    staging.uploadData(dec.pixels.data(), dataSize);

    // Free the CPU-side pixel data now that it's in the staging buffer.
    dec.pixels.clear();
    dec.pixels.shrink_to_fit();

    // Create the device-local GPU image.
    auto image = std::make_unique<Image>();
    image->create(m_allocator, m_device, Image::CreateInfo{
        static_cast<uint32_t>(dec.width),
        static_cast<uint32_t>(dec.height),
        dec.format,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |  // mipmap blit source
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |  // staging copy destination
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        dec.mipLevels
    });

    // Allocate a transient command pool + command buffer for this upload.
    VkCommandPool pool = VK_NULL_HANDLE;
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_queueFamily;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
            throw std::runtime_error("[StreamLoader] Failed to create upload command pool");
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = pool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_device, &allocInfo, &cmd) != VK_SUCCESS) {
            vkDestroyCommandPool(m_device, pool, nullptr);
            throw std::runtime_error(
                "[StreamLoader] Failed to allocate upload command buffer");
        }
    }

    // Record: layout transition -> copy -> mip generation (or final transition).
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        Image::transitionLayout(cmd, image->handle(),
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                dec.mipLevels);
        Image::copyFromBuffer(cmd, staging.handle(), image->handle(),
                              static_cast<uint32_t>(dec.width),
                              static_cast<uint32_t>(dec.height));
        if (dec.mipLevels > 1) {
            // generateMipmaps leaves all levels in SHADER_READ_ONLY_OPTIMAL.
            Image::generateMipmaps(cmd, image->handle(),
                                   static_cast<uint32_t>(dec.width),
                                   static_cast<uint32_t>(dec.height),
                                   dec.mipLevels);
        } else {
            Image::transitionLayout(cmd, image->handle(),
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        vkEndCommandBuffer(cmd);
    }

    // Create an unsignaled per-upload fence (C2: staging buffer lifetime tracking).
    VkFence fence = VK_NULL_HANDLE;
    {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // No VK_FENCE_CREATE_SIGNALED_BIT -- we want to wait for completion.
        if (vkCreateFence(m_device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            vkDestroyCommandPool(m_device, pool, nullptr);
            throw std::runtime_error("[StreamLoader] Failed to create upload fence");
        }
    }

    // Submit the upload with the fence so we know when the GPU is done.
    {
        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &cmd;
        if (vkQueueSubmit(m_uploadQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
            vkDestroyFence(m_device, fence, nullptr);
            vkDestroyCommandPool(m_device, pool, nullptr);
            throw std::runtime_error("[StreamLoader] Failed to submit upload");
        }
    }

    // Update job status to Uploading.
    {
        std::scoped_lock lock(m_jobsMutex);
        auto it = m_jobs.find(dec.handle);
        if (it != m_jobs.end())
            it->second.status = StreamStatus::Uploading;
    }
    m_cntUploading.fetch_add(1, std::memory_order_relaxed);

    std::printf("[StreamLoader] Submitted GPU upload for stream %u "
                "(%dx%d, %u mip%s)\n",
                dec.handle, dec.width, dec.height,
                dec.mipLevels, dec.mipLevels == 1 ? "" : "s");

    // Store the in-flight job so tick() can retire it when the fence signals.
    UploadJob job;
    job.handle    = dec.handle;
    job.fence     = fence;
    job.pool      = pool;
    job.staging   = std::move(staging);
    job.image     = std::move(image);
    job.mipLevels = dec.mipLevels;
    m_uploading.push_back(std::move(job));
}

void StreamLoader::retireUpload(UploadJob& job) {
    // The fence has signaled -- the GPU is done reading the staging buffer.
    // Destroy the fence, command pool (implicitly frees the command buffer),
    // and staging buffer (C2: safe destruction after fence signals).
    vkDestroyFence(m_device, job.fence, nullptr);
    job.fence = VK_NULL_HANDLE;
    vkDestroyCommandPool(m_device, job.pool, nullptr);
    job.pool = VK_NULL_HANDLE;
    job.staging.destroy();

    // Create the sampler now that we know the final mip count.
    // Sampler creation is CPU-only (no command buffer needed).
    auto sampler = std::make_unique<Sampler>();
    sampler->create(m_device, m_physDevice,
                    Sampler::linearConfig(static_cast<float>(job.mipLevels)));

    // Promote the job to Ready -- from this tick() call onward, result()
    // returns the final view + sampler instead of the placeholder.
    {
        std::scoped_lock lock(m_jobsMutex);
        auto it = m_jobs.find(job.handle);
        if (it != m_jobs.end()) {
            it->second.status  = StreamStatus::Ready;
            it->second.image   = std::move(job.image);
            it->second.sampler = std::move(sampler);
        }
    }

    m_cntUploading.fetch_sub(1, std::memory_order_relaxed);
    m_cntCompleted.fetch_add(1, std::memory_order_relaxed);
    m_activeCount.fetch_sub(1, std::memory_order_relaxed);

    std::printf("[StreamLoader] Stream %u promoted to Ready\n", job.handle);
}

} // namespace vkt
