#pragma once

// Async GPU texture streaming (Track C: C1-C4).
//
// Three-stage pipeline:
//   Stage 1  -- file I/O + pixel decode on worker threads (ThreadPool).
//   Stage 2  -- staging buffer fill + GPU transfer submit (main thread, tick()).
//   Stage 3  -- upload fence poll + resource promotion to Ready (main thread, tick()).
//
// Usage (init):
//   loader.init(device, physDevice, allocator,
//               graphicsQueue, graphicsQueueIndex,
//               placeholderView, placeholderSampler);
//
// Usage (per-frame, before recording commands):
//   loader.tick();
//
// Usage (request):
//   StreamHandle h = loader.request("assets/albedo.png");
//   if (h == kInvalidStream) { /* queue full -- backpressure */ }
//
// Usage (poll each frame after tick()):
//   auto r = loader.result(h);    // r.view/sampler always valid (placeholder or final)
//   auto c = loader.counters();   // c.queued, c.uploading, c.completed, ...

#include "resources/Buffer.h"
#include "resources/Image.h"
#include "resources/Sampler.h"
#include "utils/ThreadPool.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkt {

using StreamHandle = uint32_t;
constexpr StreamHandle kInvalidStream = ~0u;

// Lifecycle states for a streaming request.
enum class StreamStatus : uint8_t {
    Queued,     // submitted, waiting for a worker thread
    Loading,    // file I/O + pixel decode on a worker thread
    Uploading,  // GPU transfer submitted, fence pending
    Ready,      // GPU texture available; result() returns the final image
    Failed,     // decode or GPU upload error
    Cancelled,  // cancelled before the GPU upload started
};

// Per-request query result (C3: placeholders + C4: status).
// view and sampler are always non-null: they point to the placeholder until
// status == Ready, at which point they point to the final GPU texture.
struct StreamResult {
    StreamStatus status        = StreamStatus::Queued;
    VkImageView  view          = VK_NULL_HANDLE;
    VkSampler    sampler       = VK_NULL_HANDLE;
    bool         isPlaceholder = true;
};

class StreamLoader {
public:
    // Color space for decoded images (matches Texture::ColorSpace semantics).
    enum class ColorSpace { Srgb, Linear };

    struct Config {
        // Backpressure cap: request() returns kInvalidStream when
        // (queued + loading + uploading) >= maxQueueDepth.
        uint32_t maxQueueDepth = 32;
        // Number of parallel IO + decode worker threads.
        uint32_t workerThreads = 2;
    };

    StreamLoader() = default;
    ~StreamLoader();
    StreamLoader(const StreamLoader&)            = delete;
    StreamLoader& operator=(const StreamLoader&) = delete;

    // Initialize the loader.
    // placeholderView and placeholderSampler must remain valid until destroy().
    void init(VkDevice device, VkPhysicalDevice physDevice,
              VmaAllocator allocator,
              VkQueue uploadQueue, uint32_t queueFamilyIndex,
              VkImageView placeholderView, VkSampler placeholderSampler,
              const Config& cfg = {});

    // Drain all pending uploads, wait for in-flight GPU fences, release resources.
    void destroy();

    // Submit an async texture load request.
    // Returns kInvalidStream if the backpressure cap is reached.
    StreamHandle request(const std::string& path,
                         ColorSpace colorSpace = ColorSpace::Srgb);

    // Cancel a Queued or Loading job. No-op if Uploading or done.
    void cancel(StreamHandle handle);

    // Query the current status without blocking.
    StreamStatus status(StreamHandle handle) const;

    // Get the current view + sampler for the request.
    // Always returns non-null handles (placeholder or final).
    StreamResult result(StreamHandle handle) const;

    // Main-thread pump -- call once per frame before recording commands.
    // Promotes decoded results to GPU uploads and retires completed fences.
    void tick();

    // Streaming counters for the debug UI (C4).
    struct Counters {
        uint32_t queued    = 0;
        uint32_t loading   = 0;
        uint32_t uploading = 0;
        uint32_t completed = 0;
        uint32_t failed    = 0;
        uint32_t cancelled = 0;
    };
    Counters counters() const;

private:
    // Decoded pixel data produced by a worker thread; consumed by tick().
    struct DecodeResult {
        StreamHandle         handle    = kInvalidStream;
        std::vector<uint8_t> pixels;
        int                  width     = 0;
        int                  height    = 0;
        VkFormat             format    = VK_FORMAT_UNDEFINED;
        uint32_t             mipLevels = 1;
        bool                 failed    = false;
        std::string          error;
    };

    // A GPU upload in flight (main thread only; no mutex needed).
    struct UploadJob {
        StreamHandle           handle    = kInvalidStream;
        VkFence                fence     = VK_NULL_HANDLE;
        VkCommandPool          pool      = VK_NULL_HANDLE;
        Buffer                 staging;
        std::unique_ptr<Image> image;
        uint32_t               mipLevels = 1;
    };

    // Per-request metadata -- guarded by m_jobsMutex.
    struct JobMeta {
        StreamStatus             status    = StreamStatus::Queued;
        ColorSpace               colorSpace = ColorSpace::Srgb;
        std::unique_ptr<Image>   image;    // non-null when Ready
        std::unique_ptr<Sampler> sampler;  // non-null when Ready
    };

    // Record upload commands, submit with a fence, and push to m_uploading.
    void submitUpload(DecodeResult& dec);

    // Signal that a fence has fired: destroy transient resources, promote to Ready.
    void retireUpload(UploadJob& job);

    VkDevice         m_device           = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice       = VK_NULL_HANDLE;
    VmaAllocator     m_allocator        = VK_NULL_HANDLE;
    VkQueue          m_uploadQueue      = VK_NULL_HANDLE;
    uint32_t         m_queueFamily      = 0;
    VkImageView      m_placeholderView  = VK_NULL_HANDLE;
    VkSampler        m_placeholderSampler = VK_NULL_HANDLE;
    Config           m_cfg;

    std::unique_ptr<ThreadPool> m_pool;

    // Active job count for backpressure (queued + loading + uploading).
    std::atomic<uint32_t> m_activeCount{0};

    // Monotonically increasing handle counter.
    std::atomic<StreamHandle> m_nextHandle{0};

    // Job metadata -- guarded by m_jobsMutex.
    mutable std::mutex                        m_jobsMutex;
    std::unordered_map<StreamHandle, JobMeta> m_jobs;

    // Decode-ready results: produced by workers, consumed by tick() on the main thread.
    std::mutex                m_decodeMutex;
    std::vector<DecodeResult> m_decodeReady;

    // In-flight GPU uploads -- main thread only.
    std::vector<UploadJob> m_uploading;

    // Streaming counters (C4).
    std::atomic<uint32_t> m_cntQueued{0};
    std::atomic<uint32_t> m_cntLoading{0};
    std::atomic<uint32_t> m_cntUploading{0};
    std::atomic<uint32_t> m_cntCompleted{0};
    std::atomic<uint32_t> m_cntFailed{0};
    std::atomic<uint32_t> m_cntCancelled{0};
};

} // namespace vkt
