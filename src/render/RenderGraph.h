#pragma once

#include "utils/DebugUtils.h"

#include <volk.h>
#include <glm/glm.hpp>

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vkt {

// Forward declaration -- avoids including GpuTimestamps.h from every consumer of RenderGraph.h.
class GpuTimestamps;

// Lightweight frame render graph (Track B: B1-B3).
//
// Declares pass/resource dependencies, derives pipeline barriers automatically,
// wraps each pass with Vulkan debug labels, and prints the DAG on demand.
//
// Usage (setup, once or after resize):
//   ResourceId rBuf = graph.addBuffer({...});
//   ResourceId rImg = graph.addImage({...});
//   graph.addPass("Shadow", color, execFn)
//        .readBuffer(rBuf, VERTEX_SHADER, SHADER_READ)
//        .writeImage(rImg, FRAGMENT_TESTS, DEPTH_STENCIL_WRITE, DEPTH_ATTACHMENT_OPTIMAL);
//   graph.compile();
//
// Usage (per frame):
//   graph.execute(cmd);
//
// Barrier semantics:
//   readBuffer/Image  -- pass reads; resource state not updated after pass
//   writeBuffer/Image -- pass writes; resource state updated after pass
//   readWriteBuffer/Image -- both (e.g. compute atomicAdd)
//
// Pass execute callbacks must NOT emit barriers (the graph emits them before each pass).
// Callbacks MAY begin/end Vulkan render passes internally.
class RenderGraph {
    // Forward-declared here (class-scope, before PassBuilder) so PassBuilder
    // can hold a PassNode* pointer.  Full definition is in the private section.
    struct PassNode;

public:
    using ResourceId = uint32_t;
    static constexpr ResourceId kInvalid   = ~0u;
    // Maximum number of passes the graph supports.
    // GpuTimestamps uses this as its pool size, so keep them in sync.
    static constexpr uint32_t   kMaxPasses = 32;

    // Per-frame getter callbacks: called during execute() to resolve current handles.
    // Use [this]{ return m_buf[m_currentFrame].handle(); } for per-frame resources.
    using GetBuffer = std::function<VkBuffer()>;
    using GetImage  = std::function<VkImage()>;

    // Buffer resource descriptor.
    // Barriers for buffers use VkMemoryBarrier (no per-buffer handle needed).
    // getter is stored for B3 DAG output only.
    struct BufferRes {
        const char*          name;
        GetBuffer            getter;      // optional; used by printDAG for diagnostics
        VkPipelineStageFlags initStage  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags        initAccess = 0;
    };

    // Image resource descriptor.
    // Barriers for images use VkImageMemoryBarrier when a layout transition is needed.
    struct ImageRes {
        const char*          name;
        GetImage             getter;      // called during execute() for VkImageMemoryBarrier
        VkImageAspectFlags   aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
        uint32_t             layerCount = 1;
        uint32_t             mipCount   = 1;
        VkImageLayout        initLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags initStage  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags        initAccess = 0;
    };

    // Fluent pass builder: returned by addPass(), chain calls to declare accesses.
    // All chain calls must complete before calling addPass() again.
    class PassBuilder {
    public:
        PassBuilder& readBuffer(ResourceId res,
                                VkPipelineStageFlags stage, VkAccessFlags access);
        PassBuilder& writeBuffer(ResourceId res,
                                 VkPipelineStageFlags stage, VkAccessFlags access);
        // Read + write (e.g. compute shader atomicAdd).
        PassBuilder& readWriteBuffer(ResourceId res,
                                     VkPipelineStageFlags stage, VkAccessFlags access);

        PassBuilder& readImage(ResourceId res,
                               VkPipelineStageFlags stage, VkAccessFlags access,
                               VkImageLayout layout);
        PassBuilder& writeImage(ResourceId res,
                                VkPipelineStageFlags stage, VkAccessFlags access,
                                VkImageLayout layout);
        PassBuilder& readWriteImage(ResourceId res,
                                    VkPipelineStageFlags stage, VkAccessFlags access,
                                    VkImageLayout layout);
        PassBuilder& writeAttachmentImage(ResourceId res,
                                          VkPipelineStageFlags finalStage,
                                          VkAccessFlags finalAccess,
                                          VkImageLayout finalLayout);

    private:
        friend class RenderGraph;
        explicit PassBuilder(PassNode* node) : m_node(node) {}
        PassNode* m_node;
    };

    RenderGraph();

public:
    // -- Resource registration --------------------------------------------------
    ResourceId addBuffer(BufferRes desc);
    ResourceId addImage(ImageRes desc);

    // -- Pass registration ------------------------------------------------------
    // Returns a PassBuilder for resource access declarations.
    // 'execute' is called after the pre-pass barrier is emitted.
    PassBuilder& addPass(const char* name,
                         glm::vec4   debugColor,
                         std::function<void(VkCommandBuffer)> execute);

    // -- Compile ----------------------------------------------------------------
    // Analyzes declared accesses and builds barrier recipes. Call once after all
    // addBuffer/addImage/addPass declarations. Re-call after reset().
    void compile();

    // -- Execute ----------------------------------------------------------------
    // Per-frame: emits barriers, wraps each pass with debug labels, calls execute.
    //
    // frameIndex  -- current frame-in-flight index (0..MAX_FRAMES_IN_FLIGHT-1).
    // ts          -- optional GpuTimestamps; when non-null writes begin/end timestamps
    //                around each pass for E1 per-pass GPU timing.
    void execute(VkCommandBuffer cmd,
                 uint32_t frameIndex = 0,
                 GpuTimestamps* ts   = nullptr) const;

    // -- Lifecycle --------------------------------------------------------------
    // Clear all passes AND resources. Call before re-declaring (e.g. after resize).
    void reset();

    // -- B3: Debug visibility ---------------------------------------------------
    // Print the pass/resource DAG to stdout: pass order, resource edges, barrier counts.
    void printDAG() const;

    // When set, printDAG() is called at the start of the next execute() (one-shot).
    bool& printOnNextExecute() { return m_printRequested; }

    // -- E2: Barrier trace ------------------------------------------------------
    // When enabled, each vkCmdPipelineBarrier emitted by execute() is logged to
    // stdout with human-readable stage and access mask names.  One-frame toggle
    // is controlled by the UI; persistent toggle via barrierTraceEnabled().
    bool& barrierTraceEnabled() { return m_barrierTraceEnabled; }

    // -- E3: Pass list accessors ------------------------------------------------
    // Iterate passes in compiled execution order for the UI pass list panel.
    // Requires compile() to have been called; returns 0/empty-string otherwise.
    uint32_t    passCount()          const;
    const char* passName(uint32_t i) const;

private:
    struct BufAccess {
        ResourceId           id;
        VkPipelineStageFlags stage;
        VkAccessFlags        access;
        bool                 isWrite;
    };
    struct ImgAccess {
        ResourceId           id;
        VkPipelineStageFlags stage;
        VkAccessFlags        access;
        VkImageLayout        layout;
        bool                 requiresPreBarrier = true;
        bool                 isWrite;
    };
    struct PassNode {
        std::string  name;
        glm::vec4    color{1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<BufAccess> bufAccesses;
        std::vector<ImgAccess> imgAccesses;
        std::function<void(VkCommandBuffer)> execute;
    };

    // Per-resource state tracked during compile().
    struct ResState {
        VkPipelineStageFlags stage  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags        access = 0;
        VkImageLayout        layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    // Compiled barrier group emitted before each pass.
    // Buffers use a global VkMemoryBarrier (no specific handle needed).
    // Images use VkImageMemoryBarrier (handle resolved via getter at execute time).
    struct ImgBarrierDesc {
        uint32_t      resIdx;
        VkAccessFlags src, dst;
        VkImageLayout oldLayout, newLayout;
    };
    struct BarrierGroup {
        // Global memory barrier covering all buffer transitions in this group.
        VkAccessFlags srcBufAccess = 0;
        VkAccessFlags dstBufAccess = 0;
        bool          hasBufBarrier = false;

        // Per-image barriers (layout transitions require specific image handles).
        std::vector<ImgBarrierDesc> imgBarriers;

        // Combined pipeline stages for vkCmdPipelineBarrier.
        VkPipelineStageFlags srcStage = 0;
        VkPipelineStageFlags dstStage = 0;

        bool empty() const { return !hasBufBarrier && imgBarriers.empty(); }
    };

    static constexpr uint32_t kMaxResources = 64;

    std::vector<BufferRes>   m_buffers;
    std::vector<ImageRes>    m_images;
    std::vector<PassNode>    m_passes;
    std::vector<PassBuilder> m_builders;  // parallel to m_passes (stable pointers)

    // Compiled (filled by compile()).
    std::vector<uint32_t>     m_execOrder;    // topologically sorted indices into m_passes
    std::vector<BarrierGroup> m_preBarriers;  // before m_passes[m_execOrder[i]]
    bool         m_compiled           = false;
    mutable bool m_printRequested      = false;
    bool         m_barrierTraceEnabled = false;
};

} // namespace vkt
