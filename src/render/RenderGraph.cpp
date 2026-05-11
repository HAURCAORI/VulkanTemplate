#include "render/RenderGraph.h"
#include "render/GpuTimestamps.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <string>

namespace vkt {

namespace {

const char* layoutName(VkImageLayout layout) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
    case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return "COLOR_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "SHADER_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "TRANSFER_SRC_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "TRANSFER_DST_OPTIMAL";
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: return "PRESENT_SRC_KHR";
    default: return "UNKNOWN_LAYOUT";
    }
}

// -- E2: Human-readable stage/access mask strings (for barrier trace) ----------

static std::string stageFlags(VkPipelineStageFlags s) {
    if (s == 0) return "NONE";
    std::string r;
    auto add = [&](VkPipelineStageFlags b, const char* n) {
        if (s & b) { if (!r.empty()) r += '|'; r += n; }
    };
    add(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,             "TOP");
    add(VK_PIPELINE_STAGE_HOST_BIT,                    "HOST");
    add(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,            "VERT_IN");
    add(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,           "VERT");
    add(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,         "FRAG");
    add(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,    "EARLY_FRAG");
    add(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,     "LATE_FRAG");
    add(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, "COLOR_OUT");
    add(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,          "COMPUTE");
    add(VK_PIPELINE_STAGE_TRANSFER_BIT,                "XFER");
    add(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,          "BOTTOM");
    if (r.empty()) r = "OTHER";
    return r;
}

static std::string accessFlags(VkAccessFlags a) {
    if (a == 0) return "NONE";
    std::string r;
    auto add = [&](VkAccessFlags b, const char* n) {
        if (a & b) { if (!r.empty()) r += '|'; r += n; }
    };
    add(VK_ACCESS_HOST_WRITE_BIT,                     "HOST_W");
    add(VK_ACCESS_SHADER_READ_BIT,                    "SHADER_R");
    add(VK_ACCESS_SHADER_WRITE_BIT,                   "SHADER_W");
    add(VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,          "COLOR_R");
    add(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,         "COLOR_W");
    add(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,  "DEPTH_R");
    add(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, "DEPTH_W");
    add(VK_ACCESS_INDIRECT_COMMAND_READ_BIT,          "INDIRECT_R");
    add(VK_ACCESS_INDEX_READ_BIT,                     "INDEX_R");
    add(VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,          "VERTEX_R");
    add(VK_ACCESS_TRANSFER_READ_BIT,                  "XFER_R");
    add(VK_ACCESS_TRANSFER_WRITE_BIT,                 "XFER_W");
    if (r.empty()) r = "OTHER";
    return r;
}

} // namespace

// -- PassBuilder --------------------------------------------------------------

RenderGraph::PassBuilder& RenderGraph::PassBuilder::readBuffer(
    ResourceId res, VkPipelineStageFlags stage, VkAccessFlags access) {
    m_node->bufAccesses.push_back({res, stage, access, false});
    return *this;
}

RenderGraph::PassBuilder& RenderGraph::PassBuilder::writeBuffer(
    ResourceId res, VkPipelineStageFlags stage, VkAccessFlags access) {
    m_node->bufAccesses.push_back({res, stage, access, true});
    return *this;
}

RenderGraph::PassBuilder& RenderGraph::PassBuilder::readWriteBuffer(
    ResourceId res, VkPipelineStageFlags stage, VkAccessFlags access) {
    // Record as both a read and a write so compile() sees the full hazard.
    m_node->bufAccesses.push_back({res, stage, access, false});
    m_node->bufAccesses.push_back({res, stage, access, true});
    return *this;
}

RenderGraph::PassBuilder& RenderGraph::PassBuilder::readImage(
    ResourceId res, VkPipelineStageFlags stage, VkAccessFlags access,
    VkImageLayout layout) {
    m_node->imgAccesses.push_back({res, stage, access, layout, true, false});
    return *this;
}

RenderGraph::PassBuilder& RenderGraph::PassBuilder::writeImage(
    ResourceId res, VkPipelineStageFlags stage, VkAccessFlags access,
    VkImageLayout layout) {
    m_node->imgAccesses.push_back({res, stage, access, layout, true, true});
    return *this;
}

RenderGraph::PassBuilder& RenderGraph::PassBuilder::readWriteImage(
    ResourceId res, VkPipelineStageFlags stage, VkAccessFlags access,
    VkImageLayout layout) {
    m_node->imgAccesses.push_back({res, stage, access, layout, true, false});
    m_node->imgAccesses.push_back({res, stage, access, layout, true, true});
    return *this;
}

RenderGraph::PassBuilder& RenderGraph::PassBuilder::writeAttachmentImage(
    ResourceId res, VkPipelineStageFlags finalStage, VkAccessFlags finalAccess,
    VkImageLayout finalLayout) {
    m_node->imgAccesses.push_back({res, finalStage, finalAccess, finalLayout, false, true});
    return *this;
}

// -- RenderGraph --------------------------------------------------------------

RenderGraph::RenderGraph() {
    m_passes.reserve(kMaxPasses);
    m_builders.reserve(kMaxPasses);
    m_buffers.reserve(kMaxResources);
    m_images.reserve(kMaxResources);
}

RenderGraph::ResourceId RenderGraph::addBuffer(BufferRes desc) {
    assert(m_buffers.size() < kMaxResources && "RenderGraph: buffer resource limit exceeded");
    m_buffers.push_back(std::move(desc));
    return static_cast<ResourceId>(m_buffers.size() - 1u);
}

RenderGraph::ResourceId RenderGraph::addImage(ImageRes desc) {
    assert(m_images.size() < kMaxResources && "RenderGraph: image resource limit exceeded");
    m_images.push_back(std::move(desc));
    return static_cast<ResourceId>(m_images.size() - 1u);
}

RenderGraph::PassBuilder& RenderGraph::addPass(
    const char* name, glm::vec4 debugColor,
    std::function<void(VkCommandBuffer)> execute) {
    assert(m_passes.size() < kMaxPasses && "RenderGraph: pass limit exceeded");
    PassNode node;
    node.name    = name;
    node.color   = debugColor;
    node.execute = std::move(execute);
    m_passes.push_back(std::move(node));
    // Construct PassBuilder in this scope (where the private ctor is accessible),
    // then push the already-constructed object into the vector.
    m_builders.push_back(PassBuilder(&m_passes.back()));
    return m_builders.back();
}

void RenderGraph::reset() {
    m_passes.clear();
    m_builders.clear();
    m_buffers.clear();
    m_images.clear();
    m_execOrder.clear();
    m_preBarriers.clear();
    m_compiled = false;
}

void RenderGraph::compile() {
    const uint32_t nBuf  = static_cast<uint32_t>(m_buffers.size());
    const uint32_t nImg  = static_cast<uint32_t>(m_images.size());
    const uint32_t nPass = static_cast<uint32_t>(m_passes.size());

    struct BufUse {
        bool                 read  = false;
        bool                 write = false;
        VkPipelineStageFlags readStage = 0;
        VkPipelineStageFlags writeStage = 0;
        VkAccessFlags        readAccess = 0;
        VkAccessFlags        writeAccess = 0;
    };
    struct ImgUse {
        bool                 read  = false;
        bool                 write = false;
        bool                 requiresPreBarrier = true;
        VkPipelineStageFlags readStage = 0;
        VkPipelineStageFlags writeStage = 0;
        VkAccessFlags        readAccess = 0;
        VkAccessFlags        writeAccess = 0;
        VkImageLayout        layout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool                 layoutSet = false;
    };

    std::vector<std::vector<BufUse>> passBufUses(
        nPass, std::vector<BufUse>(nBuf));
    std::vector<std::vector<ImgUse>> passImgUses(
        nPass, std::vector<ImgUse>(nImg));

    for (uint32_t p = 0; p < nPass; ++p) {
        const PassNode& pass = m_passes[p];

        for (const auto& acc : pass.bufAccesses) {
            assert(acc.id < nBuf && "RenderGraph: buffer ResourceId out of range");
            BufUse& use = passBufUses[p][acc.id];
            if (acc.isWrite) {
                use.write = true;
                use.writeStage  |= acc.stage;
                use.writeAccess |= acc.access;
            } else {
                use.read = true;
                use.readStage  |= acc.stage;
                use.readAccess |= acc.access;
            }
        }

        for (const auto& acc : pass.imgAccesses) {
            assert(acc.id < nImg && "RenderGraph: image ResourceId out of range");
            ImgUse& use = passImgUses[p][acc.id];
            if (use.layoutSet) {
                assert(use.layout == acc.layout &&
                       "RenderGraph: conflicting image layouts for one resource in a single pass");
                assert(use.requiresPreBarrier == acc.requiresPreBarrier &&
                       "RenderGraph: mixed attachment-owned and graph-owned image transitions in one pass");
            } else {
                use.layout = acc.layout;
                use.layoutSet = true;
                use.requiresPreBarrier = acc.requiresPreBarrier;
            }

            if (acc.isWrite) {
                use.write = true;
                use.writeStage  |= acc.stage;
                use.writeAccess |= acc.access;
            } else {
                use.read = true;
                use.readStage  |= acc.stage;
                use.readAccess |= acc.access;
            }
        }
    }

    std::vector<std::vector<uint32_t>> edges(nPass);
    std::vector<uint32_t> indegree(nPass, 0);
    auto addEdge = [&](uint32_t src, uint32_t dst) {
        if (src == dst) return;
        auto& out = edges[src];
        if (std::find(out.begin(), out.end(), dst) == out.end()) {
            out.push_back(dst);
            ++indegree[dst];
        }
    };

    std::vector<int32_t> lastBufWriter(nBuf, -1);
    std::vector<std::vector<uint32_t>> lastBufReaders(nBuf);
    for (uint32_t p = 0; p < nPass; ++p) {
        for (uint32_t r = 0; r < nBuf; ++r) {
            const BufUse& use = passBufUses[p][r];
            if (!use.read && !use.write) continue;

            if (use.read && lastBufWriter[r] >= 0)
                addEdge(static_cast<uint32_t>(lastBufWriter[r]), p);
            if (use.write) {
                if (lastBufWriter[r] >= 0)
                    addEdge(static_cast<uint32_t>(lastBufWriter[r]), p);
                for (uint32_t reader : lastBufReaders[r])
                    addEdge(reader, p);
                lastBufReaders[r].clear();
                lastBufWriter[r] = static_cast<int32_t>(p);
            } else {
                lastBufReaders[r].push_back(p);
            }
        }
    }

    std::vector<int32_t> lastImgWriter(nImg, -1);
    std::vector<std::vector<uint32_t>> lastImgReaders(nImg);
    for (uint32_t p = 0; p < nPass; ++p) {
        for (uint32_t r = 0; r < nImg; ++r) {
            const ImgUse& use = passImgUses[p][r];
            if (!use.read && !use.write) continue;

            if (use.read && lastImgWriter[r] >= 0)
                addEdge(static_cast<uint32_t>(lastImgWriter[r]), p);
            if (use.write) {
                if (lastImgWriter[r] >= 0)
                    addEdge(static_cast<uint32_t>(lastImgWriter[r]), p);
                for (uint32_t reader : lastImgReaders[r])
                    addEdge(reader, p);
                lastImgReaders[r].clear();
                lastImgWriter[r] = static_cast<int32_t>(p);
            } else {
                lastImgReaders[r].push_back(p);
            }
        }
    }

    m_execOrder.clear();
    m_execOrder.reserve(nPass);
    std::vector<uint32_t> ready;
    ready.reserve(nPass);
    for (uint32_t p = 0; p < nPass; ++p)
        if (indegree[p] == 0) ready.push_back(p);

    while (!ready.empty()) {
        const auto it = std::min_element(ready.begin(), ready.end());
        const uint32_t p = *it;
        ready.erase(it);
        m_execOrder.push_back(p);
        for (uint32_t dst : edges[p]) {
            assert(indegree[dst] > 0 && "RenderGraph: invalid indegree during topological sort");
            --indegree[dst];
            if (indegree[dst] == 0)
                ready.push_back(dst);
        }
    }
    assert(m_execOrder.size() == nPass &&
           "RenderGraph: compile detected a pass dependency cycle or ambiguous hazard");

    // Initialize resource states from their declared initial conditions.
    std::vector<ResState> bufSt(nBuf), imgSt(nImg);
    for (uint32_t i = 0; i < nBuf; ++i)
        bufSt[i] = {m_buffers[i].initStage, m_buffers[i].initAccess, VK_IMAGE_LAYOUT_UNDEFINED};
    for (uint32_t i = 0; i < nImg; ++i)
        imgSt[i] = {m_images[i].initStage,  m_images[i].initAccess,  m_images[i].initLayout};

    m_preBarriers.assign(nPass, BarrierGroup{});

    for (uint32_t execIdx = 0; execIdx < nPass; ++execIdx) {
        const uint32_t passIdx = m_execOrder[execIdx];
        BarrierGroup&   bg     = m_preBarriers[execIdx];

        // -- Buffer accesses ---------------------------------------------------
        // Use a single global VkMemoryBarrier covering all buffer transitions.
        for (uint32_t r = 0; r < nBuf; ++r) {
            const BufUse& use = passBufUses[passIdx][r];
            if (!use.read && !use.write) continue;
            ResState& st = bufSt[r];

            // A barrier is needed if:
            //   - Previous stage/access could conflict (write-after-write or
            //     read-after-write or write-after-read).
            const bool prevWrote  = (st.access & (VK_ACCESS_SHADER_WRITE_BIT |
                                                   VK_ACCESS_HOST_WRITE_BIT   |
                                                   VK_ACCESS_TRANSFER_WRITE_BIT)) != 0;
            const VkPipelineStageFlags useStages = use.readStage | use.writeStage;
            const VkAccessFlags useAccess = use.readAccess | use.writeAccess;
            const bool stagesDiff = st.stage != useStages;
            const bool needsBarrier = prevWrote || (use.write && st.access != 0) || stagesDiff;

            if (needsBarrier) {
                bg.srcBufAccess |= st.access;
                bg.dstBufAccess |= useAccess;
                bg.srcStage     |= st.stage;
                bg.dstStage     |= useStages;
                bg.hasBufBarrier = true;
            }

            // Update state only on writes.
            if (use.write) {
                st.stage  = use.writeStage;
                st.access = use.writeAccess;
            }
        }

        // -- Image accesses ----------------------------------------------------
        // Use VkImageMemoryBarrier when a layout transition is needed.
        for (uint32_t r = 0; r < nImg; ++r) {
            const ImgUse& use = passImgUses[passIdx][r];
            if (!use.read && !use.write) continue;
            ResState& st = imgSt[r];

            const bool layoutChanges = st.layout != use.layout;
            const bool prevWrote     = (st.access & (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT      |
                                                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                                      VK_ACCESS_SHADER_WRITE_BIT                   |
                                                      VK_ACCESS_TRANSFER_WRITE_BIT)) != 0;

            if (use.requiresPreBarrier && (layoutChanges || prevWrote)) {
                bg.imgBarriers.push_back({r, st.access,
                                          use.readAccess | use.writeAccess,
                                          st.layout, use.layout});
                bg.srcStage |= st.stage;
                bg.dstStage |= use.readStage | use.writeStage;
            }

            // Update state on writes.
            if (use.write) {
                st.stage  = use.writeStage;
                st.access = use.writeAccess;
                st.layout = use.layout;
            }
        }
    }

    m_compiled = true;
}

void RenderGraph::execute(VkCommandBuffer cmd, uint32_t frameIndex, GpuTimestamps* ts) const {
    assert(m_compiled && "RenderGraph::execute() called before compile()");

    if (m_printRequested) {
        printDAG();
        m_printRequested = false;
    }

    for (uint32_t execIdx = 0; execIdx < static_cast<uint32_t>(m_execOrder.size()); ++execIdx) {
        const BarrierGroup& bg = m_preBarriers[execIdx];

        if (!bg.empty()) {
            // Global memory barrier for all buffer transitions in this group.
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = bg.srcBufAccess;
            mb.dstAccessMask = bg.dstBufAccess;

            // Per-image barriers for layout transitions.
            std::vector<VkImageMemoryBarrier> imgBarriers;
            imgBarriers.reserve(bg.imgBarriers.size());
            for (const auto& ib : bg.imgBarriers) {
                const ImageRes& res = m_images[ib.resIdx];
                VkImageMemoryBarrier imb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                imb.srcAccessMask               = ib.src;
                imb.dstAccessMask               = ib.dst;
                imb.oldLayout                   = ib.oldLayout;
                imb.newLayout                   = ib.newLayout;
                imb.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
                imb.image                       = res.getter();
                imb.subresourceRange.aspectMask     = res.aspect;
                imb.subresourceRange.baseMipLevel   = 0;
                imb.subresourceRange.levelCount     = res.mipCount;
                imb.subresourceRange.baseArrayLayer = 0;
                imb.subresourceRange.layerCount     = res.layerCount;
                imgBarriers.push_back(imb);
            }

            const VkPipelineStageFlags src =
                bg.srcStage ? bg.srcStage : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            // E2: Barrier trace -- log transitions before emitting (when enabled).
            if (m_barrierTraceEnabled) {
                const PassNode& tracePass = m_passes[m_execOrder[execIdx]];
                std::printf("[BarrierTrace] Before \"%s\":\n", tracePass.name.c_str());
                if (bg.hasBufBarrier) {
                    std::printf("  buf  src=%s  dst=%s  stage: %s -> %s\n",
                                accessFlags(bg.srcBufAccess).c_str(),
                                accessFlags(bg.dstBufAccess).c_str(),
                                stageFlags(src).c_str(),
                                stageFlags(bg.dstStage).c_str());
                }
                for (const auto& ib : bg.imgBarriers) {
                    const ImageRes& res = m_images[ib.resIdx];
                    std::printf("  img  \"%s\"  layout: %s -> %s  src=%s  dst=%s\n",
                                res.name,
                                layoutName(ib.oldLayout), layoutName(ib.newLayout),
                                accessFlags(ib.src).c_str(),
                                accessFlags(ib.dst).c_str());
                }
            }

            vkCmdPipelineBarrier(
                cmd, src, bg.dstStage, 0,
                bg.hasBufBarrier ? 1u : 0u, bg.hasBufBarrier ? &mb : nullptr,
                0, nullptr,
                static_cast<uint32_t>(imgBarriers.size()),
                imgBarriers.empty() ? nullptr : imgBarriers.data());
        }

        const PassNode& pass = m_passes[m_execOrder[execIdx]];

        // E1: Begin timestamp AFTER the barrier but BEFORE pass work.
        // This measures pure pass GPU cost, excluding pipeline stall time.
        if (ts) ts->beginPass(cmd, frameIndex, execIdx, pass.name.c_str());

        {
            debug::LabelScope label(cmd, pass.name.c_str(), pass.color);
            pass.execute(cmd);
        }

        // E1: End timestamp after pass work.
        if (ts) ts->endPass(cmd, frameIndex, execIdx);
    }
}

// -- B3: DAG printing ---------------------------------------------------------

void RenderGraph::printDAG() const {
    std::printf("\n=== RenderGraph DAG ===\n");
    std::printf("  Buffers : %zu\n", m_buffers.size());
    std::printf("  Images  : %zu\n", m_images.size());
    std::printf("  Passes  : %zu\n\n", m_passes.size());

    std::vector<VkImageLayout> trackedLayouts(m_images.size(), VK_IMAGE_LAYOUT_UNDEFINED);
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_images.size()); ++i)
        trackedLayouts[i] = m_images[i].initLayout;

    for (uint32_t execIdx = 0; execIdx < static_cast<uint32_t>(m_execOrder.size()); ++execIdx) {
        const uint32_t passIdx = m_execOrder[execIdx];
        const PassNode&   pass = m_passes[passIdx];
        const BarrierGroup& bg = m_preBarriers[execIdx];

        std::printf("  [%u] \"%s\"", execIdx, pass.name.c_str());
        if (!bg.empty()) {
            std::printf("  -- barriers: %s%zu img",
                        bg.hasBufBarrier ? "1 buf  " : "",
                        bg.imgBarriers.size());
        }
        std::printf("\n");

        for (const auto& acc : pass.bufAccesses) {
            std::printf("        %s  buf[%u] \"%s\"\n",
                        acc.isWrite ? "WRITE" : "READ ",
                        acc.id, m_buffers[acc.id].name);
        }
        for (const auto& acc : pass.imgAccesses) {
            const VkImageLayout oldLayout = trackedLayouts[acc.id];
            if (acc.requiresPreBarrier) {
                if (oldLayout == acc.layout) {
                    std::printf("        %s  img[%u] \"%s\"  layout=%s  (no transition)\n",
                                acc.isWrite ? "WRITE" : "READ ",
                                acc.id, m_images[acc.id].name,
                                layoutName(acc.layout));
                } else {
                    std::printf("        %s  img[%u] \"%s\"  layout=%s -> %s\n",
                                acc.isWrite ? "WRITE" : "READ ",
                                acc.id, m_images[acc.id].name,
                                layoutName(oldLayout), layoutName(acc.layout));
                }
            } else {
                std::printf("        %s  img[%u] \"%s\"  final=%s  (pass-owned transition from %s)\n",
                            acc.isWrite ? "WRITE" : "READ ",
                            acc.id, m_images[acc.id].name,
                            layoutName(acc.layout), layoutName(oldLayout));
            }
            if (acc.isWrite)
                trackedLayouts[acc.id] = acc.layout;
        }
    }
    std::printf("=======================\n\n");
}

// -- E3: Pass list accessors ---------------------------------------------------

uint32_t RenderGraph::passCount() const {
    if (!m_compiled) return 0u;
    return static_cast<uint32_t>(m_execOrder.size());
}

const char* RenderGraph::passName(uint32_t execIdx) const {
    if (!m_compiled || execIdx >= static_cast<uint32_t>(m_execOrder.size())) return "";
    return m_passes[m_execOrder[execIdx]].name.c_str();
}

} // namespace vkt
