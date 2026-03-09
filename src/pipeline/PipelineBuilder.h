#pragma once

#include <volk.h>
#include <vector>

namespace vkt {

// Fluent builder for a graphics VkPipeline.
// Defaults: triangle list, back-face cull, CCW winding, depth test+write,
// no blending, 1x MSAA, dynamic viewport+scissor (set at draw time).
// Shader modules may be destroyed after build() returns.
//
// Usage:
//   VkPipeline pipeline = PipelineBuilder{}
//       .addShaderStage(vert.stageInfo(VK_SHADER_STAGE_VERTEX_BIT))
//       .addShaderStage(frag.stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT))
//       .setVertexInput(Vertex::bindingDesc(), Vertex::attributeDescs())
//       .setDepthTest(true, true)
//       .setCullMode(VK_CULL_MODE_BACK_BIT)
//       .build(device, layout, renderPass);
class PipelineBuilder {
public:
    PipelineBuilder();

    // -- Shader stages ----------------------------------------------------------
    PipelineBuilder& addShaderStage(VkPipelineShaderStageCreateInfo stage);

    // -- Vertex input ----------------------------------------------------------
    PipelineBuilder& setVertexInput(
        const std::vector<VkVertexInputBindingDescription>&   bindings,
        const std::vector<VkVertexInputAttributeDescription>& attributes);

    // -- Input assembly --------------------------------------------------------
    PipelineBuilder& setTopology(VkPrimitiveTopology topology);

    // -- Rasterization ---------------------------------------------------------
    PipelineBuilder& setCullMode(VkCullModeFlags cullMode, VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE);
    PipelineBuilder& setPolygonMode(VkPolygonMode mode);
    PipelineBuilder& setLineWidth(float width);

    // Enable polygon depth bias (depth_bias_constant + slope * depth_bias_slope).
    // Call vkCmdSetDepthBias() at draw time to set the actual values dynamically.
    PipelineBuilder& setDepthBias(bool enable);

    // Set number of color attachments in the render pass (0 for depth-only passes).
    PipelineBuilder& setColorAttachments(uint32_t count);

    // -- Depth / stencil -------------------------------------------------------
    PipelineBuilder& setDepthTest(bool enableTest, bool enableWrite,
                                  VkCompareOp compareOp = VK_COMPARE_OP_LESS);

    // -- Color blending --------------------------------------------------------
    PipelineBuilder& setBlendingEnabled(bool enable);
    // Preset: standard alpha blending (src_alpha / one_minus_src_alpha)
    PipelineBuilder& setAlphaBlending();

    // -- Multisampling ---------------------------------------------------------
    PipelineBuilder& setMultisampling(VkSampleCountFlagBits samples);

    // -- Viewport/scissor (dynamic by default -- set at draw time) ------------
    // Call this only if you need static viewport/scissor.
    PipelineBuilder& setStaticViewport(VkViewport viewport, VkRect2D scissor);

    // -- Build -----------------------------------------------------------------
    // pipelineCache: pass a VkPipelineCache handle to reuse compiled shader binaries
    // across runs (load/save handled externally). VK_NULL_HANDLE disables caching.
    VkPipeline build(VkDevice device, VkPipelineLayout layout, VkRenderPass renderPass,
                     uint32_t subpass = 0,
                     VkPipelineCache pipelineCache = VK_NULL_HANDLE) const;

    // -- Presets ---------------------------------------------------------------
    // Returns a builder pre-configured for a fullscreen triangle pass:
    //   - No vertex input (position generated from gl_VertexIndex in the vertex shader)
    //   - No depth test or depth write
    //   - No back-face culling
    // The caller adds shader stages and calls .setMultisampling() if needed, then build().
    static PipelineBuilder forFullscreenQuad();

private:
    std::vector<VkPipelineShaderStageCreateInfo>      m_stages;
    std::vector<VkVertexInputBindingDescription>       m_bindingDescs;
    std::vector<VkVertexInputAttributeDescription>     m_attributeDescs;

    VkPipelineInputAssemblyStateCreateInfo m_inputAssembly{};
    VkPipelineRasterizationStateCreateInfo m_rasterization{};
    VkPipelineDepthStencilStateCreateInfo  m_depthStencil{};
    VkPipelineColorBlendAttachmentState    m_blendAttachment{};
    VkPipelineMultisampleStateCreateInfo   m_multisampling{};

    bool      m_useStaticViewport    = false;
    VkViewport m_viewport{};
    VkRect2D  m_scissor{};
    uint32_t  m_colorAttachmentCount = 1;
};

} // namespace vkt
