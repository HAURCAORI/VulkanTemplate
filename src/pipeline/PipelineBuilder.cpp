#include "PipelineBuilder.h"

#include <stdexcept>
#include <array>

namespace vkt {

PipelineBuilder::PipelineBuilder() {
    // -- Input assembly --------------------------------------------------------
    m_inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    m_inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    m_inputAssembly.primitiveRestartEnable = VK_FALSE;

    // -- Rasterization ---------------------------------------------------------
    m_rasterization.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    m_rasterization.depthClampEnable        = VK_FALSE;
    m_rasterization.rasterizerDiscardEnable = VK_FALSE;
    m_rasterization.polygonMode             = VK_POLYGON_MODE_FILL;
    m_rasterization.cullMode                = VK_CULL_MODE_BACK_BIT;
    m_rasterization.frontFace              = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    m_rasterization.depthBiasEnable         = VK_FALSE;
    m_rasterization.lineWidth               = 1.0f;

    // -- Depth / stencil -------------------------------------------------------
    m_depthStencil.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    m_depthStencil.depthTestEnable       = VK_TRUE;
    m_depthStencil.depthWriteEnable      = VK_TRUE;
    m_depthStencil.depthCompareOp        = VK_COMPARE_OP_LESS;
    m_depthStencil.depthBoundsTestEnable = VK_FALSE;
    m_depthStencil.stencilTestEnable     = VK_FALSE;

    // -- Color blend attachment ------------------------------------------------
    m_blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    m_blendAttachment.blendEnable = VK_FALSE;

    // -- Multisampling ---------------------------------------------------------
    m_multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    m_multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    m_multisampling.sampleShadingEnable  = VK_FALSE;
}

PipelineBuilder& PipelineBuilder::addShaderStage(VkPipelineShaderStageCreateInfo stage) {
    m_stages.push_back(stage);
    return *this;
}

PipelineBuilder& PipelineBuilder::setVertexInput(
    const std::vector<VkVertexInputBindingDescription>&   bindings,
    const std::vector<VkVertexInputAttributeDescription>& attributes)
{
    m_bindingDescs   = bindings;
    m_attributeDescs = attributes;
    return *this;
}

PipelineBuilder& PipelineBuilder::setTopology(VkPrimitiveTopology topology) {
    m_inputAssembly.topology = topology;
    return *this;
}

PipelineBuilder& PipelineBuilder::setCullMode(VkCullModeFlags cullMode, VkFrontFace frontFace) {
    m_rasterization.cullMode  = cullMode;
    m_rasterization.frontFace = frontFace;
    return *this;
}

PipelineBuilder& PipelineBuilder::setPolygonMode(VkPolygonMode mode) {
    m_rasterization.polygonMode = mode;
    return *this;
}

PipelineBuilder& PipelineBuilder::setLineWidth(float width) {
    m_rasterization.lineWidth = width;
    return *this;
}

PipelineBuilder& PipelineBuilder::setDepthTest(bool enableTest, bool enableWrite,
                                                VkCompareOp compareOp) {
    m_depthStencil.depthTestEnable  = enableTest  ? VK_TRUE : VK_FALSE;
    m_depthStencil.depthWriteEnable = enableWrite ? VK_TRUE : VK_FALSE;
    m_depthStencil.depthCompareOp   = compareOp;
    return *this;
}

PipelineBuilder& PipelineBuilder::setBlendingEnabled(bool enable) {
    m_blendAttachment.blendEnable = enable ? VK_TRUE : VK_FALSE;
    return *this;
}

PipelineBuilder& PipelineBuilder::setAlphaBlending() {
    m_blendAttachment.blendEnable         = VK_TRUE;
    m_blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    m_blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    m_blendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    m_blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    m_blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    m_blendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
    return *this;
}

PipelineBuilder& PipelineBuilder::setMultisampling(VkSampleCountFlagBits samples) {
    m_multisampling.rasterizationSamples = samples;
    return *this;
}

PipelineBuilder& PipelineBuilder::setDepthBias(bool enable) {
    m_rasterization.depthBiasEnable = enable ? VK_TRUE : VK_FALSE;
    return *this;
}

PipelineBuilder& PipelineBuilder::setColorAttachments(uint32_t count) {
    m_colorAttachmentCount = count;
    return *this;
}

PipelineBuilder& PipelineBuilder::setStaticViewport(VkViewport viewport, VkRect2D scissor) {
    m_useStaticViewport = true;
    m_viewport          = viewport;
    m_scissor           = scissor;
    return *this;
}

VkPipeline PipelineBuilder::build(VkDevice device, VkPipelineLayout layout,
                                   VkRenderPass renderPass, uint32_t subpass,
                                   VkPipelineCache pipelineCache) const {
    if (m_stages.empty())
        throw std::runtime_error("[PipelineBuilder] No shader stages added before build()");

    // Vertex input
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = static_cast<uint32_t>(m_bindingDescs.size());
    vertexInput.pVertexBindingDescriptions      = m_bindingDescs.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(m_attributeDescs.size());
    vertexInput.pVertexAttributeDescriptions    = m_attributeDescs.data();

    // Viewport state
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;
    if (m_useStaticViewport) {
        viewportState.pViewports = &m_viewport;
        viewportState.pScissors  = &m_scissor;
    }

    // Dynamic state (viewport + scissor + depth bias set at draw time unless overridden)
    std::array<VkDynamicState, 3> dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    // Color blend state
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.logicOpEnable   = VK_FALSE;
    colorBlend.attachmentCount = m_colorAttachmentCount;
    colorBlend.pAttachments    = m_colorAttachmentCount > 0 ? &m_blendAttachment : nullptr;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = static_cast<uint32_t>(m_stages.size());
    pipelineInfo.pStages             = m_stages.data();
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &m_inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &m_rasterization;
    pipelineInfo.pMultisampleState   = &m_multisampling;
    pipelineInfo.pDepthStencilState  = &m_depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    // pDynamicState must be nullptr (not pointing to an empty struct) when
    // using static viewport/scissor; the driver ignores it otherwise.
    pipelineInfo.pDynamicState       = m_useStaticViewport ? nullptr : &dynamicState;
    pipelineInfo.layout              = layout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = subpass;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("[PipelineBuilder] Failed to create graphics pipeline");
    return pipeline;
}

PipelineBuilder PipelineBuilder::forFullscreenQuad() {
    // No vertex buffer (gl_VertexIndex generates positions), no depth, no cull.
    return PipelineBuilder{}
        .setCullMode(VK_CULL_MODE_NONE)
        .setDepthTest(false, false);
}

} // namespace vkt
