#pragma once

#include <volk.h>
#include <glm/glm.hpp>

// GPU debug utilities via VK_EXT_debug_utils.
// Active only when VKT_VALIDATION is defined (Debug); all functions compile to
// zero-cost no-ops in Release  --  no #ifdef guards needed at call sites.
// vk-bootstrap requests the extension automatically via use_default_debug_messenger().
//
// Usage:
//   vkt::debug::beginLabel(cmd, "Shadow Pass", {0.8f, 0.4f, 0.1f, 1.0f});
//   vkt::debug::endLabel(cmd);
//   vkt::debug::setObjectName(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "Main");

namespace vkt::debug {

// -- Object naming -------------------------------------------------------------
// Attach a human-readable label to any Vulkan handle.
// Visible in RenderDoc's Resource Inspector, NSight, and Aftermath.

inline void setObjectName(VkDevice device, VkObjectType type,
                           uint64_t handle, const char* name) {
#if defined(VKT_VALIDATION)
    if (!vkSetDebugUtilsObjectNameEXT) return;
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType   = type;
    info.objectHandle = handle;
    info.pObjectName  = name;
    vkSetDebugUtilsObjectNameEXT(device, &info);
#else
    (void)device; (void)type; (void)handle; (void)name;
#endif
}

// Typed overload -- casts any dispatchable/non-dispatchable handle to uint64_t.
template<typename T>
inline void setObjectName(VkDevice device, VkObjectType type, T handle, const char* name) {
    setObjectName(device, type, reinterpret_cast<uint64_t>(handle), name);
}

// -- Command-buffer labels -----------------------------------------------------
// Labels appear as collapsible groups in RenderDoc's Event Browser.

// Begin a named, colored region in the GPU command stream.
inline void beginLabel(VkCommandBuffer cmd, const char* name,
                       glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}) {
#if defined(VKT_VALIDATION)
    if (!vkCmdBeginDebugUtilsLabelEXT) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0]   = color.r;
    label.color[1]   = color.g;
    label.color[2]   = color.b;
    label.color[3]   = color.a;
    vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
#else
    (void)cmd; (void)name; (void)color;
#endif
}

// End the innermost open region.
inline void endLabel(VkCommandBuffer cmd) {
#if defined(VKT_VALIDATION)
    if (!vkCmdEndDebugUtilsLabelEXT) return;
    vkCmdEndDebugUtilsLabelEXT(cmd);
#else
    (void)cmd;
#endif
}

// Insert a single-point annotation (not a region).
inline void insertLabel(VkCommandBuffer cmd, const char* name,
                        glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}) {
#if defined(VKT_VALIDATION)
    if (!vkCmdInsertDebugUtilsLabelEXT) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0]   = color.r;
    label.color[1]   = color.g;
    label.color[2]   = color.b;
    label.color[3]   = color.a;
    vkCmdInsertDebugUtilsLabelEXT(cmd, &label);
#else
    (void)cmd; (void)name; (void)color;
#endif
}

// -- Queue labels --------------------------------------------------------------
// Queue labels annotate the GPU queue timeline in RenderDoc's Queue Events panel.
// Useful for marking frame or workload boundaries at the queue submission level.
// Active only when VKT_VALIDATION is defined (same guard as command-buffer labels).

inline void beginQueueLabel(VkQueue queue, const char* name,
                             glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}) {
#if defined(VKT_VALIDATION)
    if (!vkQueueBeginDebugUtilsLabelEXT) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0]   = color.r;
    label.color[1]   = color.g;
    label.color[2]   = color.b;
    label.color[3]   = color.a;
    vkQueueBeginDebugUtilsLabelEXT(queue, &label);
#else
    (void)queue; (void)name; (void)color;
#endif
}

inline void endQueueLabel(VkQueue queue) {
#if defined(VKT_VALIDATION)
    if (!vkQueueEndDebugUtilsLabelEXT) return;
    vkQueueEndDebugUtilsLabelEXT(queue);
#else
    (void)queue;
#endif
}

// Insert a single-point annotation on the queue timeline.
inline void insertQueueLabel(VkQueue queue, const char* name,
                              glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f}) {
#if defined(VKT_VALIDATION)
    if (!vkQueueInsertDebugUtilsLabelEXT) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0]   = color.r;
    label.color[1]   = color.g;
    label.color[2]   = color.b;
    label.color[3]   = color.a;
    vkQueueInsertDebugUtilsLabelEXT(queue, &label);
#else
    (void)queue; (void)name; (void)color;
#endif
}

// -- RAII scope guard ----------------------------------------------------------
// Calls beginLabel on construction and endLabel on destruction.
// Eliminates the need to remember paired endLabel calls.
//
// Usage:
//   {
//       vkt::debug::LabelScope scope(cmd, "Geometry Pass", {0.2f, 0.6f, 0.9f, 1.0f});
//       // ... draw calls ...
//   } // endLabel called automatically here
struct LabelScope {
    explicit LabelScope(VkCommandBuffer cmd, const char* name,
                        glm::vec4 color = {1.0f, 1.0f, 1.0f, 1.0f})
        : m_cmd(cmd) { beginLabel(cmd, name, color); }
    ~LabelScope() { endLabel(m_cmd); }

    LabelScope(const LabelScope&)            = delete;
    LabelScope& operator=(const LabelScope&) = delete;

private:
    VkCommandBuffer m_cmd;
};

} // namespace vkt::debug
