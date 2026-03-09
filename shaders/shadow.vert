#version 450

// Depth-only shadow pass vertex shader.
// Reads the model matrix from the instance SSBO (same data as the main pass).
// Only inPosition is used; other attributes are bound but not read.

layout(location = 0) in vec3 inPosition;
// Attributes 1-3 (normal, texCoord, color) are bound by the vertex buffer
// but not declared here -- unused input locations are silently ignored.

// Instance SSBO (set=0, binding=1) -- must match InstanceData in InstanceData.h
struct InstanceData {
    mat4 model;
    mat4 normalMatrix;  // unused in depth pass
    uint materialId;    // unused in depth pass
    uint _pad0, _pad1, _pad2;
};

layout(set = 0, binding = 1, std430) readonly buffer InstanceSSBO {
    InstanceData instances[];
} ssbo;

// Light space VP matrix (push constant)
layout(push_constant) uniform ShadowPush {
    mat4 lightSpaceMatrix;
} push;

void main() {
    mat4 model  = ssbo.instances[gl_InstanceIndex].model;
    gl_Position = push.lightSpaceMatrix * model * vec4(inPosition, 1.0);
}
