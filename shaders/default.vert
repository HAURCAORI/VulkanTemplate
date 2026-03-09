#version 450

// -- Vertex inputs (must match Vertex::attributeDescs() in Mesh.h) -------------
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inColor;

// -- Interpolated outputs to fragment shader ------------------------------------
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec4 fragColor;
layout(location = 4) out flat uint fragMaterialId;  // no interpolation for integer types
layout(location = 5) out float fragViewDepth;

// -- Uniform block (must match GlobalUBO in Application.h, std140) -------------
struct GPULight {
    vec4 position;   // xyz=pos or dir, w=type (0=dir, 1=point, 2=spot)
    vec4 color;      // rgb=color, a=intensity
    vec4 direction;  // xyz=direction, w=range
    vec4 spotAngles; // x=cos(inner), y=cos(outer)
};

#ifndef MAX_LIGHTS
#define MAX_LIGHTS 8
#endif

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4     view;
    mat4     proj;
    vec4     ambientColor; // rgb=color, a=intensity
    vec4     cameraPos;    // xyz=world position, w=unused
    GPULight lights[MAX_LIGHTS];
    int      lightCount;
    float    _pad0;  // matches C++ float _pad[3] at offsets 676, 680, 684
    float    _pad1;  // (vec3 _pad0 would align to 16 bytes in std140, shifting
    float    _pad2;  //  lightSpaceMatrices to offset 704 instead of C++'s 688)
    mat4     lightSpaceMatrices[4];
    vec4     cascadeSplits;
    ivec4    shadowMeta;
    vec4     shadowParams;
} ubo;

// -- Per-instance SSBO (must match InstanceData in InstanceData.h, std430) -----
// Indexed by gl_InstanceIndex, written each frame by RenderWorld::draw().
// model:        camera-relative world transform (floating-origin offset applied by CPU)
// normalMatrix: mat4(mat3(inverseTranspose(model))), corrects normals under non-uniform scale
// materialId:   index into the GPUMaterial SSBO at set=0, binding=3
struct InstanceData {
    mat4 model;
    mat4 normalMatrix;
    uint materialId;
    uint _pad0, _pad1, _pad2;  // keeps struct 16-byte aligned (std430)
};

layout(set = 0, binding = 1, std430) readonly buffer InstanceSSBO {
    InstanceData instances[];
} ssbo;

void main() {
    InstanceData inst = ssbo.instances[gl_InstanceIndex];
    vec4 worldPos     = inst.model * vec4(inPosition, 1.0);
    fragWorldPos      = worldPos.xyz;
    // Transform normal to world space using the normal matrix; renormalize
    // because interpolation across the triangle can shrink the vector.
    fragNormal        = normalize(mat3(inst.normalMatrix) * inNormal);
    fragTexCoord      = inTexCoord;
    fragColor         = inColor;
    fragMaterialId    = inst.materialId;
    // Signed camera-forward depth in view space (RH view: in-front => -z).
    // Keep sign so behind-camera fragments are not treated as valid cascade depth.
    fragViewDepth     = -(ubo.view * worldPos).z;
    gl_Position       = ubo.proj * ubo.view * worldPos;
}
