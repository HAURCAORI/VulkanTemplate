#version 450

layout(location = 0) out vec3 outDir;

// Minimal GlobalUBO view: only view and proj are used.
// The struct matches the first 128 bytes of the full GlobalUBO (std140).
layout(set = 0, binding = 0) uniform SkyUBO {
    mat4 view;
    mat4 proj;
} ubo;

// Unit cube vertices: 36 positions (6 faces x 2 triangles x 3 verts).
// Positions are in [-1,1] cube space; used directly as cubemap direction vectors.
const vec3 kCubeVerts[36] = vec3[](
    // +X face (right)
    vec3( 1,-1, 1), vec3( 1,-1,-1), vec3( 1, 1,-1),
    vec3( 1, 1,-1), vec3( 1, 1, 1), vec3( 1,-1, 1),
    // -X face (left)
    vec3(-1,-1,-1), vec3(-1,-1, 1), vec3(-1, 1, 1),
    vec3(-1, 1, 1), vec3(-1, 1,-1), vec3(-1,-1,-1),
    // +Y face (top)
    vec3(-1, 1, 1), vec3( 1, 1, 1), vec3( 1, 1,-1),
    vec3( 1, 1,-1), vec3(-1, 1,-1), vec3(-1, 1, 1),
    // -Y face (bottom)
    vec3(-1,-1,-1), vec3( 1,-1,-1), vec3( 1,-1, 1),
    vec3( 1,-1, 1), vec3(-1,-1, 1), vec3(-1,-1,-1),
    // +Z face (front)
    vec3(-1,-1, 1), vec3( 1,-1, 1), vec3( 1, 1, 1),
    vec3( 1, 1, 1), vec3(-1, 1, 1), vec3(-1,-1, 1),
    // -Z face (back)
    vec3( 1,-1,-1), vec3(-1,-1,-1), vec3(-1, 1,-1),
    vec3(-1, 1,-1), vec3( 1, 1,-1), vec3( 1,-1,-1)
);

void main() {
    vec3 pos = kCubeVerts[gl_VertexIndex];
    outDir   = pos;

    // Remove translation from view matrix so the sky stays at infinity.
    mat4 viewNoTrans = ubo.view;
    viewNoTrans[3]   = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 clipPos = ubo.proj * viewNoTrans * vec4(pos, 1.0);

    // Force depth to the far plane (z/w = 1.0) so the sky is always behind geometry.
    // Setting w=z achieves depth=1.0 after perspective divide (xyww trick).
    gl_Position = clipPos.xyww;
}
