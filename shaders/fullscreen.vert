#version 450

// Fullscreen triangle -- no vertex buffer.
// Positions and UVs are derived entirely from gl_VertexIndex so no VkBuffer is needed.
//
// Vertex layout (NDC, Vulkan Y-down):
//   index 0: uv=(0,0)  pos=(-1,-1)  -- top-left
//   index 1: uv=(2,0)  pos=( 3,-1)  -- far right (off-screen)
//   index 2: uv=(0,2)  pos=(-1, 3)  -- far bottom (off-screen)
// The triangle covers the entire [0,1]x[0,1] UV region visible on screen.

layout(location = 0) out vec2 outUV;

void main() {
    // Bit trick to produce (0,0), (2,0), (0,2) from indices 0, 1, 2.
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    outUV       = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
