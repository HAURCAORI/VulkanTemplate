#version 450

layout(location = 0) in  vec3 inDir;
layout(location = 0) out vec4 outColor;

// Sky push constants: 32 bytes.
layout(push_constant) uniform SkyPush {
    vec4 skyColor;  // top/horizon color used in gradient mode
    int  skyMode;   // 0=color gradient, 1=cubemap, 2=procedural sky sphere
    int  _pad0;
    int  _pad1;
    int  _pad2;
} push;

// Cubemap for skybox mode (set=1, binding=0).
layout(set = 1, binding = 0) uniform samplerCube skyCubemap;

void main() {
    if (push.skyMode == 1) {
        // Cubemap mode: sample the environment map directly.
        outColor = texture(skyCubemap, normalize(inDir));
    } else if (push.skyMode == 2) {
        vec3 dir = normalize(inDir);
        float horizon = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
        float sunGlow = pow(max(dot(dir, normalize(vec3(0.25, 0.92, 0.18))), 0.0), 96.0);
        vec3 zenith   = push.skyColor.rgb;
        vec3 horizonColor = mix(vec3(0.18, 0.22, 0.28), zenith, 0.6);
        vec3 groundColor  = vec3(0.06, 0.05, 0.05);
        vec3 color = mix(groundColor, horizonColor, smoothstep(0.0, 0.45, horizon));
        color = mix(color, zenith, smoothstep(0.35, 1.0, horizon));
        color += vec3(1.0, 0.82, 0.55) * sunGlow * 0.35;
        outColor = vec4(color, 1.0);
    } else {
        // Gradient mode: blend ground -> horizon -> sky top based on Y component.
        float t = clamp(normalize(inDir).y * 0.5 + 0.5, 0.0, 1.0);
        vec3 groundColor  = vec3(0.1, 0.08, 0.06);
        vec3 horizonColor = vec3(0.5, 0.6, 0.7);
        vec3 skyTop       = push.skyColor.rgb;
        vec3 color = mix(mix(groundColor, horizonColor, t), skyTop, t * t);
        outColor   = vec4(color, 1.0);
    }
}
