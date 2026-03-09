#version 450

layout(location = 0) in  vec2 inUV;
layout(location = 0) out vec4 outColor;

// HDR scene color (R16G16B16A16_SFLOAT, rendered in the geometry pass).
layout(set = 0, binding = 0) uniform sampler2D hdrInput;

// Per-blit parameters sent as push constants (16 bytes).
layout(push_constant) uniform PostPush {
    float exposure;     // linear multiplier applied before tonemapping
    float gamma;        // output gamma (typically 2.2)
    int   tonemapMode;  // 0=Reinhard, 1=ACES approx, 2=clamp only
    float _pad;
} push;

// Reinhard tonemapping (simple, gentle highlights).
vec3 tonemapReinhard(vec3 hdr) {
    return hdr / (hdr + vec3(1.0));
}

// Narkowicz 2015 ACES film approximation -- punchy look, good contrast.
vec3 tonemapAces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(hdrInput, inUV).rgb * push.exposure;

    vec3 ldr;
    if (push.tonemapMode == 1) {
        ldr = tonemapAces(hdr);
    } else if (push.tonemapMode == 0) {
        ldr = tonemapReinhard(hdr);
    } else {
        // Mode 2: simple clamp -- no tonemapping, useful for debugging.
        ldr = clamp(hdr, 0.0, 1.0);
    }

    // Gamma correction: linear -> sRGB.
    outColor = vec4(pow(ldr, vec3(1.0 / push.gamma)), 1.0);
}
