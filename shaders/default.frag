#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragColor;
layout(location = 4) in flat uint fragMaterialId;
layout(location = 5) in float fragViewDepth;

layout(location = 0) out vec4 outColor;

// -- Uniform block (must match GlobalUBO in Application.h, std140) -------------
struct GPULight {
    vec4 position;    // xyz=pos or dir, w=type (0=dir, 1=point, 2=spot)
    vec4 color;       // rgb=color, a=intensity
    vec4 direction;   // xyz=direction, w=range
    vec4 spotAngles;  // x=cos(inner), y=cos(outer)
};

// Compile-time storage max. Runtime loop cap comes from specialization constant id 0.
#ifndef MAX_LIGHTS
#define MAX_LIGHTS 8
#endif
layout(constant_id = 0) const int LIGHT_CAP = MAX_LIGHTS;

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4     view;
    mat4     proj;
    vec4     ambientColor;  // rgb=color, a=intensity
    vec4     cameraPos;     // xyz=world position
    GPULight lights[MAX_LIGHTS];
    int      lightCount;
    float    _pad0;         // explicit float padding to match C++ float _pad[3]
    float    _pad1;
    float    _pad2;
    // CSM shadow fields -- must match Application.h GlobalUBO.
    mat4     lightSpaceMatrices[4];
    vec4     cascadeSplits;
    ivec4    shadowMeta;    // x=cascadeCount, y=enabled
    vec4     shadowParams;  // x=receiver bias
} ubo;

// Per-material albedo texture (set 1, binding 0)
layout(set = 1, binding = 0) uniform sampler2D albedoTex;

// Material SSBO (set=0, binding=3) -- must match GPUMaterial in Material.h (std430, 48 bytes)
struct GPUMaterial {
    vec4  baseColor;  // rgba tint applied to albedo
    float roughness;  // [0,1] reserved for PBR
    float metallic;   // [0,1] reserved for PBR
    float emissive;   // emissive glow multiplier (albedo * emissive added to output)
    uint  flags;      // bit0=unlit (skip lighting), bit1=alphaTest
    uint  _pad0, _pad1, _pad2, _pad3;  // reserved
};
layout(set = 0, binding = 3, std430) readonly buffer MaterialSSBO {
    GPUMaterial materials[];
} matSSBO;

// Shadow map array (set=0, binding=2) -- one map per cascade.
layout(set = 0, binding = 2) uniform sampler2D shadowMaps[4];

// -- Blinn-Phong per-light contribution ----------------------------------------
// Returns the combined diffuse + specular radiance contribution of one light.
// Uses the half-vector (Blinn) instead of reflect() (Phong) for the specular
// term  --  more physically correct at grazing angles and cheaper to evaluate.
vec3 blinnPhong(GPULight light, vec3 fragPos, vec3 normal, vec3 viewDir, vec3 albedo) {
    vec3  lightDir;
    float attenuation = 1.0;
    int   ltype = clamp(int(light.position.w + 0.5), 0, 2);

    if (ltype == 0) {
        // -- Directional: infinite distance, uniform direction, no attenuation --
        lightDir = normalize(-light.direction.xyz);

    } else {
        // -- Point / Spot: positional, distance-based attenuation --------------
        vec3  toLight = light.position.xyz - fragPos;
        float dist    = length(toLight);
        float safeDist = max(dist, 1e-4);
        lightDir      = toLight / safeDist;

        float range = max(light.direction.w, 1e-4);
        float ratio = clamp(dist / range, 0.0, 1.0);
        // Smooth inverse-square-like falloff that reaches 0 at range
        attenuation = clamp(1.0 - ratio * ratio, 0.0, 1.0);
        attenuation *= attenuation;

        if (ltype == 2) {
            // -- Spot: angular cone falloff on top of distance attenuation -----
            float cosAngle = dot(-lightDir, normalize(light.direction.xyz));
            float inner    = light.spotAngles.x; // cos(innerAngle)
            float outer    = light.spotAngles.y; // cos(outerAngle)
            // Smooth blend between inner (full) and outer (zero) cone edges
            attenuation *= clamp((cosAngle - outer) / max(inner - outer, 0.001), 0.0, 1.0);
        }
    }

    // -- Diffuse (Lambertian) ---------------------------------------------------
    float NdotL  = max(dot(normal, lightDir), 0.0);
    vec3  diffuse = NdotL * albedo;

    // -- Specular (Blinn-Phong half-vector) ------------------------------------
    // Half-vector between light and view; the epsilon prevents a zero-vector
    // normalize when lightDir and viewDir are perfectly opposite.
    // Shininess=64 is hardcoded; promote to a material UBO field to vary per object.
    vec3  halfVec  = normalize(lightDir + viewDir + vec3(1e-6));
    float NdotH    = max(dot(normal, halfVec), 0.0);
    float specular = pow(NdotH, 64.0) * 0.4;

    vec3 lightRadiance = light.color.rgb * light.color.a;
    return (diffuse + specular) * lightRadiance * attenuation;
}

// -- Manual 3x3 PCF shadow test ------------------------------------------------
// Returns 1.0 (fully lit) to 0.0 (fully shadowed).
vec2 cascadeTexelSize(int cascadeIdx) {
    if (cascadeIdx == 0) return 1.0 / vec2(textureSize(shadowMaps[0], 0));
    if (cascadeIdx == 1) return 1.0 / vec2(textureSize(shadowMaps[1], 0));
    if (cascadeIdx == 2) return 1.0 / vec2(textureSize(shadowMaps[2], 0));
    return 1.0 / vec2(textureSize(shadowMaps[3], 0));
}

float cascadeDepth(int cascadeIdx, vec2 uv) {
    if (cascadeIdx == 0) return texture(shadowMaps[0], uv).r;
    if (cascadeIdx == 1) return texture(shadowMaps[1], uv).r;
    if (cascadeIdx == 2) return texture(shadowMaps[2], uv).r;
    return texture(shadowMaps[3], uv).r;
}

float shadowTest(int cascadeIdx, vec3 fragWorldPos, float bias) {
    vec4 lightSpacePos = ubo.lightSpaceMatrices[cascadeIdx] * vec4(fragWorldPos, 1.0);
    vec3 projCoords    = lightSpacePos.xyz / lightSpacePos.w;
    // Convert from NDC [-1,1] to shadow map UV [0,1]
    vec2 shadowUV      = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    // Fragment is outside the shadow frustum -- consider it lit
    if (currentDepth > 1.0 || currentDepth < 0.0) return 1.0;

    float shadow    = 0.0;
    vec2  texelSize = cascadeTexelSize(cascadeIdx);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float shadowDepth = cascadeDepth(cascadeIdx, shadowUV + vec2(x, y) * texelSize);
            shadow += (currentDepth - bias > shadowDepth) ? 0.0 : 1.0;
        }
    }
    return shadow / 9.0;
}

void main() {
    // Fetch the material for this instance (index provided by vertex shader).
    GPUMaterial mat = matSSBO.materials[fragMaterialId];

    vec3 normal  = normalize(fragNormal);
    vec3 viewDir = normalize(ubo.cameraPos.xyz - fragWorldPos);

    // Albedo: texture sample * vertex color * material tint.
    vec4 texSample = texture(albedoTex, fragTexCoord) * fragColor;
    vec3 albedo    = texSample.rgb * mat.baseColor.rgb;
    float alpha    = texSample.a  * mat.baseColor.a;

    // Emissive: additive glow (not affected by lighting or shadows).
    vec3 emissive = albedo * mat.emissive;

    vec3 result;

    if ((mat.flags & 1u) != 0u) {
        // Unlit: skip all lighting; output albedo + emissive directly.
        result = albedo + emissive;
    } else {
        // -- Lit path (Blinn-Phong + shadow) -----------------------------------
        // Ambient term: not shadowed (shadow applies to direct light only)
        result = ubo.ambientColor.rgb * ubo.ambientColor.a * albedo;

        // Shadow visibility (1.0 = fully lit, 0.0 = fully in shadow)
        float shadowVis = 1.0;
        if (ubo.shadowMeta.y != 0 && fragViewDepth > 0.0) {
            int cascadeIdx = max(ubo.shadowMeta.x - 1, 0);
            for (int i = 0; i < ubo.shadowMeta.x; ++i) {
                if (fragViewDepth <= ubo.cascadeSplits[i]) {
                    cascadeIdx = i;
                    break;
                }
            }
            shadowVis = shadowTest(cascadeIdx, fragWorldPos, ubo.shadowParams.x);
        }

        // Accumulate direct light contributions, modulated by shadow
        int activeLights = min(ubo.lightCount, LIGHT_CAP);
        for (int i = 0; i < activeLights; ++i) {
            result += blinnPhong(ubo.lights[i], fragWorldPos, normal, viewDir, albedo) * shadowVis;
        }

        result += emissive;
    }

    outColor = vec4(result, alpha);
}
