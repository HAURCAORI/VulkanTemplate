#version 450
#extension GL_EXT_nonuniform_qualifier : enable

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
    ivec4    iblParams;     // x=iblEnabled, y=maxPrefilterLod (packed as int)
} ubo;

// Bindless texture array (set=1, binding=0) -- G2 bindless path.
// All scene textures are registered here; accessed via per-instance indices in GPUMaterial.
layout(set = 1, binding = 0) uniform sampler2D bindlessTextures[];

// Material SSBO (set=0, binding=3) -- must match GPUMaterial in Material.h (std430, 48 bytes)
struct GPUMaterial {
    vec4  baseColor;     // rgba tint applied to albedo
    float roughness;     // [0,1] Cook-Torrance roughness
    float metallic;      // [0,1] metallic factor
    float emissive;      // emissive glow multiplier (albedo * emissive added to output)
    uint  flags;         // bit0=unlit, bit1=alphaTest, bit2=UsePBR, bit3=HasNormalMap
    uint  albedoTexIdx;  // bindless index for albedo texture
    uint  normalTexIdx;  // bindless index for normal map
    uint  _pad2, _pad3;  // reserved
};
layout(set = 0, binding = 3, std430) readonly buffer MaterialSSBO {
    GPUMaterial materials[];
} matSSBO;

// Shadow map array (set=0, binding=2) -- one map per cascade.
layout(set = 0, binding = 2) uniform sampler2D shadowMaps[4];

// IBL samplers (set=0, bindings 4,5,6)
layout(set = 0, binding = 4) uniform samplerCube iblIrradiance;   // diffuse irradiance
layout(set = 0, binding = 5) uniform samplerCube iblPrefilter;    // specular prefiltered env
layout(set = 0, binding = 6) uniform sampler2D   iblBrdfLut;      // BRDF integration LUT

// -- Blinn-Phong per-light contribution ----------------------------------------
vec3 blinnPhong(GPULight light, vec3 fragPos, vec3 normal, vec3 viewDir, vec3 albedo) {
    vec3  lightDir;
    float attenuation = 1.0;
    int   ltype = clamp(int(light.position.w + 0.5), 0, 2);

    if (ltype == 0) {
        lightDir = normalize(-light.direction.xyz);
    } else {
        vec3  toLight = light.position.xyz - fragPos;
        float dist    = length(toLight);
        float safeDist = max(dist, 1e-4);
        lightDir      = toLight / safeDist;

        float range = max(light.direction.w, 1e-4);
        float ratio = clamp(dist / range, 0.0, 1.0);
        attenuation = clamp(1.0 - ratio * ratio, 0.0, 1.0);
        attenuation *= attenuation;

        if (ltype == 2) {
            float cosAngle = dot(-lightDir, normalize(light.direction.xyz));
            float inner    = light.spotAngles.x;
            float outer    = light.spotAngles.y;
            attenuation *= clamp((cosAngle - outer) / max(inner - outer, 0.001), 0.0, 1.0);
        }
    }

    float NdotL  = max(dot(normal, lightDir), 0.0);
    vec3  diffuse = NdotL * albedo;

    vec3  halfVec  = normalize(lightDir + viewDir + vec3(1e-6));
    float NdotH    = max(dot(normal, halfVec), 0.0);
    float specular = pow(NdotH, 64.0) * 0.4;

    vec3 lightRadiance = light.color.rgb * light.color.a;
    return (diffuse + specular) * lightRadiance * attenuation;
}

// -- Cook-Torrance PBR BRDF functions ------------------------------------------

float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

float G_SmithGGX(float NdotV, float NdotL, float roughness) {
    float r  = roughness + 1.0;
    float k  = (r * r) / 8.0;
    float g1 = NdotV / (NdotV * (1.0 - k) + k);
    float g2 = NdotL / (NdotL * (1.0 - k) + k);
    return g1 * g2;
}

vec3 F_Schlick(float VdotH, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

vec3 F_SchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 cookTorrance(GPULight light, vec3 fragPos, vec3 normal, vec3 viewDir,
                  vec3 albedo, float roughness, float metallic) {
    vec3  lightDir;
    float attenuation = 1.0;
    int   ltype = clamp(int(light.position.w + 0.5), 0, 2);

    if (ltype == 0) {
        lightDir = normalize(-light.direction.xyz);
    } else {
        vec3  toLight  = light.position.xyz - fragPos;
        float dist     = length(toLight);
        float safeDist = max(dist, 1e-4);
        lightDir       = toLight / safeDist;

        float range = max(light.direction.w, 1e-4);
        float ratio = clamp(dist / range, 0.0, 1.0);
        attenuation = clamp(1.0 - ratio * ratio, 0.0, 1.0);
        attenuation *= attenuation;

        if (ltype == 2) {
            float cosAngle = dot(-lightDir, normalize(light.direction.xyz));
            float inner    = light.spotAngles.x;
            float outer    = light.spotAngles.y;
            attenuation *= clamp((cosAngle - outer) / max(inner - outer, 0.001), 0.0, 1.0);
        }
    }

    float NdotL = max(dot(normal, lightDir), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    vec3  halfVec = normalize(lightDir + viewDir);
    float NdotV   = max(dot(normal, viewDir), 0.0);
    float NdotH   = max(dot(normal, halfVec), 0.0);
    float VdotH   = max(dot(viewDir, halfVec), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float D = D_GGX(NdotH, max(roughness, 0.05));
    float G = G_SmithGGX(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(VdotH, F0);

    vec3 numerator    = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3  specular    = numerator / denominator;

    vec3 kd = (1.0 - F) * (1.0 - metallic);

    vec3 lightRadiance = light.color.rgb * light.color.a;
    return (kd * albedo / 3.14159265 + specular) * lightRadiance * attenuation * NdotL;
}

// -- Normal map perturbation (bindless variant) --------------------------------
// normalTexIdx: bindless array index for the normal map texture.
vec3 perturbNormal(vec3 worldNormal, vec3 worldPos, vec2 uv, uint normalTexIdx) {
    vec3 dPdx = dFdx(worldPos);
    vec3 dPdy = dFdy(worldPos);
    vec2 dUdx = dFdx(uv);
    vec2 dUdy = dFdy(uv);

    float det = dUdx.x * dUdy.y - dUdy.x * dUdx.y;
    float rcpDet = 1.0 / (abs(det) + 1e-8);

    vec3 T = rcpDet * (dUdy.y * dPdx - dUdx.y * dPdy);
    vec3 B = rcpDet * (dUdx.x * dPdy - dUdy.x * dPdx);

    vec3 N = normalize(worldNormal);
    T = normalize(T - dot(T, N) * N);
    B = normalize(B - dot(B, N) * N);

    // Sample the normal map from the bindless array using nonuniformEXT.
    vec3 tn = texture(bindlessTextures[nonuniformEXT(normalTexIdx)], uv).rgb * 2.0 - 1.0;

    return normalize(T * tn.x + B * tn.y + N * tn.z);
}

// -- IBL ambient contribution --------------------------------------------------
vec3 evalIBL(vec3 normal, vec3 viewDir, vec3 albedo, float roughness, float metallic, vec3 F0) {
    float NdotV = max(dot(normal, viewDir), 0.0);

    vec3 F     = F_SchlickRoughness(NdotV, F0, roughness);
    vec3 kd    = (1.0 - F) * (1.0 - metallic);

    vec3 irradiance = texture(iblIrradiance, normal).rgb;
    vec3 diffuse    = kd * irradiance * albedo;

    vec3  reflDir   = reflect(-viewDir, normal);
    float maxLod    = float(ubo.iblParams.y);
    float mipLevel  = roughness * maxLod;
    vec3  prefilteredColor = textureLod(iblPrefilter, reflDir, mipLevel).rgb;

    vec2 brdf = texture(iblBrdfLut, vec2(NdotV, roughness)).rg;
    vec3 specular = prefilteredColor * (F0 * brdf.x + brdf.y);

    return diffuse + specular;
}

// -- Manual 3x3 PCF shadow test ------------------------------------------------
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
    vec2 shadowUV      = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

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
    GPUMaterial mat = matSSBO.materials[fragMaterialId];

    vec3 normal = normalize(fragNormal);

    // Optionally perturb the surface normal using a tangent-space normal map.
    if ((mat.flags & 8u) != 0u) {
        normal = perturbNormal(normal, fragWorldPos, fragTexCoord, mat.normalTexIdx);
    }

    vec3 viewDir = normalize(ubo.cameraPos.xyz - fragWorldPos);

    // Albedo: sample from bindless array using per-instance albedoTexIdx.
    vec4 texSample = texture(bindlessTextures[nonuniformEXT(mat.albedoTexIdx)],
                             fragTexCoord) * fragColor;
    vec3 albedo    = texSample.rgb * mat.baseColor.rgb;
    float alpha    = texSample.a  * mat.baseColor.a;

    vec3 emissive = albedo * mat.emissive;

    vec3 result;

    if ((mat.flags & 1u) != 0u) {
        result = albedo + emissive;
    } else if ((mat.flags & 4u) != 0u) {
        // -- PBR lit path (Cook-Torrance BRDF) ---------------------------------
        float roughness = clamp(mat.roughness, 0.04, 1.0);
        float metallic  = clamp(mat.metallic,  0.0,  1.0);
        vec3  F0        = mix(vec3(0.04), albedo, metallic);

        if (ubo.iblParams.x != 0) {
            result = evalIBL(normal, viewDir, albedo, roughness, metallic, F0);
        } else {
            result = ubo.ambientColor.rgb * ubo.ambientColor.a * albedo;
        }

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

        int activeLights = min(ubo.lightCount, LIGHT_CAP);
        for (int i = 0; i < activeLights; ++i) {
            result += cookTorrance(ubo.lights[i], fragWorldPos, normal, viewDir,
                                   albedo, roughness, metallic) * shadowVis;
        }

        result += emissive;
    } else {
        // -- Lit path (Blinn-Phong + shadow) -----------------------------------
        result = ubo.ambientColor.rgb * ubo.ambientColor.a * albedo;

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

        int activeLights = min(ubo.lightCount, LIGHT_CAP);
        for (int i = 0; i < activeLights; ++i) {
            result += blinnPhong(ubo.lights[i], fragWorldPos, normal, viewDir, albedo) * shadowVis;
        }

        result += emissive;
    }

    outColor = vec4(result, alpha);

    // -- E3: Debug visualization overlay (iblParams.z = debugVizMode) ----------
    int vizMode = ubo.iblParams.z;

    if (vizMode == 1) {
        if (ubo.shadowMeta.y != 0) {
            int cascadeIdx = max(ubo.shadowMeta.x - 1, 0);
            for (int i = 0; i < ubo.shadowMeta.x; ++i) {
                if (fragViewDepth <= ubo.cascadeSplits[i]) {
                    cascadeIdx = i;
                    break;
                }
            }
            vec3 cascadeColors[4];
            cascadeColors[0] = vec3(1.0, 0.2, 0.2);
            cascadeColors[1] = vec3(0.2, 1.0, 0.2);
            cascadeColors[2] = vec3(0.2, 0.4, 1.0);
            cascadeColors[3] = vec3(1.0, 1.0, 0.2);
            outColor = vec4(cascadeColors[clamp(cascadeIdx, 0, 3)], 1.0);
        } else {
            outColor = vec4(0.5, 0.5, 0.5, 1.0);
        }
    } else if (vizMode == 2) {
        outColor = vec4(normal * 0.5 + 0.5, 1.0);
    } else if (vizMode == 3) {
        outColor = vec4(mat.metallic, mat.roughness, 0.0, 1.0);
    }
}
