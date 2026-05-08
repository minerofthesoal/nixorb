// assets/shaders/orb_glow.frag
// Compile with: qsb --glsl 100es,120,150 --hlsl 50 --msl 12 orb_glow.frag -o orb_glow.frag.qsb
// Voronoi-based animated glow for the orb.
// Passed from Qt QML ShaderEffect.

#version 440

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 qt_Matrix;
    float qt_Opacity;
    float time;
    float amplitude;
    vec4  baseColor;
};

// ------------------------------------------------------------------ //
//  Utility                                                            //
// ------------------------------------------------------------------ //
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec2 hash2(vec2 p) {
    return vec2(hash(p), hash(p + vec2(57.0, 131.0)));
}

// Animated voronoi — returns distance to nearest cell center
float voronoi(vec2 uv, float t) {
    vec2 i = floor(uv);
    vec2 f = fract(uv);
    float minDist = 1e10;

    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 cell = vec2(float(x), float(y));
            vec2 seed = hash2(i + cell);
            // Animate cell centers
            vec2 center = cell + 0.5 + 0.45 * sin(t * 0.6 + 6.28318 * seed);
            float d = length(f - center);
            minDist = min(minDist, d);
        }
    }
    return minDist;
}

// ------------------------------------------------------------------ //
//  Main                                                               //
// ------------------------------------------------------------------ //
void main() {
    // Normalized coords centered at (0,0)
    vec2 uv = qt_TexCoord0 * 2.0 - 1.0;
    float dist = length(uv);

    // Discard outside circle
    if (dist > 1.0) {
        fragColor = vec4(0.0);
        return;
    }

    // ---- Core radial glow ----
    float coreGlow  = pow(1.0 - dist, 2.5 + amplitude * 1.5);
    float edgeGlow  = pow(1.0 - dist, 0.8) * 0.4;
    float glow      = coreGlow + edgeGlow;

    // ---- Voronoi texture on surface ----
    vec2 vUV = uv * 3.0;
    float vor = voronoi(vUV, time);
    float vorPattern = smoothstep(0.3, 0.8, vor);

    // ---- Pulse wave from center ----
    float wave = sin(dist * 12.0 - time * 2.5) * 0.5 + 0.5;
    wave *= (1.0 - dist) * amplitude * 0.6;

    // ---- Combine ----
    float brightness = glow * 0.8 + vorPattern * 0.15 + wave;
    brightness *= (0.85 + amplitude * 0.3);

    // ---- Color: base color tinted toward white at peak ----
    vec3 col = mix(baseColor.rgb, vec3(1.0), amplitude * 0.35);
    col = mix(col * 0.3, col, brightness);

    // ---- Alpha: soft edge fade ----
    float alpha = smoothstep(1.0, 0.85, dist) * qt_Opacity * (0.7 + amplitude * 0.3);

    fragColor = vec4(col * brightness, alpha);
}
