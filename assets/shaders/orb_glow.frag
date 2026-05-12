// assets/shaders/orb_glow.frag
// Compile: qsb --glsl "100es,120,150" --hlsl 50 --msl 12 -o orb_glow.frag.qsb orb_glow.frag
#version 440

layout(location = 0) in  vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    float time;       // seconds, loops 0→2π
    float amplitude;  // 0.0–1.0  TTS audio RMS
    vec4  baseColor;  // state colour from OrbBridge
};

// ── Hash / noise helpers ────────────────────────────────────────── //
float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

// Animated Voronoi — distance to nearest jittered cell centre
float voronoi(vec2 uv, float t) {
    vec2 cell = floor(uv);
    vec2 frac = fract(uv);
    float minD = 8.0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec2  offset = vec2(float(x), float(y));
            vec2  seed   = hash22(cell + offset);
            // Animate each cell centre with a unique phase
            vec2  centre = offset + 0.5 + 0.42 * sin(t * 0.8 + 6.2832 * seed);
            float d      = length(frac - centre);
            minD = min(minD, d);
        }
    }
    return minD;
}

// Smooth fbm-like value noise
float fbm(vec2 p, float t) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 3; i++) {
        v   += amp * voronoi(p, t + float(i) * 1.7);
        p   *= 2.1;
        amp *= 0.45;
    }
    return v;
}

// ── Main ────────────────────────────────────────────────────────── //
void main() {
    // Normalised coords centred at (0,0), range [-1, 1]
    vec2  uv   = qt_TexCoord0 * 2.0 - 1.0;
    float dist = length(uv);

    // Circular clip — everything outside the orb is transparent
    if (dist > 1.02) {
        fragColor = vec4(0.0);
        return;
    }

    // ── Radial glow layers ───────────────────────────────────────── //
    float core = pow(max(0.0, 1.0 - dist), 2.2 + amplitude * 1.8);
    float halo = pow(max(0.0, 1.0 - dist), 0.7) * 0.35;
    float glow = core + halo;

    // ── Voronoi surface texture ──────────────────────────────────── //
    vec2  vUV    = uv * 3.5 + vec2(time * 0.05);
    float vor    = fbm(vUV, time);
    float pattern = smoothstep(0.25, 0.75, vor) * (1.0 - dist * 0.8);

    // ── Outward pulse rings (audio-reactive) ─────────────────────── //
    float ring1 = sin(dist * 14.0 - time * 3.1) * 0.5 + 0.5;
    float ring2 = sin(dist * 22.0 - time * 4.7 + 1.2) * 0.5 + 0.5;
    float rings = (ring1 * 0.6 + ring2 * 0.4) * (1.0 - dist) * amplitude * 0.7;

    // ── Sparkle — small bright flecks at the surface ─────────────── //
    float fleck  = hash11(floor(dist * 18.0 + time * 0.3) * 73.1 +
                          floor(atan(uv.y, uv.x) * 9.0) * 17.3);
    float sparkle = step(0.96, fleck) * (1.0 - dist) * amplitude * 1.5;

    // ── Combine brightness ───────────────────────────────────────── //
    float brightness = glow * 0.75
                     + pattern * 0.18
                     + rings
                     + sparkle;
    brightness *= 0.85 + amplitude * 0.40;

    // ── Colour: state colour → white at amplitude peaks ──────────── //
    vec3 col = mix(baseColor.rgb, vec3(1.0, 1.0, 1.0), amplitude * 0.30);
    col = col * brightness;

    // ── Alpha: soft circular edge fade ───────────────────────────── //
    float alpha = smoothstep(1.01, 0.82, dist)
                * qt_Opacity
                * (0.72 + amplitude * 0.28);

    fragColor = vec4(col, alpha);
}
