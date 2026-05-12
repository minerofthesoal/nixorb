// assets/shaders/particle.frag
// Compile: qsb --glsl "100es,120,150" --hlsl 50 --msl 12 -o particle.frag.qsb particle.frag
#version 440

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 1) in float vAlpha;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    float time;
    float amplitude;
    vec4  color;
};

void main() {
    vec2  uv   = qt_TexCoord0 * 2.0 - 1.0;
    float dist = length(uv);
    float glow = exp(-dist * dist * 4.0);
    fragColor = vec4(color.rgb * glow, glow * vAlpha * color.a);
}
