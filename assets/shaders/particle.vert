// assets/shaders/particle.vert
// Compile: qsb --glsl "100es,120,150" --hlsl 50 --msl 12 -o particle.vert.qsb particle.vert
#version 440

layout(location = 0) in vec4 qt_Vertex;
layout(location = 1) in vec2 qt_MultiTexCoord0;
layout(location = 0) out vec2 qt_TexCoord0;
layout(location = 1) out float vAlpha;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    float time;
    float amplitude;
};

void main() {
    qt_TexCoord0 = qt_MultiTexCoord0;
    // Particle fade based on distance from center
    vec2 center = vec2(0.5, 0.5);
    float dist = length(qt_MultiTexCoord0 - center) * 2.0;
    vAlpha = max(0.0, 1.0 - dist) * qt_Opacity * amplitude;
    gl_Position = qt_Matrix * qt_Vertex;
}
