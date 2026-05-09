// assets/shaders/orb_glow.vert
// Compile: qsb --glsl "100es,120,150" --hlsl 50 --msl 12 -o orb_glow.vert.qsb orb_glow.vert
#version 440

layout(location = 0) in vec4 qt_Vertex;
layout(location = 1) in vec2 qt_MultiTexCoord0;
layout(location = 0) out vec2 qt_TexCoord0;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    float time;
    float amplitude;
    vec4  baseColor;
};

void main() {
    qt_TexCoord0 = qt_MultiTexCoord0;
    gl_Position  = qt_Matrix * qt_Vertex;
}
