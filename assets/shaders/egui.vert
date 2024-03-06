#version 460
#include "draw.playout"

layout(location = 0) in vec2 in_Position;
layout(location = 1) in vec2 in_UV;
layout(location = 2) in vec4 in_Color;
layout(location = 0) out vec4 out_Color;
layout(location = 1) out vec2 out_UV;



void main(){
    vec2 size = vec2(1280.0, 720.0);
    gl_Position = vec4(2.0 * in_Position / u_transform - 1.0, 0.0, 1.0);
    out_Color = in_Color;
    out_UV = in_UV;
}
