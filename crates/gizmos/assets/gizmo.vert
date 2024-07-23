#version 460

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec4 in_Color;
layout(location = 0) out vec4 out_Color;

vec3 srgbToLinear(vec3 sRGB)
{
	bvec3 cutoff = lessThan(sRGB, vec3(0.04045));
	vec3 higher = pow((sRGB + vec3(0.055))/vec3(1.055), vec3(2.4));
	vec3 lower = sRGB/vec3(12.92);

	return mix(higher, lower, cutoff);
}

void main(){
    gl_Position = vec4(in_Position, 1.0);
    out_Color = vec4(srgbToLinear(in_Color.xyz), in_Color.w);
}
