#version 450

in vec4 pointPos;
in float pointIntensity;

out float pointColor;
out vec3 out_pos;

uniform mat4 mvp;

void main()
{
	gl_Position = mvp * vec4(pointPos.xyz, 1);

	pointColor = pointPos.w;
	out_pos = pointPos.xyz;
}