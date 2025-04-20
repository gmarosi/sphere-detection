#version 450

in vec4 pointPos;

out float pointColor;

uniform mat4 mvp;

void main()
{
	gl_Position = mvp * vec4(pointPos.xyz, 1);

	pointColor = pointPos.w;
}