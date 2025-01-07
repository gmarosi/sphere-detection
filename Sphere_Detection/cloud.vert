#version 450

in vec3 pointPos;
in float pointIntensity;

uniform mat4 mvp;

void main()
{
	gl_Position = mvp * vec4(pointPos, 1);
}