#version 450

in vec4 pointPos;

uniform mat4 mvp;

void main()
{
	gl_Position = mvp * pointPos;
}