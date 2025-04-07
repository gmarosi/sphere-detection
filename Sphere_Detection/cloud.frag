#version 450

in float pointColor;

out vec4 fs_out_col;

void main()
{
	fs_out_col = vec4(1, 1 - pointColor, 1 - pointColor, 1);
}