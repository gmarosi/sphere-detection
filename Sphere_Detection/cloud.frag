#version 450

in float pointColor;

out vec4 fs_out_col;

void main()
{
	fs_out_col = pointColor == 1 ? vec4(1, 0, 0, 1) : vec4(1, 1, 1, 1);
}