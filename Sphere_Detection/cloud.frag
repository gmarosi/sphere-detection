#version 450

in float pointColor;
in vec3 out_pos;

out vec4 fs_out_col;

void main()
{
	// fs_out_col = vec4(1, 1 - pointColor, 1 - pointColor, 1);
	int asd = int(distance(vec2(0, 0), out_pos.xz) < 7 && out_pos.y < 1);
	fs_out_col = vec4(1, 1 - pointColor, (2 - pointColor - asd) / 2, 1);
}