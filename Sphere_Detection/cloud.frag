#version 450

in float pointColor;
in vec3 out_pos;

out vec4 fs_out_col;

void main()
{
	// fs_out_col = vec4(1, 1 - pointColor, 1 - pointColor, 1);
	
	float dist = distance(vec2(0, 0), out_pos.xz);
	int asd = int(dist > 3 && dist < 7 && out_pos.y < 1);
	fs_out_col = vec4(1, 1 - pointColor, (2 - pointColor - asd) / 2, 1);
	
}