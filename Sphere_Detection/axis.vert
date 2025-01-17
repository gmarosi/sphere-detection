#version 450

out vec4 vs_out_col;

vec3 pos[6] =
{
	{0, 0, 0},{1, 0, 0},
	{0, 0, 0},{0, 1, 0},
	{0, 0, 0},{0, 0, 1}
};

uniform mat4 MVP;

void main()
{
	gl_Position = MVP * vec4(pos[gl_VertexID], 1);

	if(gl_VertexID == 0 || gl_VertexID == 1)
		vs_out_col = vec4(pos[1], 1);
	else if(gl_VertexID == 2 || gl_VertexID == 3)
		vs_out_col = vec4(pos[3], 1);
	else if(gl_VertexID == 4 || gl_VertexID == 5)
		vs_out_col = vec4(pos[5], 1);
}