#include "PointCloud.h"

PointCloud::PointCloud()
{
	mapMem = nullptr;
}

void PointCloud::Init(const MemoryNames& memNames)
{
	mapMem = new SHMManager(memNames.first, memNames.second, sizeof(int), POINT_CLOUD_SIZE * CHANNELS * sizeof(int));

	pointsPos.resize(POINT_CLOUD_SIZE);
	pointsIntensity.resize(POINT_CLOUD_SIZE);

	posVBO.BufferData(pointsPos);
	intensityVBO.BufferData(pointsIntensity);
	cloudVAO.Init({
		{ CreateAttribute<0, glm::vec3, 0, sizeof(glm::vec3)>, posVBO },
		{ CreateAttribute<1, float, 0, sizeof(float)>, intensityVBO }
	});

	program.AttachShaders({
		{GL_VERTEX_SHADER,	 "cloud.vert"},
		{GL_FRAGMENT_SHADER, "cloud.frag"}
	});

	program.BindAttribLocations({
		{0, "pointPos"},
		{1, "pointIntensity"}
	});

	program.LinkProgram();
}

void PointCloud::Update()
{
	if (mapMem->hasBufferChanged())
	{
		std::vector<float> rawData;
		rawData.resize(POINT_CLOUD_SIZE * CHANNELS);
		mapMem->readData(rawData.data());

		for (size_t i = 0; i < POINT_CLOUD_SIZE * CHANNELS; i += CHANNELS)
		{
			auto tmp = glm::vec4(rawData[i], rawData[i + 2], -rawData[i + 1], 1);
			pointsPos[i / CHANNELS] = glm::vec3(tmp.x, tmp.y, tmp.z);
			pointsIntensity[i / CHANNELS] = rawData[i + 3];
		}

		posVBO.BufferData(pointsPos);
		intensityVBO.BufferData(pointsIntensity);
	}
}

void PointCloud::Render(const glm::mat4& viewProj)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointSize(pointRenderSize);

	program.Use();
	cloudVAO.Bind();

	glm::mat4 cloudWorld(1.0f);
	program.SetUniform("mvp", viewProj * cloudWorld);

	glDrawArrays(GL_POINTS, 0, POINT_CLOUD_SIZE);

	program.Unuse();

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
}