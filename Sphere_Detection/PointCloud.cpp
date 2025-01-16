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
}

void PointCloud::Update(const glm::mat4& viewProj)
{
	matViewProj = viewProj;

	if (mapMem->hasBufferChanged())
	{
		std::vector<float> rawData;
		rawData.resize(POINT_CLOUD_SIZE * CHANNELS);
		mapMem->readData(rawData.data());

		for (size_t i = 0; i < POINT_CLOUD_SIZE * CHANNELS; i += CHANNELS)
		{
			pointsPos[i / CHANNELS] = glm::vec3(rawData[i], rawData[i + 2], -rawData[i + 1]);
			pointsIntensity[i / CHANNELS] = rawData[i + 3];
		}

		posVBO.BufferData(pointsPos);
		intensityVBO.BufferData(pointsIntensity);
	}
}

void PointCloud::Render()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointSize(pointRenderSize);

	program.Use();
	program.SetUniform("mvp", matViewProj * glm::mat4(1.f));
	cloudVAO.Bind();

	glDrawArrays(GL_POINTS, 0, POINT_CLOUD_SIZE);

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
}