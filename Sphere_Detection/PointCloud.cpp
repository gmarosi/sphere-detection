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

	// Setup VAO & VBOs
	glGenVertexArrays(1, &cloudVAO);
	glBindVertexArray(cloudVAO);

	glGenBuffers(1, &posVBO);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferData( GL_ARRAY_BUFFER,
		POINT_CLOUD_SIZE * sizeof(glm::vec3),
		nullptr,
		GL_DYNAMIC_DRAW
	);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		(GLuint)0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glGenBuffers(1, &intensityVBO);
	glBindBuffer(GL_ARRAY_BUFFER, intensityVBO);
	glBufferData(GL_ARRAY_BUFFER,
		POINT_CLOUD_SIZE * sizeof(float),
		nullptr,
		GL_DYNAMIC_DRAW
	);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		(GLuint)1,
		1,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glBindVertexArray(0);
	glBindVertexArray(0);

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

		glBindBuffer(GL_ARRAY_BUFFER, posVBO);
		glBufferData(GL_ARRAY_BUFFER, POINT_CLOUD_SIZE * sizeof(glm::vec3), pointsPos.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, intensityVBO);
		glBufferData(GL_ARRAY_BUFFER, POINT_CLOUD_SIZE * sizeof(float), pointsIntensity.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void PointCloud::Render(const glm::mat4& viewProj)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointSize(pointRenderSize);

	program.Use();
	glBindVertexArray(cloudVAO);

	glm::mat4 cloudWorld(1.0f);
	program.SetUniform("mvp", viewProj * cloudWorld);

	glDrawArrays(GL_POINTS, 0, POINT_CLOUD_SIZE);

	glBindVertexArray(0);
	program.Unuse();

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
}