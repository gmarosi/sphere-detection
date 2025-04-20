#include "PointCloud.h"
#include <fstream>
#include <chrono>
#include <random>

inline unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}

PointCloud::PointCloud()
{
	mapMem = nullptr;
}

void PointCloud::ChangeMode()
{
	switch (fitMode)
	{
	case SPHERE:
		fitMode = CYLINDER;
		currentFitter = cylinderFitter;
		break;
	case CYLINDER:
		fitMode = SPHERE;
		currentFitter = sphereFitter;
		break;
	}
}

bool PointCloud::Init(const MemoryNames& memNames)
{
	mapMem = new SHMManager(memNames.first, memNames.second, sizeof(int), POINT_CLOUD_SIZE * CHANNELS * sizeof(int));

	// Setting up point cloud rendering
	// Setup VAO & VBOs
	glGenVertexArrays(1, &cloudVAO);
	glBindVertexArray(cloudVAO);

	// position buffer
	glGenBuffers(1, &posVBO);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferData( GL_ARRAY_BUFFER,
		POINT_CLOUD_SIZE * sizeof(glm::vec4),
		nullptr,
		GL_DYNAMIC_DRAW
	);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		(GLuint)0,
		4,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	program = glCreateProgram();
	{
		// vertex shader
		GLuint vert = glCreateShader(GL_VERTEX_SHADER);
		std::ifstream vertFile("cloud.vert");
		if (!vertFile.is_open())
		{
			std::cout << "Could not open cloud.vert" << std::endl;
			exit(1);
		}
		std::string vertCode(std::istreambuf_iterator<char>(vertFile), (std::istreambuf_iterator<char>()));
		const char* vertPtr = vertCode.c_str();
		vertFile.close();

		glShaderSource(vert, 1, &vertPtr, nullptr);
		glCompileShader(vert);

		// fragment shader
		GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
		std::ifstream fragFile("cloud.frag");
		if (!fragFile.is_open())
		{
			std::cout << "Could not open cloud.frag" << std::endl;
			exit(1);
		}
		std::string fragCode(std::istreambuf_iterator<char>(fragFile), (std::istreambuf_iterator<char>()));
		const char* fragPtr = fragCode.c_str();
		fragFile.close();

		glShaderSource(frag, 1, &fragPtr, nullptr);
		glCompileShader(frag);

		glAttachShader(program, vert);
		glAttachShader(program, frag);

		glBindAttribLocation(program, 0, "pointPos");

		glLinkProgram(program);
		glDeleteShader(vert);
		glDeleteShader(frag);
	}

	// Setup of sphere rendering
	SphereInit();

	// Setup of cylinder rendering
	CylinderInit();

	return true;
}

void PointCloud::SphereInit()
{
	glGenVertexArrays(1, &sphVAO);
	glBindVertexArray(sphVAO);

	glGenBuffers(1, &sphVBO);
	glBindBuffer(GL_ARRAY_BUFFER, sphVBO);
	glBufferData(GL_ARRAY_BUFFER,
		(vCount + 1) * (hCount + 1) * sizeof(glm::vec4),
		nullptr,
		GL_DYNAMIC_DRAW
	);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		(GLuint)0,
		4,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glGenBuffers(1, &sphIds);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphIds);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		3 * 2 * vCount * hCount,
		nullptr,
		GL_DYNAMIC_DRAW
	);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	sphProgram = glCreateProgram();

	// vertex shader
	GLuint vert = glCreateShader(GL_VERTEX_SHADER);
	std::ifstream vertFile("sphere.vert");
	if (!vertFile.is_open())
	{
		std::cout << "Could not open sphere.vert" << std::endl;
		exit(1);
	}
	std::string vertCode(std::istreambuf_iterator<char>(vertFile), (std::istreambuf_iterator<char>()));
	const char* vertPtr = vertCode.c_str();
	vertFile.close();

	glShaderSource(vert, 1, &vertPtr, nullptr);
	glCompileShader(vert);

	// fragment shader
	GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
	std::ifstream fragFile("sphere.frag");
	if (!fragFile.is_open())
	{
		std::cout << "Could not open sphere.frag" << std::endl;
		exit(1);
	}
	std::string fragCode(std::istreambuf_iterator<char>(fragFile), (std::istreambuf_iterator<char>()));
	const char* fragPtr = fragCode.c_str();
	fragFile.close();

	glShaderSource(frag, 1, &fragPtr, nullptr);
	glCompileShader(frag);

	glAttachShader(sphProgram, vert);
	glAttachShader(sphProgram, frag);

	glBindAttribLocation(sphProgram, 0, "pointPos");

	glLinkProgram(sphProgram);
	glDeleteShader(vert);
	glDeleteShader(frag);
}

void PointCloud::CylinderInit()
{
	glGenVertexArrays(1, &cylVAO);
	glBindVertexArray(cylVAO);

	glGenBuffers(1, &cylVBO);
	glBindBuffer(GL_ARRAY_BUFFER, cylVBO);
	glBufferData(GL_ARRAY_BUFFER,
		(cCount + 1) * sizeof(glm::vec4),
		nullptr,
		GL_DYNAMIC_DRAW
	);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		(GLuint)0,
		4,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glGenBuffers(1, &cylIds);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylIds);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		3 * 2 * cCount,
		nullptr,
		GL_DYNAMIC_DRAW
	);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	cylProgram = glCreateProgram();

	// vertex shader
	GLuint vert = glCreateShader(GL_VERTEX_SHADER);
	std::ifstream vertFile("cylinder.vert");
	if (!vertFile.is_open())
	{
		std::cout << "Could not open cylinder.vert" << std::endl;
		exit(1);
	}
	std::string vertCode(std::istreambuf_iterator<char>(vertFile), (std::istreambuf_iterator<char>()));
	const char* vertPtr = vertCode.c_str();
	vertFile.close();

	glShaderSource(vert, 1, &vertPtr, nullptr);
	glCompileShader(vert);

	// fragment shader
	GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
	std::ifstream fragFile("cylinder.frag");
	if (!fragFile.is_open())
	{
		std::cout << "Could not open cylinder.frag" << std::endl;
		exit(1);
	}
	std::string fragCode(std::istreambuf_iterator<char>(fragFile), (std::istreambuf_iterator<char>()));
	const char* fragPtr = fragCode.c_str();
	fragFile.close();

	glShaderSource(frag, 1, &fragPtr, nullptr);
	glCompileShader(frag);

	glAttachShader(cylProgram, vert);
	glAttachShader(cylProgram, frag);

	glBindAttribLocation(cylProgram, 0, "pointPos");

	glLinkProgram(cylProgram);
	glDeleteShader(vert);
	glDeleteShader(frag);
}

bool PointCloud::InitCl(cl::Context& context, const cl::vector<cl::Device>& devices)
{
	try
	{
		posBuffer = cl::BufferGL(context, CL_MEM_READ_WRITE, posVBO);

		sphereFitter = new SphereFitter();
		sphereFitter->Init(context, devices);

		cylinderFitter = new CylinderFitter();
		cylinderFitter->Init(context, devices);

		currentFitter = sphereFitter;
	}
	catch (cl::Error& error)
	{
		std::cout << "PointCloud::InitCl(): " << error.what() << std::endl;
		return false;
	}

	return true;
}

void PointCloud::Update()
{
	if (mapMem->hasBufferChanged())
	{
		fit = true;
		std::vector<float> rawData;
		std::vector<glm::vec4> pointsPos;
		rawData.resize(POINT_CLOUD_SIZE * CHANNELS);
		pointsPos.resize(POINT_CLOUD_SIZE);
		mapMem->readData(rawData.data());

		for (size_t i = 0; i < POINT_CLOUD_SIZE * CHANNELS; i += CHANNELS)
		{
			// last coordinate will be used by OpenCL kernel
			glm::vec4 point = glm::vec4(rawData[i], rawData[i + 2], -rawData[i + 1], 0);
			pointsPos[i / CHANNELS] = point;

			// storing indices of candidate points
			currentFitter->EvalCandidate(point, i / CHANNELS);
		}

		glBindBuffer(GL_ARRAY_BUFFER, posVBO);
		glBufferData(GL_ARRAY_BUFFER, POINT_CLOUD_SIZE * sizeof(glm::vec4), pointsPos.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void PointCloud::Render(const glm::mat4& viewProj)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointSize(pointRenderSize);

	glUseProgram(program);
	glBindVertexArray(cloudVAO);

	glm::mat4 cloudWorld(1.0f);
	glm::mat4 mvp = viewProj * cloudWorld;
	GLuint matrix = glGetUniformLocation(program, "mvp");
	glUniformMatrix4fv(matrix, 1, GL_FALSE, glm::value_ptr(mvp));

	glDrawArrays(GL_POINTS, 0, POINT_CLOUD_SIZE);

	glBindVertexArray(0);
	glUseProgram(0);

	if (foundFit)
	{
		switch (fitMode)
		{
		case SPHERE:
			RenderSphere(viewProj);
			break;
		case CYLINDER:
			RenderCylinder(viewProj);
			break;
		}
	}

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
}

void PointCloud::Fit(cl::CommandQueue& queue)
{
	if (!fit)
		return;
	fit = false;

	try
	{
		fitResult = currentFitter->Fit(queue, posBuffer);
		foundFit = true;
	}
	catch (cl::Error&)
	{
		exit(1);
	}
}

void PointCloud::RenderSphere(const glm::mat4& viewProj) const
{
	float x0 = fitResult.x;
	float y0 = fitResult.y;
	float z0 = fitResult.z;
	float r  = fitResult.w;
	constexpr float pi = glm::pi<float>();

	std::vector<glm::vec4> vertices;
	for (int i = 0; i <= hCount; i++)
	{
		float h = i / (float)hCount;
		float theta = 2 * pi * h;
		float costh = cosf(theta);
		float sinth = sinf(theta);

		for (int j = 0; j <= vCount; j++)
		{
			float v = j / (float)vCount;
			float phi = v * pi;
			
			vertices.push_back({
				x0 + r * sinf(phi) * costh,
				y0 + r * cosf(phi),
				z0 + r * sinf(phi) * sinth,
				1
			});
		}
	}

	std::vector<unsigned int> indices;
	for (int i = 0; i < hCount; i++)
	{
		for (int j = 0; j < vCount; j++)
		{
			// two triangles per segment
			indices.push_back(i		 + j		* (hCount + 1));
			indices.push_back((i + 1) + j		* (hCount + 1));
			indices.push_back(i		 + (j + 1)	* (hCount + 1));
			indices.push_back((i + 1) + j		* (hCount + 1));
			indices.push_back((i + 1) + (j + 1) * (hCount + 1));
			indices.push_back(i		 + (j + 1)  * (hCount + 1));
		}
	}
	
	glBindBuffer(GL_ARRAY_BUFFER, sphVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphIds);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec4), vertices.data(), GL_DYNAMIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(sphProgram);
	glBindVertexArray(sphVAO);

	glm::mat4 world(1.0f);
	glm::mat4 mvp = viewProj * world;
	GLuint matrix = glGetUniformLocation(sphProgram, "mvp");
	glUniformMatrix4fv(matrix, 1, GL_FALSE, glm::value_ptr(mvp));

	glDrawElements(GL_TRIANGLES, 3 * 2 * (vCount + 1) * (hCount + 1), GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
	glUseProgram(0);
}

void PointCloud::RenderCylinder(const glm::mat4& viewProj) const
{
	float x0 = fitResult.x;
	float y0 = fitResult.y;
	float z0 = fitResult.z;
	float r  = fitResult.w;

	std::vector<glm::vec4> vertices;
	for (int i = 0; i <= cCount; i++)
	{
		float phi = 2 * glm::pi<float>() * i / (float)cCount;

		// botton point
		vertices.push_back({
			x0 + r * cosf(phi),
			y0,
			z0 + r * sinf(phi),
			1
		});

		// top point
		vertices.push_back({
			x0 + r * cosf(phi),
			y0 + 5,
			z0 + r * sinf(phi),
			1
		});
	}

	std::vector<unsigned int> indices;
	for (int i = 0; i < 2 * cCount; i += 2)
	{
		indices.push_back(i + 1);
		indices.push_back(i + 2);
		indices.push_back(i);

		indices.push_back(i + 3);
		indices.push_back(i + 2);
		indices.push_back(i + 1);
	}

	glBindBuffer(GL_ARRAY_BUFFER, cylVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylIds);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec4), vertices.data(), GL_DYNAMIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(cylProgram);
	glBindVertexArray(cylVAO);

	glm::mat4 world(1.0f);
	glm::mat4 mvp = viewProj * world;
	GLuint matrix = glGetUniformLocation(cylProgram, "mvp");
	glUniformMatrix4fv(matrix, 1, GL_FALSE, glm::value_ptr(mvp));

	glDrawElements(GL_TRIANGLES, 3 * 2 * (cCount + 1), GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
	glUseProgram(0);
}