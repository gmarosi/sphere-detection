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
	if (++currentFitter == fitters.end())
	{
		currentFitter = fitters.begin();
	}
}

bool PointCloud::Init(const MemoryNames& memNames)
{
	mapMem = new SHMManager(memNames.first, memNames.second, sizeof(int), POINT_CLOUD_SIZE * CHANNELS * sizeof(int));

	pointsPos.resize(POINT_CLOUD_SIZE);

	// Setup VAO & VBOs
	glGenVertexArrays(1, &cloudVAO);
	glBindVertexArray(cloudVAO);

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

	program = glCreateProgram();

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

	return true;
}

bool PointCloud::InitCl(cl::Context& context, const cl::vector<cl::Device>& devices)
{
	try
	{
		posBuffer = cl::BufferGL(context, CL_MEM_READ_WRITE, posVBO);

		auto sphereFitter = new SphereFitter();
		sphereFitter->Init(context, devices);
		fitters.push_back(sphereFitter);

		auto cylinderFitter = new CylinderFitter();
		cylinderFitter->Init(context, devices);
		fitters.push_back(cylinderFitter);

		currentFitter = fitters.begin();
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
		rawData.resize(POINT_CLOUD_SIZE * CHANNELS);
		mapMem->readData(rawData.data());

		for (size_t i = 0; i < POINT_CLOUD_SIZE * CHANNELS; i += CHANNELS)
		{
			// last coordinate will be used by OpenCL kernel
			glm::vec4 point = glm::vec4(rawData[i], rawData[i + 2], -rawData[i + 1], 0);
			pointsPos[i / CHANNELS] = point;

			// storing indices of candidate points
			(*currentFitter)->EvalCandidate(point, i / CHANNELS);
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
		(*currentFitter)->Fit(queue, posBuffer);
	}
	catch (cl::Error&)
	{
		exit(1);
	}
}