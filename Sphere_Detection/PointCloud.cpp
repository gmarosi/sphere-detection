#include "PointCloud.h"
#include <fstream>

PointCloud::PointCloud()
{
	mapMem = nullptr;
}

bool PointCloud::Init(const MemoryNames& memNames)
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

	return true;
}

bool PointCloud::InitCl(cl::Context& context, const cl::vector<cl::Device>& devices)
{
	try
	{
		// Read kernel source code
		std::ifstream sourceFile("sphere_detect.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		clContext = &context;
		clProgram = cl::Program(*clContext, source);

		try {
			clProgram.build(devices);
		}
		catch (cl::Error error) {
			std::cout << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
			throw error;
		}

		sphereCalcKernel = cl::Kernel(clProgram, "calcSphere");
		// sphereFitKernel = cl::Kernel(clProgram, "fitSphere");

		posBuffer    = cl::BufferGL(*clContext, CL_MEM_READ_WRITE, posVBO);
		indexBuffer  = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, POINT_CLOUD_SIZE * sizeof(cl_int4));
		sphereBuffer = cl::Buffer(*clContext, CL_MEM_READ_ONLY, ITER_NUM * sizeof(cl_float4));
	}
	catch (cl::Error error)
	{
		std::cout << "PointCloud::InitCl : " << error.what() << std::endl;
		return false;
	}

	return true;
}

void PointCloud::Update()
{
	if (mapMem->hasBufferChanged())
	{
		fitSphere = true;
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

void PointCloud::FitSphere(cl::CommandQueue& queue)
{
	if (!fitSphere)
		return;

	/*
	Idea: calculate eg. 1000 spheres w/ their centers and radii
		- 4 random indexes 0..POINT_CLOUD_SIZE
		- calc spheres in kernel
	*/
	const int ITER_NUM = 1000;

	std::vector<cl_int4> indices;
	for (int i = 0; i < ITER_NUM; i++)
	{
		// get 4 distinct numbers
		int a, b, c, d;
		a = rand() % POINT_CLOUD_SIZE;
		do {
			b = rand() % POINT_CLOUD_SIZE;
		} while (b == a);
		do {
			c = rand() % POINT_CLOUD_SIZE;
		} while (c == a || c == b);
		do {
			d = rand() % POINT_CLOUD_SIZE;
		} while (d == a || d == b || d == c);
		indices.push_back({ a, b, c, d });
	}

	try
	{
		queue.enqueueWriteBuffer(indexBuffer, CL_TRUE, 0, ITER_NUM * sizeof(cl_int4), indices.data());
		queue.finish();

		// acquire GL position buffer
		cl::vector<cl::Memory> acq;
		acq.push_back(posBuffer);
		queue.enqueueAcquireGLObjects(&acq);

		// set kernel args
		sphereCalcKernel.setArg(0, posBuffer);
		sphereCalcKernel.setArg(1, indexBuffer);
		sphereCalcKernel.setArg(2, sphereBuffer);

		queue.enqueueNDRangeKernel(sphereCalcKernel, cl::NullRange, ITER_NUM, cl::NullRange);

		queue.enqueueReleaseGLObjects(&acq);
	}
	catch (cl::Error error)
	{
		std::cout << "PointCloud::FitSphere : " << error.what() << std::endl;
		exit(1);
	}
	
}