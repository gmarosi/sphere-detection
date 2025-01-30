#include "PointCloud.h"
#include <fstream>
#include <chrono>

unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}

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
		sphereFitKernel  = cl::Kernel(clProgram, "fitSphere");
		reduceKernel	 = cl::Kernel(clProgram, "reduce");
		sphereFillKernel = cl::Kernel(clProgram, "sphereFill");

		posBuffer    = cl::BufferGL(*clContext, CL_MEM_READ_WRITE, posVBO);
		indexBuffer  = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, ITER_NUM * sizeof(cl_int4));
		sphereBuffer = cl::Buffer(*clContext, CL_MEM_READ_ONLY, ITER_NUM * sizeof(cl_float4));
		inlierBuffer = cl::Buffer(*clContext, CL_MEM_READ_WRITE, ITER_NUM * sizeof(float));
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
			// last coordinate will be used by OpenCL kernel
			glm::vec4 point = glm::vec4(rawData[i], rawData[i + 2], -rawData[i + 1], 0);
			pointsPos[i / CHANNELS] = point;
			pointsIntensity[i / CHANNELS] = rawData[i + 3];

			// storing indices of candidate points
			if (glm::distance(glm::vec2(0, 0), glm::vec2(point.x, point.z)) < 6 && point.z < 0)
			{
				candidates.push_back(i / CHANNELS);
			}
		}

		glBindBuffer(GL_ARRAY_BUFFER, posVBO);
		glBufferData(GL_ARRAY_BUFFER, POINT_CLOUD_SIZE * sizeof(glm::vec4), pointsPos.data(), GL_DYNAMIC_DRAW);
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
	fitSphere = false;

	/*
	Idea: calculate eg. 1000 spheres w/ their centers and radii
		- 4 random indexes 0..POINT_CLOUD_SIZE
		- calc spheres in kernel
	*/

	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = std::chrono::high_resolution_clock::now();

	std::vector<cl_int4> indices;
	for (int i = 0; i < ITER_NUM; i++)
	{
		// choosing 4 random indices from candidate vector
		int a, b, c, d;
		a = rand() % candidates.size();
		do {
			b = rand() % candidates.size();
		} while (b == a);
		do {
			c = rand() % candidates.size();
		} while (c == a || c == b);
		do {
			d = rand() % candidates.size();
		} while (d == a || d == b || d == c);
		indices.push_back({ candidates[a], candidates[b], candidates[c], candidates[d] });
		
		/*
		// points close in data are close in space => more likely part of same object
		int a = rand() % (candidates.size() - 3);
		indices.push_back({ candidates[a], candidates[a + 1], candidates[a + 2], candidates[a + 3] });
		*/
	}

	try
	{
		queue.enqueueWriteBuffer(indexBuffer, CL_TRUE, 0, ITER_NUM * sizeof(cl_int4), indices.data());
		queue.finish();

		// acquire GL position buffer
		cl::vector<cl::Memory> acq;
		acq.push_back(posBuffer);
		queue.enqueueAcquireGLObjects(&acq);

		// calculate spheres
		sphereCalcKernel.setArg(0, posBuffer);
		sphereCalcKernel.setArg(1, indexBuffer);
		sphereCalcKernel.setArg(2, sphereBuffer);

		queue.enqueueNDRangeKernel(sphereCalcKernel, cl::NullRange, ITER_NUM, cl::NullRange);

		// Debug print for spheres
		/*
		glm::vec4 sphere;
		queue.enqueueReadBuffer(sphereBuffer, CL_TRUE, 0, sizeof(cl_float4), &sphere);
		std::cout << sphere.x << "; " << sphere.y << "; " << sphere.z << "; r: " << sphere.w << std::endl;
		*/

		// evaluate sphere inlier ratio
		sphereFitKernel.setArg(0, posBuffer);
		sphereFitKernel.setArg(1, sphereBuffer);
		sphereFitKernel.setArg(2, inlierBuffer);

		queue.enqueueNDRangeKernel(sphereFitKernel, cl::NullRange, ITER_NUM, cl::NullRange);
		
		// reduction to get sphere with highest inlier ratio
		const unsigned GROUP_SIZE = 64;
		reduceKernel.setArg(0, inlierBuffer);
		reduceKernel.setArg(1, sphereBuffer);
		reduceKernel.setArg(2, GROUP_SIZE * sizeof(float), nullptr);
		reduceKernel.setArg(3, GROUP_SIZE * sizeof(int), nullptr);

		for (unsigned rem_size = ITER_NUM; rem_size > 1; rem_size = round_up_div(rem_size, GROUP_SIZE))
		{
			int t1 = round_up_div(rem_size, GROUP_SIZE) * GROUP_SIZE;
			queue.enqueueNDRangeKernel(reduceKernel, cl::NullRange, t1, GROUP_SIZE);
		}
		
		// color points which are on the best sphere
		sphereFillKernel.setArg(0, posBuffer);
		sphereFillKernel.setArg(1, sphereBuffer);

		queue.enqueueNDRangeKernel(sphereFillKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		queue.enqueueReleaseGLObjects(&acq);

		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << elapsed.count() * 1000 << " ms" << std::endl;
	}
	catch (cl::Error error)
	{
		std::cout << "PointCloud::FitSphere : " << error.what() << std::endl;
		exit(1);
	}

}