#include "PointCloud.h"
#include <fstream>
#include <chrono>

inline unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}

PointCloud::PointCloud()
{
	mapMem = nullptr;
	mode = SPHERE;
}

void PointCloud::ChangeMode()
{
	switch (mode)
	{
	case SPHERE: 
		mode = CYLINDER;
		break;
	case CYLINDER:
		mode = SPHERE;
		break;
	}
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
		clContext = &context;
		posBuffer = cl::BufferGL(*clContext, CL_MEM_READ_WRITE, posVBO);

		sphereFitter.Init(context, devices);


		// Cylinder detection init
		std::ifstream cylinderFile("cylinder_detect.cl");
		std::string cylinderCode(std::istreambuf_iterator<char>(cylinderFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources cylinderSource(1, std::make_pair(cylinderCode.c_str(), cylinderCode.length() + 1));

		clCylinderProgram = cl::Program(*clContext, cylinderSource);

		try
		{
			clCylinderProgram.build(devices);
		}
		catch (cl::Error& error)
		{
			std::cout << clCylinderProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
			throw error;
		}

		planeCalcKernel	  = cl::Kernel(clCylinderProgram, "calcPlane");
		planeFitKernel	  = cl::Kernel(clCylinderProgram, "fitPlane");
		planeReduceKernel = cl::Kernel(clCylinderProgram, "reducePlane");
		planeFillKernel	  = cl::Kernel(clCylinderProgram, "fillPlane");

		planeIdxBuffer	   = cl::Buffer(*clContext, CL_MEM_READ_ONLY, ITER_NUM * sizeof(cl_int3));
		planePointsBuffer  = cl::Buffer(*clContext, CL_MEM_READ_WRITE, ITER_NUM * sizeof(cl_float3));
		planeNormalsBuffer = cl::Buffer(*clContext, CL_MEM_READ_WRITE, ITER_NUM * sizeof(cl_float3));
		planeInliersBuffer = cl::Buffer(*clContext, CL_MEM_READ_WRITE, ITER_NUM * sizeof(int));

		cylinderCalcKernel	 = cl::Kernel(clCylinderProgram, "calcCylinder");
		cylinderFitKernel	 = cl::Kernel(clCylinderProgram, "fitCylinder");
		cylinderReduceKernel = cl::Kernel(clCylinderProgram, "reduceCylinder");
		cylinderColorKernel  = cl::Kernel(clCylinderProgram, "colorCylinder");

		cylinderRandBuffer	  = cl::Buffer(*clContext, CL_MEM_READ_ONLY, CYLINDER_ITER_NUM * sizeof(cl_int3));
		cylinderDataBuffer	  = cl::Buffer(*clContext, CL_MEM_READ_WRITE, CYLINDER_ITER_NUM * sizeof(cl_int3));
		cylinderInliersBuffer = cl::Buffer(*clContext, CL_MEM_READ_WRITE, CYLINDER_ITER_NUM * sizeof(int));
	}
	catch (cl::Error& error)
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
		fit = true;
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
			if (mode == SPHERE)
			{
				sphereFitter.EvalCandidate(point, i / CHANNELS);
			}
			else if (mode == CYLINDER && point.y < -1)
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

void PointCloud::Fit(cl::CommandQueue& queue)
{
	if (!fit)
		return;
	fit = false;

	switch (mode)
	{
	case SPHERE:
		try
		{
			sphereFitter.Fit(queue, posBuffer);
		}
		catch (cl::Error& error)
		{
			exit(1);
		}
		break;
	case CYLINDER:
		FitCylinder(queue);
		break;
	}
}

void PointCloud::FitCylinder(cl::CommandQueue& queue)
{
	std::vector<cl_int3> indices;
	for (int i = 0; i < ITER_NUM; i++)
	{
		// choosing 3 random indices from candidate vector
		int a, b, c;
		a = rand() % candidates.size();
		do {
			b = rand() % candidates.size();
		} while (b == a);
		do {
			c = rand() % candidates.size();
		} while (c == a || c == b);
		indices.push_back({ candidates[a], candidates[b], candidates[c] });
	}

	try
	{
		// set inlier buffer to all zeroes
		std::vector<int> zero(ITER_NUM, 0);
		queue.enqueueWriteBuffer(planeInliersBuffer, CL_TRUE, 0, ITER_NUM * sizeof(int), zero.data());
		queue.enqueueWriteBuffer(planeIdxBuffer, CL_TRUE, 0, ITER_NUM * sizeof(cl_int3), indices.data());
		queue.finish();

		// acquire GL position buffer
		cl::vector<cl::Memory> acq;
		acq.push_back(posBuffer);
		queue.enqueueAcquireGLObjects(&acq);

		// calculate planes
		planeCalcKernel.setArg(0, posBuffer);
		planeCalcKernel.setArg(1, planeIdxBuffer);
		planeCalcKernel.setArg(2, planePointsBuffer);
		planeCalcKernel.setArg(3, planeNormalsBuffer);

		queue.enqueueNDRangeKernel(planeCalcKernel, cl::NullRange, ITER_NUM, cl::NullRange);

		// evaluate plane inlier ratio
		planeFitKernel.setArg(0, posBuffer);
		planeFitKernel.setArg(1, planePointsBuffer);
		planeFitKernel.setArg(2, planeNormalsBuffer);
		planeFitKernel.setArg(3, planeInliersBuffer);
		planeFitKernel.setArg(4, ITER_NUM);

		queue.enqueueNDRangeKernel(planeFitKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		// reduction to get plane with highest inlier count
		const unsigned GROUP_SIZE = 64;
		planeReduceKernel.setArg(0, planeInliersBuffer);
		planeReduceKernel.setArg(1, planePointsBuffer);
		planeReduceKernel.setArg(2, planeNormalsBuffer);
		planeReduceKernel.setArg(3, GROUP_SIZE * sizeof(float), nullptr);
		planeReduceKernel.setArg(4, GROUP_SIZE * sizeof(int), nullptr);

		for (unsigned rem_size = ITER_NUM; rem_size > 1; rem_size = round_up_div(rem_size, GROUP_SIZE))
		{
			int t1 = round_up_div(rem_size, GROUP_SIZE) * GROUP_SIZE;
			queue.enqueueNDRangeKernel(planeReduceKernel, cl::NullRange, t1, GROUP_SIZE);
		}

		// mark points that are part of the best plane
		planeFillKernel.setArg(0, posBuffer);
		planeFillKernel.setArg(1, planePointsBuffer);
		planeFillKernel.setArg(2, planeNormalsBuffer);

		queue.enqueueNDRangeKernel(planeFillKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		// create new buffer with points that are part of the plane
		std::vector<glm::vec4> pcl;
		std::vector<cl_float3> planePoints;
		pcl.resize(POINT_CLOUD_SIZE);
		queue.enqueueReadBuffer(posBuffer, CL_TRUE, 0, POINT_CLOUD_SIZE * sizeof(glm::vec4), pcl.data());

		for (int i = 0; i < POINT_CLOUD_SIZE; i++)
		{
			if (pcl[i].w != 0)
			{
				// store both the coords and the original index
				planePoints.push_back({pcl[i].x, pcl[i].y, pcl[i].z});
			}
		}
		cylinderPointsBuffer = cl::Buffer(*clContext, CL_MEM_READ_WRITE, planePoints.size() * sizeof(cl_float3));

		// get random indices for RANSAC (0 .. planePoints.size())
		std::vector<cl_int3> indices;
		for (int i = 0; i < CYLINDER_ITER_NUM; i++)
		{
			int a, b, c;
			a = rand() % planePoints.size();
			do {
				b = rand() % planePoints.size();
			} while (b == a);
			do {
				c = rand() % planePoints.size();
			} while (c == a || c == b);
			indices.push_back({ a, b, c });
		}

		queue.enqueueWriteBuffer(cylinderRandBuffer, CL_FALSE, 0, CYLINDER_ITER_NUM * sizeof(cl_int3), indices.data());
		queue.enqueueWriteBuffer(cylinderPointsBuffer, CL_TRUE, 0, planePoints.size() * sizeof(glm::vec3), planePoints.data());

		cylinderCalcKernel.setArg(0, cylinderRandBuffer);
		cylinderCalcKernel.setArg(1, cylinderPointsBuffer);
		cylinderCalcKernel.setArg(2, planeNormalsBuffer);
		cylinderCalcKernel.setArg(3, cylinderDataBuffer);

		queue.enqueueNDRangeKernel(cylinderCalcKernel, cl::NullRange, CYLINDER_ITER_NUM, cl::NullRange);

		cylinderFitKernel.setArg(0, posBuffer);
		cylinderFitKernel.setArg(1, cylinderDataBuffer);
		cylinderFitKernel.setArg(2, cylinderInliersBuffer);
		cylinderFitKernel.setArg(3, CYLINDER_ITER_NUM);

		queue.enqueueNDRangeKernel(cylinderFitKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		cylinderReduceKernel.setArg(0, cylinderInliersBuffer);
		cylinderReduceKernel.setArg(1, cylinderDataBuffer);
		cylinderReduceKernel.setArg(2, GROUP_SIZE * sizeof(float), nullptr);
		cylinderReduceKernel.setArg(3, GROUP_SIZE * sizeof(int), nullptr);

		for (unsigned rem_size = CYLINDER_ITER_NUM; rem_size > 1; rem_size = round_up_div(rem_size, GROUP_SIZE))
		{
			int t1 = round_up_div(rem_size, GROUP_SIZE) * GROUP_SIZE;
			queue.enqueueNDRangeKernel(cylinderReduceKernel, cl::NullRange, t1, GROUP_SIZE);
		}

		cylinderColorKernel.setArg(0, posBuffer);
		cylinderColorKernel.setArg(1, cylinderDataBuffer);

		queue.enqueueNDRangeKernel(cylinderColorKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		queue.enqueueReleaseGLObjects(&acq);
	}
	catch (cl::Error& error)
	{
		std::cout << "PointCloud::FitCylinder : " << error.what() << std::endl;
		exit(1);
	}
}