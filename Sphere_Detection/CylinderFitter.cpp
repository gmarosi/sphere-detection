#include "CylinderFitter.h"

inline unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}

CylinderFitter::CylinderFitter() = default;

void CylinderFitter::Init(cl::Context& context, const cl::vector<cl::Device>& devices)
{
	this->context = &context;

	std::ifstream cylinderFile("cylinder_detect.cl");
	std::string cylinderCode(std::istreambuf_iterator<char>(cylinderFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources cylinderSource(1, std::make_pair(cylinderCode.c_str(), cylinderCode.length() + 1));

	program = cl::Program(context, cylinderSource);

	try
	{
		program.build(devices);
	}
	catch (cl::Error& error)
	{
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		throw error;
	}

	planeCalcKernel = cl::Kernel(program, "calcPlane");
	planeFitKernel = cl::Kernel(program, "fitPlane");
	planeReduceKernel = cl::Kernel(program, "reducePlane");
	planeFillKernel = cl::Kernel(program, "fillPlane");

	planeIdxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, ITER_NUM * sizeof(cl_int3));
	planePointsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, ITER_NUM * sizeof(cl_float3));
	planeNormalsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, ITER_NUM * sizeof(cl_float3));
	planeInliersBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, ITER_NUM * sizeof(int));

	cylinderCalcKernel = cl::Kernel(program, "calcCylinder");
	cylinderFitKernel = cl::Kernel(program, "fitCylinder");
	cylinderReduceKernel = cl::Kernel(program, "reduceCylinder");
	cylinderColorKernel = cl::Kernel(program, "colorCylinder");

	cylinderRandBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, CYLINDER_ITER_NUM * sizeof(cl_int3));
	cylinderDataBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, CYLINDER_ITER_NUM * sizeof(cl_float4));
	cylinderInliersBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, CYLINDER_ITER_NUM * sizeof(int));
}

void CylinderFitter::EvalCandidate(const glm::vec4& point, const int idx)
{
	if (point.y < -1)
	{
		candidates.push_back(idx);
	}
}

void CylinderFitter::Fit(cl::CommandQueue& queue, cl::BufferGL& posBuffer)
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
		std::vector<cl_float3> closePoints;
		pcl.resize(POINT_CLOUD_SIZE);
		queue.enqueueReadBuffer(posBuffer, CL_TRUE, 0, POINT_CLOUD_SIZE * sizeof(glm::vec4), pcl.data());

		// select close points from the plane
		const float close = 7;
		for (int i = 0; i < POINT_CLOUD_SIZE; i++)
		{
			float dist = glm::distance(glm::vec2(0, 0), glm::vec2(pcl[i].x, pcl[i].z));
			if (dist < close && dist > 3 && pcl[i].y < 1)
			{
				if (pcl[i].w != 0)
					planePoints.push_back({ pcl[i].x, pcl[i].y, pcl[i].z });
				else
					closePoints.push_back({ pcl[i].x, pcl[i].y, pcl[i].z });
			}
		}
		cylinderPointsBuffer = cl::Buffer(*context, CL_MEM_READ_WRITE, planePoints.size() * sizeof(cl_float3));
		cl::Buffer closeBuffer(*context, CL_MEM_READ_ONLY, closePoints.size() * sizeof(cl_float3));

		// get random indices for RANSAC (0 .. candidates.size())
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

		// zeroing out cylinder inlier buffer
		std::vector<int> zero_c(CYLINDER_ITER_NUM, 0);
		queue.enqueueWriteBuffer(cylinderInliersBuffer, CL_TRUE, 0, CYLINDER_ITER_NUM * sizeof(int), zero_c.data());

		queue.enqueueWriteBuffer(cylinderRandBuffer, CL_TRUE, 0, CYLINDER_ITER_NUM * sizeof(cl_int3), indices.data());
		queue.enqueueWriteBuffer(closeBuffer, CL_TRUE, 0, closePoints.size() * sizeof(cl_float3), closePoints.data());
		queue.enqueueWriteBuffer(cylinderPointsBuffer, CL_TRUE, 0, planePoints.size() * sizeof(glm::vec3), planePoints.data());
		queue.finish();

		cylinderCalcKernel.setArg(0, cylinderRandBuffer);
		cylinderCalcKernel.setArg(1, cylinderPointsBuffer);
		cylinderCalcKernel.setArg(2, planeNormalsBuffer);
		cylinderCalcKernel.setArg(3, cylinderDataBuffer);

		queue.enqueueNDRangeKernel(cylinderCalcKernel, cl::NullRange, CYLINDER_ITER_NUM, cl::NullRange);

		size_t size = closePoints.size();
		cylinderFitKernel.setArg(0, closeBuffer);
		cylinderFitKernel.setArg(1, planeNormalsBuffer);
		cylinderFitKernel.setArg(2, cylinderDataBuffer);
		cylinderFitKernel.setArg(3, cylinderInliersBuffer);

		queue.enqueueNDRangeKernel(cylinderFitKernel, cl::NullRange, cl::NDRange(CYLINDER_ITER_NUM, size), cl::NullRange);

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
		cylinderColorKernel.setArg(1, planeNormalsBuffer);
		cylinderColorKernel.setArg(2, cylinderDataBuffer);

		queue.enqueueNDRangeKernel(cylinderColorKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		queue.enqueueReleaseGLObjects(&acq);
		candidates.clear();
	}
	catch (cl::Error& error)
	{
		std::cout << "CylinderFitter::Fit(): " << error.what() << std::endl;
		throw error;
	}
}