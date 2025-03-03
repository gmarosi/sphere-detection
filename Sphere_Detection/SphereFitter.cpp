#include "SphereFitter.h"

inline unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}

SphereFitter::SphereFitter() = default;

void SphereFitter::Init(cl::Context& context, const cl::vector<cl::Device>& devices)
{
	std::ifstream sphereFile("sphere_detect.cl");
	std::string sphereCode(std::istreambuf_iterator<char>(sphereFile), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sphereSource(1, std::make_pair(sphereCode.c_str(), sphereCode.length() + 1));
	program = cl::Program(context, sphereSource);

	try
	{
		program.build(devices);
	}
	catch (cl::Error& error)
	{
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		throw error;
	}

	calcKernel	 = cl::Kernel(program, "calcSphere");
	fitKernel	 = cl::Kernel(program, "fitSphere");
	reduceKernel = cl::Kernel(program, "reduce");
	fillKernel   = cl::Kernel(program, "sphereFill");

	indexBuffer  = cl::Buffer(context, CL_MEM_WRITE_ONLY, ITER_NUM * sizeof(cl_int4));
	sphereBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, ITER_NUM * sizeof(cl_float4));
	inlierBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, ITER_NUM * sizeof(int));
}

void SphereFitter::EvalCandidate(const glm::vec4& point, const int idx)
{
	if (glm::distance(glm::vec2(0, 0), glm::vec2(point.x, point.z)) < 6 && point.z < 0)
	{
		candidates.push_back(idx);
	}
}

void SphereFitter::Fit(cl::CommandQueue& queue, cl::BufferGL& posBuffer)
{
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
	}

	try
	{
		// set inlier buffer to all zeroes
		std::vector<int> zero(ITER_NUM, 0);
		queue.enqueueWriteBuffer(inlierBuffer, CL_TRUE, 0, ITER_NUM * sizeof(int), zero.data());
		queue.enqueueWriteBuffer(indexBuffer, CL_TRUE, 0, ITER_NUM * sizeof(cl_int4), indices.data());
		queue.finish();

		// acquire GL position buffer
		cl::vector<cl::Memory> acq;
		acq.push_back(posBuffer);
		queue.enqueueAcquireGLObjects(&acq);

		// calculate spheres
		calcKernel.setArg(0, posBuffer);
		calcKernel.setArg(1, indexBuffer);
		calcKernel.setArg(2, sphereBuffer);

		queue.enqueueNDRangeKernel(calcKernel, cl::NullRange, ITER_NUM, cl::NullRange);

		// evaluate sphere inlier ratio
		fitKernel.setArg(0, posBuffer);
		fitKernel.setArg(1, sphereBuffer);
		fitKernel.setArg(2, inlierBuffer);
		fitKernel.setArg(3, ITER_NUM);

		queue.enqueueNDRangeKernel(fitKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

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
		fillKernel.setArg(0, posBuffer);
		fillKernel.setArg(1, sphereBuffer);

		queue.enqueueNDRangeKernel(fillKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		queue.enqueueReleaseGLObjects(&acq);
		candidates.clear();
	}
	catch (cl::Error& error)
	{
		std::cout << "SphereFitter::Fit(): " << error.what() << std::endl;
		throw error;
	}
}