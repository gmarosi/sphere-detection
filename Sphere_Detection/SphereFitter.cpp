#include "SphereFitter.h"

inline unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}

SphereFitter::SphereFitter() = default;

void SphereFitter::Init(cl::Context& context, const cl::vector<cl::Device>& devices)
{
	std::ifstream sphereFile("sphere_detect.cl");
	if (!sphereFile.is_open())
	{
		std::cerr << "SphereFitter::Init(): sphere_detect.cl could not be opened" << std::endl;
		exit(1);
	}
	std::string sphereCode(std::istreambuf_iterator<char>(sphereFile), (std::istreambuf_iterator<char>()));
	sphereFile.close();

	cl::Program::Sources sphereSource(1, std::make_pair(sphereCode.c_str(), sphereCode.length() + 1));
	program = cl::Program(context, sphereSource);

	try
	{
		program.build(devices);
	}
	catch (cl::Error& error)
	{
		std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		throw error;
	}

	calcKernel	 = cl::Kernel(program, "calcSphere");
	fitKernel	 = cl::Kernel(program, "fitSphere");
	reduceKernel = cl::Kernel(program, "reduce");
	fillKernel   = cl::Kernel(program, "fillSphere");

	indexBuffer  = cl::Buffer(context, CL_MEM_WRITE_ONLY, ITER_NUM * FIT_NUM * sizeof(int));
	sphereBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, ITER_NUM * sizeof(cl_float4));
	inlierBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, ITER_NUM * sizeof(int));
	// allocate some memory for unknown number of candidates
	candidateBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, CAND_SIZE * sizeof(cl_float4));
}

void SphereFitter::EvalCandidate(const glm::vec4& point, const int idx)
{
	float dist = glm::distance(glm::vec2(0, 0), glm::vec2(point.x, point.z));
	if (candidates.size() < CAND_SIZE && dist < 3.2 && dist > 1.8 && point.z < 0)
	{
		// constructing cl_float4 from glm::vec4 just to make sure
		candidates.push_back({point.x, point.y, point.z, point.w});
	}
}

glm::vec4 SphereFitter::Fit(cl::CommandQueue& queue, cl::BufferGL& posBuffer)
{
	if (candidates.empty())
	{
		std::cout << "SphereFitter::Fit(): no candidates, skipping sphere fit\n";
		return { 0,0,0,0 };
	}

	std::vector<int> indices;
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
		indices.push_back(a);
		indices.push_back(b);
		indices.push_back(c);
		indices.push_back(d);
	}

	try
	{
		// set inlier buffer to all zeroes
		std::vector<int> zero(ITER_NUM, 0);
		queue.enqueueWriteBuffer(inlierBuffer, CL_TRUE, 0, ITER_NUM * sizeof(int), zero.data());

		// write selected indices to GPU
		queue.enqueueWriteBuffer(indexBuffer, CL_TRUE, 0, ITER_NUM * FIT_NUM * sizeof(int), indices.data());

		// write candidate points to GPU
		queue.enqueueWriteBuffer(candidateBuffer, CL_TRUE, 0, candidates.size() * sizeof(cl_float4), candidates.data());
		queue.finish();

		// calculate spheres
		calcKernel.setArg(0, candidateBuffer);
		calcKernel.setArg(1, indexBuffer);
		calcKernel.setArg(2, sphereBuffer);

		queue.enqueueNDRangeKernel(calcKernel, cl::NullRange, ITER_NUM, cl::NullRange);

		// evaluate sphere inlier ratio
		fitKernel.setArg(0, candidateBuffer);
		fitKernel.setArg(1, sphereBuffer);
		fitKernel.setArg(2, inlierBuffer);

		queue.enqueueNDRangeKernel(fitKernel, cl::NullRange, cl::NDRange(ITER_NUM, candidates.size()), cl::NullRange);

		// reduction to get sphere with highest inlier ratio
		const unsigned GROUP_SIZE = 64;
		reduceKernel.setArg(0, inlierBuffer);
		reduceKernel.setArg(1, sphereBuffer);
		reduceKernel.setArg(2, GROUP_SIZE * sizeof(int), nullptr);
		reduceKernel.setArg(3, GROUP_SIZE * sizeof(int), nullptr);

		for (unsigned rem_size = ITER_NUM; rem_size > 1; rem_size = round_up_div(rem_size, GROUP_SIZE))
		{
			int t1 = round_up_div(rem_size, GROUP_SIZE) * GROUP_SIZE;
			queue.enqueueNDRangeKernel(reduceKernel, cl::NullRange, t1, GROUP_SIZE);
		}

		// acquire GL position buffer
		cl::vector<cl::Memory> acq;
		acq.push_back(posBuffer);
		queue.enqueueAcquireGLObjects(&acq);

		// color points which are on the best sphere
		fillKernel.setArg(0, posBuffer);
		fillKernel.setArg(1, sphereBuffer);

		queue.enqueueNDRangeKernel(fillKernel, cl::NullRange, POINT_CLOUD_SIZE, cl::NullRange);

		queue.enqueueReleaseGLObjects(&acq);
		candidates.clear();
		
		cl_float4 result;
		queue.enqueueReadBuffer(sphereBuffer, CL_TRUE, 0, sizeof(cl_float4), &result);
		return { result.s[0], result.s[1], result.s[2], result.s[3] };
	}
	catch (cl::Error& error)
	{
		std::cerr << "SphereFitter::Fit(): " << error.what() << std::endl;
		throw error;
	}
}