#include <fstream>
#include <vector>

#include "pch.h"
#include "CppUnitTest.h"

#include "PointCloud.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SphereDetectionTest
{
	void OCLSetup(cl::vector<cl::Device>& devices, cl::Context& context, cl::CommandQueue& queue)
	{
		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		bool create_context_success = false;
		for (auto platform : platforms)
		{
			cl_context_properties props[] =
			{
				CL_CONTEXT_PLATFORM,	(cl_context_properties)(platform)(),
				CL_GL_CONTEXT_KHR,		(cl_context_properties)wglGetCurrentContext(),
				CL_WGL_HDC_KHR,			(cl_context_properties)wglGetCurrentDC(),
				0
			};

			try
			{
				context = cl::Context(CL_DEVICE_TYPE_GPU, props);
				create_context_success = true;
				break;
			}
			catch (cl::Error error) {}
		}

		if (!create_context_success)
			throw cl::Error(CL_INVALID_CONTEXT, "Failed to create CL/GL shared context");

		devices = context.getInfo<CL_CONTEXT_DEVICES>();
		queue = cl::CommandQueue(context, devices[0]);
	}

	TEST_CLASS(SphereDetectionTest)
	{
	public:
		TEST_METHOD(SphereKernelTest)
		{
			cl::vector<cl::Device> devices;
			cl::Context context;
			cl::CommandQueue queue;

			try
			{
				OCLSetup(devices, context, queue);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << std::endl;
				return;
			}

			std::ifstream sphereFile("../Sphere_Detection/sphere_detect.cl");
			std::string sphereCode(std::istreambuf_iterator<char>(sphereFile), (std::istreambuf_iterator<char>()));

			cl::Program::Sources sphereSource(1, std::make_pair(sphereCode.c_str(), sphereCode.length() + 1));
			cl::Program program(context, sphereSource);

			try
			{
				program.build(devices);
			}
			catch (cl::Error& error)
			{
				std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
				throw error;
			}

			cl::Kernel kernel(program, "calcSphere");

			std::vector<cl_float4> points = {
				// TODO: make up testing points
			};

			std::vector<cl_int4> idx;
			for (int i = 0; i < points.size(); i += 4)
			{
				idx.push_back({i, i+1, i+2, i+3});
			}

			cl::Buffer pointsBuffer(context, CL_MEM_READ_ONLY, points.size() * sizeof(cl_float4), points.data());
			cl::Buffer idxBuffer(context, CL_MEM_READ_ONLY, idx.size() * sizeof(cl_int4), idx.data());
			cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, idx.size() * sizeof(cl_float4));

			queue.enqueueWriteBuffer(pointsBuffer, CL_TRUE, 0, points.size() * sizeof(cl_float4), points.data());
			queue.enqueueWriteBuffer(idxBuffer, CL_TRUE, 0, idx.size() * sizeof(cl_int4), idx.data());
			queue.finish();

			kernel.setArg(0, pointsBuffer);
			kernel.setArg(1, idxBuffer);
			kernel.setArg(2, resultBuffer);

			queue.enqueueNDRangeKernel(kernel, cl::NullRange, idx.size(), cl::NullRange);

			std::vector<cl_float4> results;
			results.resize(idx.size());

			queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, idx.size() * sizeof(cl_float4), results.data());
		}
	};
}
