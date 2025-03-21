#include <fstream>
#include <vector>
#include <algorithm>

#include "pch.h"
#include "CppUnitTest.h"

#include "PointCloud.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SphereDetectionTest
{
	const char* getErrorString(cl_int error)
	{
		switch (error) {
			// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

			// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

			// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
		}
	}

	inline unsigned round_up_div(unsigned a, unsigned b) {
		return static_cast<int>(ceil((double)a / b));
	}

	TEST_CLASS(SphereDetectionTest)
	{
	private:
		cl::vector<cl::Device> devices;
		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;

	public:
		TEST_METHOD_INITIALIZE(SphereKernelTestInit)
		{
			cl::vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);

			bool create_context_success = false;
			for (auto platform : platforms)
			{
				cl_context_properties props[] =
				{
					CL_CONTEXT_PLATFORM,	(cl_context_properties)(platform)(),
					0
				};

				try
				{
					context = cl::Context(CL_DEVICE_TYPE_GPU, props);
					create_context_success = true;
					break;
				}
				catch (cl::Error& error)
				{
					std::cout << error.what() << "\n"
						<< getErrorString(error.err()) << std::endl;
				}
			}

			if (!create_context_success)
				throw cl::Error(CL_INVALID_CONTEXT, "Failed to create CL/GL shared context");

			devices = context.getInfo<CL_CONTEXT_DEVICES>();
			queue = cl::CommandQueue(context, devices[0]);

			std::ifstream sphereFile("../../../Sphere_Detection/sphere_detect.cl");
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
				std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
				throw error;
			}
		}

		TEST_METHOD(SphereCalcTest)
		{
			std::vector<cl_float4> points = {
				{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {-1,0,0,0},
				{1,2,3,0}, {2,3,1,0}, {3,1,2,0}, {3,3,3,0},
				{0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0},
				{10.5,17.23,3.67,0.0}, {9.45,23.76,11.11,0}, {0.54,0.29,0.52,0}, {100,200,0,0}
			};

			std::vector<cl_int4> idx;
			for (int i = 0; i < points.size(); i += 4)
			{
				idx.push_back({ i, i + 1, i + 2, i + 3 });
			}

			try
			{
				cl::Kernel kernel(program, "calcSphere");

				cl::Buffer pointsBuffer(context, CL_MEM_READ_ONLY, points.size() * sizeof(cl_float4));
				cl::Buffer idxBuffer(context, CL_MEM_READ_ONLY, idx.size() * sizeof(cl_int4));
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

				Assert::AreEqual(results[0].v4.m128_f32[0], 0.0f, 0.001f);
				Assert::AreEqual(results[0].v4.m128_f32[1], 0.0f, 0.001f);
				Assert::AreEqual(results[0].v4.m128_f32[2], 0.0f, 0.001f);
				Assert::AreEqual(results[0].v4.m128_f32[3], 1.0f, 0.001f);

				Assert::AreEqual(results[1].v4.m128_f32[0], 2.166f, 0.001f);
				Assert::AreEqual(results[1].v4.m128_f32[1], 2.166f, 0.001f);
				Assert::AreEqual(results[1].v4.m128_f32[2], 2.166f, 0.001f);
				Assert::AreEqual(results[1].v4.m128_f32[3], 1.443f, 0.001f);

				Assert::AreEqual(results[2].v4.m128_f32[0], 0.0f, 0.001f);
				Assert::AreEqual(results[2].v4.m128_f32[1], 0.0f, 0.001f);
				Assert::AreEqual(results[2].v4.m128_f32[2], 0.0f, 0.001f);
				Assert::AreEqual(results[2].v4.m128_f32[3], 0.0f, 0.001f);

				Assert::AreEqual(results[3].v4.m128_f32[0], -486.248f, 0.001f);
				Assert::AreEqual(results[3].v4.m128_f32[1], 366.388f, 0.001f);
				Assert::AreEqual(results[3].v4.m128_f32[2], -366.228f, 0.001f);
				Assert::AreEqual(results[3].v4.m128_f32[3], 710.981f, 0.001f);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(ReduceTest)
		{
			const size_t size = 1024;
			std::vector<int> inliers;
			std::vector<cl_float4> spheres;
			for (int i = 0; i < size; i++)
			{
				int x = rand();
				inliers.push_back(x);
				spheres.push_back({ (float)x, 0, 0, 0 });
			}

			try
			{
				cl::Kernel kernel(program, "reduce");

				cl::Buffer inlierBuffer(context, CL_MEM_READ_WRITE, size * sizeof(int));
				cl::Buffer sphereBuffer(context, CL_MEM_READ_WRITE, size * sizeof(cl_float4));

				queue.enqueueWriteBuffer(inlierBuffer, CL_TRUE, 0, size * sizeof(int), inliers.data());
				queue.enqueueWriteBuffer(sphereBuffer, CL_TRUE, 0, size * sizeof(cl_float4), spheres.data());

				const unsigned GROUP_SIZE = 64;
				kernel.setArg(0, inlierBuffer);
				kernel.setArg(1, sphereBuffer);
				kernel.setArg(2, GROUP_SIZE * sizeof(float), nullptr);
				kernel.setArg(3, GROUP_SIZE * sizeof(int), nullptr);

				for (unsigned rem_size = size; rem_size > 1; rem_size = round_up_div(rem_size, GROUP_SIZE))
				{
					int t1 = round_up_div(rem_size, GROUP_SIZE) * GROUP_SIZE;
					queue.enqueueNDRangeKernel(kernel, cl::NullRange, t1, GROUP_SIZE);
				}

				std::vector<int> inl_res;
				inl_res.resize(size);
				queue.enqueueReadBuffer(inlierBuffer, CL_TRUE, 0, size * sizeof(int), inl_res.data());

				std::vector<cl_float4> sph_res;
				sph_res.resize(size);
				queue.enqueueReadBuffer(inlierBuffer, CL_TRUE, 0, size * sizeof(cl_float4), sph_res.data());

				for (int i = 0; i < size; i++)
				{
					Assert::IsTrue(inliers[0] >= inliers[i]);
					Assert::AreEqual((float)(inl_res[i]), sph_res[i].v4.m128_f32[0], 0.1f);
				}
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}
	};
}
