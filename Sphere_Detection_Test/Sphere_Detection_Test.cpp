#include <iostream>
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

	TEST_CLASS(OCLKernelTest)
	{
	private:
		cl::vector<cl::Device> devices;
		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Program program_c;

	public:
		TEST_METHOD_INITIALIZE(OCLKernelTestInit)
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
			{
				std::cout << "Failed to create CL context" << std::endl;
				return;
			}

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

			std::ifstream cylinderFile("../../../Sphere_Detection/cylinder_detect.cl");
			std::string cylinderCode(std::istreambuf_iterator<char>(cylinderFile), (std::istreambuf_iterator<char>()));
			cylinderFile.close();

			cl::Program::Sources cylinderSource(1, std::make_pair(cylinderCode.c_str(), cylinderCode.length() + 1));
			program_c = cl::Program(context, cylinderSource);

			try
			{
				program_c.build(devices);
			}
			catch (cl::Error& error)
			{
				std::cout << program_c.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
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

				Assert::AreEqual(results[0].v4.m128_f32[0], 0.0f, 0.01f);
				Assert::AreEqual(results[0].v4.m128_f32[1], 0.0f, 0.01f);
				Assert::AreEqual(results[0].v4.m128_f32[2], 0.0f, 0.01f);
				Assert::AreEqual(results[0].v4.m128_f32[3], 1.0f, 0.01f);

				Assert::AreEqual(results[1].v4.m128_f32[0], 2.16f, 0.01f);
				Assert::AreEqual(results[1].v4.m128_f32[1], 2.16f, 0.01f);
				Assert::AreEqual(results[1].v4.m128_f32[2], 2.16f, 0.01f);
				Assert::AreEqual(results[1].v4.m128_f32[3], 1.44f, 0.01f);

				Assert::AreEqual(results[2].v4.m128_f32[0], 0.0f, 0.01f);
				Assert::AreEqual(results[2].v4.m128_f32[1], 0.0f, 0.01f);
				Assert::AreEqual(results[2].v4.m128_f32[2], 0.0f, 0.01f);
				Assert::AreEqual(results[2].v4.m128_f32[3], 0.0f, 0.01f);

				Assert::AreEqual(results[3].v4.m128_f32[0], -486.22f, 0.01f);
				Assert::AreEqual(results[3].v4.m128_f32[1], 366.37f, 0.01f);
				Assert::AreEqual(results[3].v4.m128_f32[2], -366.22f, 0.01f);
				Assert::AreEqual(results[3].v4.m128_f32[3], 710.95f, 0.01f);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< "spherecalc\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(SphereInlierTest)
		{
			cl_float4 sphere = { 0, 0, 0, 1 };

			// expected: first 8 fits, last 5 doesn't
			std::vector<cl_float4> points = {
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
				{-1, 0, 0, 0},
				{0, -1, 0, 0},
				{0, 0, -1, 0},
				{0.98, 0, 0, 0},
				{0, 1.02, 0, 0},
				{2, 1, 0, 0},
				{10, 10, 10, 0},
				{-100, 0, 0, 0},
				{1.05, 0, 0, 0},
				{0, 0, 0.7, 0}
			};

			// for inlier init
			int zero = 0;

			size_t size = points.size();

			try
			{
				cl::Kernel kernel(program, "fitSphere");

				cl::Buffer pointsBuffer(context, CL_MEM_READ_ONLY, size * sizeof(cl_float4));
				cl::Buffer sphereBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float4));
				cl::Buffer inlierBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int));

				queue.enqueueWriteBuffer(pointsBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());
				queue.enqueueWriteBuffer(sphereBuffer, CL_TRUE, 0, sizeof(cl_float4), &sphere);
				queue.enqueueWriteBuffer(inlierBuffer, CL_TRUE, 0, sizeof(int), &zero);
				queue.finish();

				kernel.setArg(0, pointsBuffer);
				kernel.setArg(1, sphereBuffer);
				kernel.setArg(2, inlierBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1, size), cl::NullRange);

				int inlier;
				queue.enqueueReadBuffer(inlierBuffer, CL_TRUE, 0, sizeof(int), &inlier);

				Assert::AreEqual(8, inlier);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< "spherecalc\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(ReduceTest)
		{
			const size_t size = 1 << 12;
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
				queue.enqueueReadBuffer(sphereBuffer, CL_TRUE, 0, size * sizeof(cl_float4), sph_res.data());

				for (int i = 0; i < size; i++)
				{
					Assert::IsTrue(inl_res[0] >= inl_res[i]);
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

		TEST_METHOD(SphereFillTest)
		{
			cl_float4 sphere = { 0, 0, 0, 1 };

			// expected: first 8 fits, last 5 doesn't
			std::vector<cl_float4> points = {
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
				{-1, 0, 0, 0},
				{0, -1, 0, 0},
				{0, 0, -1, 0},
				{0.98, 0, 0, 0},
				{0, 1.02, 0, 0},
				{2, 1, 0, 0},
				{10, 10, 10, 0},
				{-100, 0, 0, 0},
				{1.05, 0, 0, 0},
				{0, 0, 0.7, 0}
			};

			size_t size = points.size();

			try
			{
				cl::Kernel kernel(program, "fillSphere");

				cl::Buffer pointsBuffer(context, CL_MEM_READ_WRITE, size * sizeof(cl_float4));
				cl::Buffer sphereBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float4));

				queue.enqueueWriteBuffer(pointsBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());
				queue.enqueueWriteBuffer(sphereBuffer, CL_TRUE, 0, sizeof(cl_float4), &sphere);
				queue.finish();

				kernel.setArg(0, pointsBuffer);
				kernel.setArg(1, sphereBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, size, cl::NullRange);

				queue.enqueueReadBuffer(pointsBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());

				for (int i = 0; i < 8; i++)
				{
					Assert::AreEqual(1.0f, points[i].s[3], 0.001f);
				}
				for (int i = 8; i < size; i++)
				{
					Assert::AreEqual(0.0f, points[i].s[3], 0.001f);
				}
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(PlaneCalcTest)
		{
			std::vector<cl_float4> points = {
				{0,0,0,0}, {1,0,0,0}, {0,0,1,0},
				{0,1,0,0}, {0,1,1,0}, {1,0,0,0},
				{10,3,4.5,0}, {0.32,8,1.3,0}, {3,0.17,11,0}
			};

			std::vector<cl_int3> idx;
			for (int i = 0; i < points.size(); i += 3)
			{
				idx.push_back({ i, i + 1, i + 2 });
			}

			try
			{
				cl::Kernel kernel(program_c, "calcPlane");
				cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, points.size() * sizeof(cl_float4));
				cl::Buffer idxBuffer(context, CL_MEM_READ_ONLY, idx.size() * sizeof(cl_int3));
				cl::Buffer pointsBuffer(context, CL_MEM_WRITE_ONLY, idx.size() * sizeof(cl_float3));
				cl::Buffer normalsBuffer(context, CL_MEM_WRITE_ONLY, idx.size() * sizeof(cl_float3));

				queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, points.size() * sizeof(cl_float4), points.data());
				queue.enqueueWriteBuffer(idxBuffer, CL_TRUE, 0, idx.size() * sizeof(cl_int3), idx.data());

				kernel.setArg(0, inputBuffer);
				kernel.setArg(1, idxBuffer);
				kernel.setArg(2, pointsBuffer);
				kernel.setArg(3, normalsBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, idx.size(), cl::NullRange);

				std::vector<cl_float3> res_pos, res_norm;
				res_pos.resize(idx.size());
				res_norm.resize(idx.size());
				queue.enqueueReadBuffer(pointsBuffer, CL_TRUE, 0, res_pos.size() * sizeof(cl_float3), res_pos.data());
				queue.enqueueReadBuffer(normalsBuffer, CL_TRUE, 0, res_norm.size() * sizeof(cl_float3), res_norm.data());

				for (int i = 0; i < res_pos.size(); i++)
					for (int j = 0; j < 3; j++)
						Assert::AreEqual(res_pos[i].v4.m128_f32[j], points[i * 3].v4.m128_f32[j], 0.001f);

				Assert::AreEqual(res_norm[0].v4.m128_f32[0], 0.0f, 0.01f);
				Assert::AreEqual(res_norm[0].v4.m128_f32[1], -1.0f, 0.01f);
				Assert::AreEqual(res_norm[0].v4.m128_f32[2], 0.0f, 0.01f);

				Assert::AreEqual(res_norm[1].v4.m128_f32[0], 0.71f, 0.01f);
				Assert::AreEqual(res_norm[1].v4.m128_f32[1], 0.71f, 0.01f);
				Assert::AreEqual(res_norm[1].v4.m128_f32[2], 0.0f, 0.01f);

				Assert::AreEqual(res_norm[2].v4.m128_f32[0], 0.22f, 0.01f);
				Assert::AreEqual(res_norm[2].v4.m128_f32[1], 0.79f, 0.01f);
				Assert::AreEqual(res_norm[2].v4.m128_f32[2], 0.58f, 0.01f);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(PlaneInlierTest)
		{
			cl_float3 point = { 0,0,0 };
			cl_float3 norm = { 0,1,0 };

			std::vector<cl_float4> points = {
				{1,0,0,0},
				{0,0,1,0},
				{-1,0,0,0},
				{0,0,-1,0},
				{1,0,1,0},
				{-1,0,-1,0},
				{0,1,0,0},
				{0,-1,0,0},
				{10,1,2,0},
				{5,0.5,0,0}
			};
			size_t size = points.size();

			int inlier = 0;
			try
			{
				cl::Kernel kernel(program_c, "fitPlane");

				cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, size * sizeof(cl_float4));
				cl::Buffer pointBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float3));
				cl::Buffer normBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float3));
				cl::Buffer inlierBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int));

				queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());
				queue.enqueueWriteBuffer(pointBuffer, CL_TRUE, 0, sizeof(cl_float3), &point);
				queue.enqueueWriteBuffer(normBuffer, CL_TRUE, 0, sizeof(cl_float3), &norm);
				queue.enqueueWriteBuffer(inlierBuffer, CL_TRUE, 0, sizeof(int), &inlier);
				queue.finish();

				kernel.setArg(0, dataBuffer);
				kernel.setArg(1, pointBuffer);
				kernel.setArg(2, normBuffer);
				kernel.setArg(3, inlierBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1, size), cl::NullRange);

				queue.enqueueReadBuffer(inlierBuffer, CL_TRUE, 0, sizeof(int), &inlier);

				Assert::AreEqual(6, inlier);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(PlaneFillTest)
		{
			cl_float3 point = { 0,0,0 };
			cl_float3 norm = { 0,1,0 };

			std::vector<cl_float4> points = {
				{1,0,0,0},
				{0,0,1,0},
				{-1,0,0,0},
				{0,0,-1,0},
				{1,0,1,0},
				{-1,0,-1,0},
				{0,1,0,0},
				{0,-1,0,0},
				{10,1,2,0},
				{5,0.5,0,0}
			};
			size_t size = points.size();

			try
			{
				cl::Kernel kernel(program_c, "fillPlane");

				cl::Buffer dataBuffer(context, CL_MEM_READ_WRITE, size * sizeof(cl_float4));
				cl::Buffer pointBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float3));
				cl::Buffer normBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float3));

				queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());
				queue.enqueueWriteBuffer(pointBuffer, CL_TRUE, 0, sizeof(cl_float3), &point);
				queue.enqueueWriteBuffer(normBuffer, CL_TRUE, 0, sizeof(cl_float3), &norm);
				queue.finish();

				kernel.setArg(0, dataBuffer);
				kernel.setArg(1, pointBuffer);
				kernel.setArg(2, normBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, size, cl::NullRange);

				queue.enqueueReadBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());

				for (int i = 0; i < 6; i++)
				{
					Assert::AreEqual(0.25f, points[i].s[3], 0.01f);
				}
				for (int i = 6; i < size; i++)
				{
					Assert::AreEqual(0.0f, points[i].s[3], 0.01f);
				}
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(CylinderCalcTest)
		{
			std::vector<cl_float3> points = {
				{-1,0,0}, {1,0,0}, {0,0,1},
				{10,0,3}, {0.32,0,8}, {3,0,0.17},
				{5.63527261068398, -0.692904204691849, -1.52133834711262},
				{5.4029353024265, -1.70681461715848, -1.51823251857139},
				{10.7040668630443, -2.23217858522227, 0.190859516798132}
			};

			std::vector<cl_int3> idx;
			for (int i = 0; i < points.size(); i += 3)
			{
				idx.push_back({ i, i + 1, i + 2 });
			}

			try
			{
				cl::Kernel kernel(program_c, "calcCylinder");
				cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, points.size() * sizeof(cl_float3));
				cl::Buffer idxBuffer(context, CL_MEM_READ_ONLY, idx.size() * sizeof(cl_int3));
				cl::Buffer pointsBuffer(context, CL_MEM_WRITE_ONLY, idx.size() * sizeof(cl_float3));

				queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, points.size() * sizeof(cl_float3), points.data());
				queue.enqueueWriteBuffer(idxBuffer, CL_TRUE, 0, idx.size() * sizeof(cl_int3), idx.data());

				kernel.setArg(0, idxBuffer);
				kernel.setArg(1, inputBuffer);
				kernel.setArg(2, pointsBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, idx.size(), cl::NullRange);

				std::vector<cl_float3> cylinders;
				cylinders.resize(idx.size());
				queue.enqueueReadBuffer(pointsBuffer, CL_TRUE, 0, cylinders.size() * sizeof(cl_float3), cylinders.data());

				Assert::AreEqual(cylinders[0].v4.m128_f32[0], 0.0f, 0.01f);
				Assert::AreEqual(cylinders[0].v4.m128_f32[1], 0.0f, 0.01f);
				Assert::AreEqual(cylinders[0].v4.m128_f32[2], 1.0f, 0.01f);

				Assert::AreEqual(cylinders[1].v4.m128_f32[0], 5.02f, 0.01f);
				Assert::AreEqual(cylinders[1].v4.m128_f32[1], 5.24f, 0.01f);
				Assert::AreEqual(cylinders[1].v4.m128_f32[2], 5.45f, 0.01f);

				Assert::AreEqual(cylinders[2].v4.m128_f32[0], 5.63f, 0.01f);
				Assert::AreEqual(cylinders[2].v4.m128_f32[1], 6.85f, 0.01f);
				Assert::AreEqual(cylinders[2].v4.m128_f32[2], 8.37f, 0.01f);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(CylinderInlierTest)
		{
			cl_float3 cylinder = { 0,0,1 };

			std::vector<cl_float4> points = {
				{1,0,0,0},
				{0,0,1,0},
				{1,5,0,0},
				{0.71,4,0.71,0},
				{1,1,1,0},
				{2,0,0,0},
				{0,1,0,0},
				{0,0,0,0}
			};
			size_t size = points.size();
			int inlier = 0;

			try
			{
				cl::Kernel kernel(program_c, "fitCylinder");

				cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, size * sizeof(cl_float4));
				cl::Buffer cylinderBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float3));
				cl::Buffer inlierBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int));

				queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());
				queue.enqueueWriteBuffer(cylinderBuffer, CL_TRUE, 0, sizeof(cl_float3), &cylinder);
				queue.enqueueWriteBuffer(inlierBuffer, CL_TRUE, 0, sizeof(int), &inlier);

				kernel.setArg(0, dataBuffer);
				kernel.setArg(1, cylinderBuffer);
				kernel.setArg(2, inlierBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1, size), cl::NullRange);

				queue.enqueueReadBuffer(inlierBuffer, CL_TRUE, 0, sizeof(int), &inlier);

				Assert::AreEqual(4, inlier);
			}
			catch (cl::Error& error)
			{
				std::cout << error.what() << "\n"
					<< getErrorString(error.err()) << std::endl;
				return;
			}
		}

		TEST_METHOD(CylinderFillTest)
		{
			cl_float3 cylinder = { 0,0,1 };

			std::vector<cl_float4> points = {
				{1,0,0,0},
				{0,0,1,0},
				{1,5,0,0},
				{0.71,4,0.71,0},
				{1,1,1,0},
				{2,0,0,0},
				{0,1,0,0},
				{0,0,0,0}
			};
			size_t size = points.size();

			try
			{
				cl::Kernel kernel(program_c, "fillCylinder");

				cl::Buffer dataBuffer(context, CL_MEM_READ_WRITE, size * sizeof(cl_float4));
				cl::Buffer cylinderBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float3));

				queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());
				queue.enqueueWriteBuffer(cylinderBuffer, CL_TRUE, 0, sizeof(cl_float3), &cylinder);

				kernel.setArg(0, dataBuffer);
				kernel.setArg(1, cylinderBuffer);

				queue.enqueueNDRangeKernel(kernel, cl::NullRange, size, cl::NullRange);

				queue.enqueueReadBuffer(dataBuffer, CL_TRUE, 0, size * sizeof(cl_float4), points.data());

				for (int i = 0; i < 4; i++)
				{
					Assert::AreEqual(0.75f, points[i].s[3], 0.01f);
				}
				for (int i = 4; i < size; i++)
				{
					Assert::AreEqual(0.0f, points[i].s[3], 0.01f);
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
