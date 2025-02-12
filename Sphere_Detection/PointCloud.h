#pragma once

// GLEW
#include <GL/glew.h>

// SDL, OpenGL
#include <SDL.h>
#include <SDL_opengl.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtc/type_ptr.hpp>

// CL
#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <CL/cl_gl.h>
#include <oclutils.hpp>

#include "ProgramObject.h"
#include "BufferObject.h"
#include "VertexArrayObject.h"

#include "SHMManager.h"

enum FitMode {SPHERE, CYLINDER};

class PointCloud
{
public:
	PointCloud();

	bool Init(const MemoryNames& memNames);
	bool InitCl(cl::Context& context, const cl::vector<cl::Device>& devices);

	void Update();
	void Render(const glm::mat4& viewProj);

	void Fit(cl::CommandQueue& queue);

	void ChangeMode();

private:
	const int POINT_CLOUD_SIZE = 14976;
	const int CHANNELS = 4;
	const int ITER_NUM = 4096;

	const float pointRenderSize = 5.f;

	SHMManager *mapMem;
	bool		fit = false;

	FitMode mode;

	std::vector<glm::vec4>	pointsPos;
	std::vector<float>		pointsIntensity;
	std::vector<int>		candidates;

	// GL
	ProgramObject program;

	GLuint cloudVAO;
	GLuint posVBO;
	GLuint intensityVBO;

	// CL
	cl::Context* clContext;
	cl::BufferGL posBuffer;

	// Sphere detection
	cl::Program clSphereProgram;

	cl::Kernel sphereCalcKernel;
	cl::Kernel sphereFitKernel;
	cl::Kernel reduceKernel;
	cl::Kernel sphereFillKernel;

	cl::Buffer indexBuffer;
	cl::Buffer sphereBuffer;
	cl::Buffer inlierBuffer;

	// Cylinder detection
	cl::Program clCylinderProgram;

	cl::Kernel planeCalcKernel;
	cl::Kernel planeFitKernel;
	cl::Kernel planeReduceKernel;
	cl::Kernel planeColorKernel;

	cl::Buffer planeIdxBuffer;
	cl::Buffer planePointsBuffer;
	cl::Buffer planeNormalsBuffer;
	cl::Buffer planeInliersBuffer;

	void FitSphere(cl::CommandQueue& queue);
	void FitCylinder(cl::CommandQueue& queue);
};