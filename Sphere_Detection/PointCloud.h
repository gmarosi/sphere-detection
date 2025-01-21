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

class PointCloud
{
public:
	PointCloud();

	bool Init(const MemoryNames& memNames);
	bool InitCl(cl::Context& context, const cl::vector<cl::Device>& devices);

	void Update();
	void Render(const glm::mat4& viewProj);

	void FitSphere(cl::CommandQueue& queue);

private:
	const int POINT_CLOUD_SIZE = 14976;
	const int CHANNELS = 4;

	const float pointRenderSize = 5.f;

	SHMManager *mapMem;
	bool		fitSphere = false;

	std::vector<glm::vec3>	pointsPos;
	std::vector<float>		pointsIntensity;

	// GL
	ProgramObject program;

	GLuint cloudVAO;
	GLuint posVBO;
	GLuint intensityVBO;

	// CL
	cl::Context* clContext;

	cl::Program clProgram;
	cl::Kernel	sphereCalcKernel;
	cl::Kernel	sphereFitKernel;

	cl::BufferGL	posBuffer;
};