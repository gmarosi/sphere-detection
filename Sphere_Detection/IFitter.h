#pragma once

#define POINT_CLOUD_SIZE 14976

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

#include <vector>

#include "ProgramObject.h"

class IFitter
{
public:
	virtual ~IFitter() {}
	virtual void Init(cl::Context&, const cl::vector<cl::Device>&) = 0;
	virtual void Fit(cl::CommandQueue&, cl::BufferGL&) = 0;
	virtual void EvalCandidate(const glm::vec4&, const int) = 0;
};