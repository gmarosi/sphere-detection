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

#include "ProgramObject.h"
#include "BufferObject.h"
#include "VertexArrayObject.h"

#include "SHMManager.h"

class PointCloud
{
public:
	PointCloud();
	void Init(const MemoryNames& memNames);

	void Update();
	void Render(const glm::mat4& viewProj);

private:
	const int POINT_CLOUD_SIZE = 14976;
	const int CHANNELS = 4;

	const float pointRenderSize = 5.f;

	SHMManager *mapMem;

	std::vector<glm::vec3>	pointsPos;
	std::vector<float>		pointsIntensity;

	ProgramObject program;

	VertexArrayObject cloudVAO;
	ArrayBuffer posVBO;
	ArrayBuffer intensityVBO;
};