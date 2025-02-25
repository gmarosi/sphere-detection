#pragma once

#include "SphereFitter.h"
#include "CylinderFitter.h"
#include "SHMManager.h"

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
	const int ITER_NUM = 4096;
	const int CHANNELS = 4;
	const int CYLINDER_ITER_NUM = 1024;

	const float pointRenderSize = 5.f;

	SHMManager *mapMem;
	bool		fit = false;

	std::vector<glm::vec4>	pointsPos;
	std::vector<float>		pointsIntensity;
	std::vector<int>		candidates;

	// GL
	ProgramObject program;

	GLuint cloudVAO;
	GLuint posVBO;
	GLuint intensityVBO;

	// CL
	cl::BufferGL posBuffer;

	std::vector<IFitter*>::iterator currentFitter;
	std::vector<IFitter*> fitters;
};