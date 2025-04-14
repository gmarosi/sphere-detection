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
	const int CHANNELS = 4;

	const float pointRenderSize = 5.f;

	SHMManager *mapMem;
	bool		fit = false;

	std::vector<glm::vec4>	pointsPos;

	// GL
	GLuint program;

	GLuint cloudVAO;
	GLuint posVBO;

	// CL
	cl::BufferGL posBuffer;

	std::vector<IFitter*>::iterator currentFitter;
	std::vector<IFitter*> fitters;
};