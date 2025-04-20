#pragma once

#include "SphereFitter.h"
#include "CylinderFitter.h"
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
	void RenderSphere(const glm::mat4& viewProj) const;
	void RenderCylinder(const glm::mat4& viewProj) const;

private:
	void SphereInit();
	void CylinderInit();

	const int CHANNELS = 4;
	const float pointRenderSize = 5.f;

	SHMManager *mapMem;

	FitMode fitMode = SPHERE;
	bool fit = false;
	bool foundFit = false;
	glm::vec4 fitResult;

	// GL
	GLuint program;
	GLuint cloudVAO;
	GLuint posVBO;

	// sphere rendering
	const int hCount = 36; // horizontal point count
	const int vCount = 18; // vertical point count
	GLuint sphProgram;
	GLuint sphVAO;
	GLuint sphVBO;
	GLuint sphIds;

	// cylinder rendering
	const int cCount = 36; // cylinder segment count
	GLuint cylProgram;
	GLuint cylVAO;
	GLuint cylVBO;
	GLuint cylIds;

	// CL
	cl::BufferGL posBuffer;

	SphereFitter *sphereFitter;
	CylinderFitter *cylinderFitter;
	IFitter *currentFitter;
};