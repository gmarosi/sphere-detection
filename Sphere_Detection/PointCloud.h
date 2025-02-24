#pragma once

#include "SphereFitter.h"
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
	const int ITER_NUM = 4096;
	const int CHANNELS = 4;
	const int CYLINDER_ITER_NUM = 1024;

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

	SphereFitter sphereFitter;

	// Cylinder detection
	cl::Program clCylinderProgram;

	cl::Kernel planeCalcKernel;
	cl::Kernel planeFitKernel;
	cl::Kernel planeReduceKernel;
	cl::Kernel planeFillKernel;

	cl::Buffer planeIdxBuffer;
	cl::Buffer planePointsBuffer;
	cl::Buffer planeNormalsBuffer;
	cl::Buffer planeInliersBuffer;

	cl::Kernel cylinderCalcKernel;
	cl::Kernel cylinderFitKernel;
	cl::Kernel cylinderReduceKernel;
	cl::Kernel cylinderColorKernel;

	cl::Buffer cylinderPointsBuffer;
	cl::Buffer cylinderRandBuffer;
	cl::Buffer cylinderDataBuffer;
	cl::Buffer cylinderInliersBuffer;

	void FitCylinder(cl::CommandQueue& queue);
};