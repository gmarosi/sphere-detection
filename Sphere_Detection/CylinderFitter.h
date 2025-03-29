#pragma once

#include "IFitter.h"

class CylinderFitter : public IFitter
{
public:
	CylinderFitter();

	void Init(cl::Context&, const cl::vector<cl::Device>&) override;
	void Fit(cl::CommandQueue&, cl::BufferGL&) override;
	void EvalCandidate(const glm::vec4&, const int) override;

private:
	const int ITER_NUM = 2048;
	const int CYLINDER_ITER_NUM = 4096 * 8;

	std::vector<int> candidates;

	cl::Program program;
	cl::Context* context;

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
};