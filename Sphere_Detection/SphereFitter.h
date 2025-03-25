#pragma once

#include "IFitter.h"

class SphereFitter : public IFitter
{
public:
	SphereFitter();

	void Init(cl::Context&, const cl::vector<cl::Device>&) override;
	void Fit(cl::CommandQueue&, cl::BufferGL&) override;
	void EvalCandidate(const glm::vec4&, const int) override;

private:
	const int ITER_NUM = 4096;
	const int CAND_SIZE = 4096;
	std::vector<cl_float4> candidates;

	cl::Program  program;

	cl::Kernel calcKernel;
	cl::Kernel fitKernel;
	cl::Kernel reduceKernel;
	cl::Kernel fillKernel;

	cl::Buffer indexBuffer;
	cl::Buffer sphereBuffer;
	cl::Buffer inlierBuffer;
	cl::Buffer candidateBuffer;
};