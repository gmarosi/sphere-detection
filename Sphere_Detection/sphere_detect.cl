const int CLOUD_SIZE = 14976;
const float EPSILON = 0.1;

__kernel void calcSphere(
	__global float4* data,
	__global int4* idx,
	__global float4* result)
{
	int g_id = get_global_id(0);
	int4 indices = idx[g_id];

	// the 4 random points
	// points[0-3].xyzw is a 4x4 matrix
	__private float4 points[4] = {data[indices.x], data[indices.y], data[indices.z], data[indices.w]};

	
	// temp matrix and vector
	__private float A[4 * 4];
	__private float f[4]; 

	// filling up the A matrix
	for(int i = 0; i < 4; i++)
	{
		A[4 * i]	 = 2 * points[i].x;
		A[4 * i + 1] = 2 * points[i].y;
		A[4 * i + 2] = 2 * points[i].z;
		A[4 * i + 3] = 1;
	}

	// filling up the f vector
	f[0] = points[0].x * points[0].x + points[0].y * points[0].y + points[0].z * points[0].z;
	f[1] = points[1].x * points[1].x + points[1].y * points[1].y + points[1].z * points[1].z;
	f[2] = points[2].x * points[2].x + points[2].y * points[2].y + points[2].z * points[2].z;
	f[3] = points[3].x * points[3].x + points[3].y * points[3].y + points[3].z * points[3].z;

	// Gauss-elimination downwards
	for(int i = 0; i < 3; i++)
	{
		if(A[i * 4 + i] == 0)
		{
			continue;
		}

		for(int j = i + 1; j < 4; j++)
		{
			float div = A[j * 4 + i] / A[i * 4 + i];

			for(int k = i; k < 4; k++)
			{
				A[j * 4 + k] = A[j * 4 + k] - A[i * 4 + k] * div;
			}

			f[j] = f[j] - f[i] * div;
		}
	}

	// elimination upwards
	for(int i = 3; i > 0; i--)
	{
		if(A[i * 4 + i] == 0)
		{
			continue;
		}

		for(int j = i - 1; j >= 0; j--)
		{
			float div = A[j * 4 + i] / A[i * 4 + i];
			A[j * 4 + i] = 0;
			f[j] = f[j] - f[i] * div;
		}
	}

	float x = f[0] / A[0 * 4 + 0];
    float y = f[1] / A[1 * 4 + 1];
    float z = f[2] / A[2 * 4 + 2];
    float r = sqrt((f[3] / A[3 * 4 + 3]) + x * x + y * y + z * z);

	result[g_id] = (float4)(
		isinf(x) || isnan(x) ? 0 : x,
		isinf(y) || isnan(y) ? 0 : y,
		isinf(z) || isnan(z) ? 0 : z,
		isinf(r) || isnan(r) ? 0 : r
	);
}

__kernel void fitSphere(
	__global float4* data,
	__global float4* spheres,
	__global int*  result,
	int ITER_NUM)
{
	int g_id = get_global_id(0);
	float4 point = data[g_id];

	// reverse naive implementation
	for(int i = 0; i < ITER_NUM; i++)
	{
		float4 sphere = spheres[i];
		float dist = distance(data[i].xyz, sphere.xyz);
		if(fabs(sphere.w - dist) < EPSILON)
		{
			atomic_inc(&result[i]);
		}
	}
}

__kernel void reduce(
	__global int*  inliers,
	__global float4* spheres,
	__local  int*  scratch, // local for inlier values
	__local  int*	 idx)	  // local for sphere global idx
{
	int l_id = get_local_id(0);
	int g_id = get_global_id(0);
	scratch[l_id] = inliers[g_id];
	idx[l_id]	  = g_id;

	barrier(CLK_LOCAL_MEM_FENCE);

	// local reduction
	// swap inlier values along with sphere idx
	for (int offset = get_local_size(0) / 2; offset != 0; offset /= 2)
	{
		if (l_id < offset && scratch[l_id] < scratch[l_id + offset])
		{
			scratch[l_id] = scratch[l_id + offset];
			idx[l_id] = idx[l_id + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (l_id == 0)
	{
		inliers[get_group_id(0)] = scratch[0];
		spheres[get_group_id(0)] = spheres[idx[0]];
	}
}

__kernel void sphereFill(
	__global float4* data,
	__global float4* spheres)
{
	int g_id = get_global_id(0);
	float dist = distance(data[g_id].xyz, spheres[0].xyz);

	if(fabs(spheres[0].w - dist) < EPSILON)
	{
		data[g_id].w = 1;
	}
}