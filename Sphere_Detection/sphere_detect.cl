#define CLOUD_SIZE 14976
#define EPSILON 0.03
#define WIDTH 4
#define HEIGHT 4

__kernel void calcSphere(
	__global float4* data,
	__global int*    idx,
	__global float4* result)
{
	int g_id = get_global_id(0);

	__private float4 points[HEIGHT];
	for(int i = 0; i < HEIGHT; i++)
	{
		points[i] = data[idx[g_id * HEIGHT + i]];
	}

	// temp matrix and vector
	float A[WIDTH*HEIGHT] = {0};
    float AT[HEIGHT*WIDTH] = {0};
    float ATA[WIDTH * WIDTH] = {0};
	for(int i = 0; i < HEIGHT; i++)
	{
		A[4 * i]	 = 2 * points[i].x;
		A[4 * i + 1] = 2 * points[i].y;
		A[4 * i + 2] = 2 * points[i].z;
		A[4 * i + 3] = 1;
	}

    for(int i = 0; i < HEIGHT; i++)
    {
        for(int j = 0; j < WIDTH; j++)
        {
            AT[j * HEIGHT + i] = A[i * WIDTH + j];
        }
    }

    float f[HEIGHT];
    for(int i = 0; i < HEIGHT; i++)
    {
        f[i] = points[i].x * points[i].x + points[i].y * points[i].y + points[i].z * points[i].z;
    }

	// matmul for ATA
    for(int i = 0; i < WIDTH; i++)
	{
		for(int k = 0; k < HEIGHT; k++)
		{
			for(int j = 0; j < WIDTH; j++)
			{
				ATA[i * WIDTH + j] += AT[i * HEIGHT + k] * A[k * WIDTH + j];
			}
		}
	}

    // calc for b = ATf
    float b[WIDTH] = {0};
    for(int i = 0; i < WIDTH; i++)
	{
		for(int j = 0; j < HEIGHT; j++)
		{
			b[i] += AT[i * HEIGHT + j] * f[j];
		}
	}

	// Gauss-elimination downwards
	for(int i = 0; i < 3; i++)
	{
		if(ATA[i * 4 + i] == 0)
		{
			continue;
		}

		for(int j = i + 1; j < 4; j++)
		{
			float div = ATA[j * 4 + i] / ATA[i * 4 + i];

			for(int k = i; k < 4; k++)
			{
				ATA[j * 4 + k] = ATA[j * 4 + k] - ATA[i * 4 + k] * div;
			}

			b[j] = b[j] - b[i] * div;
		}
	}

	// elimination upwards
	for(int i = 3; i > 0; i--)
	{
		if(ATA[i * 4 + i] == 0)
		{
			continue;
		}

		for(int j = i - 1; j >= 0; j--)
		{
			float div = ATA[j * 4 + i] / ATA[i * 4 + i];
			ATA[j * 4 + i] = 0;
			b[j] = b[j] - b[i] * div;
		}
	}

	float x = b[0] / ATA[0 * 4 + 0];
    float y = b[1] / ATA[1 * 4 + 1];
    float z = b[2] / ATA[2 * 4 + 2];
    float r = sqrt((b[3] / ATA[3 * 4 + 3]) + x * x + y * y + z * z);

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
	__global int*  result)
{
	int g_id0 = get_global_id(0);
	int g_id1 = get_global_id(1);
	float4 sphere = spheres[g_id0];
	float4 point = data[g_id1];

	float dist = distance(point.xyz, sphere.xyz);
	if(fabs(sphere.w - dist) < EPSILON)
	{
		atomic_inc(&result[g_id0]);
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

	data[g_id].w = fabs(spheres[0].w - dist) < EPSILON;
}