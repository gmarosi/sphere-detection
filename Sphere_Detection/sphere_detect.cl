__kernel void calcSphere(
	__global float3* data,
	__global int4* idx,
	__global float4* result)
{
	int g_id = get_global_id(0);
	int4 indices = idx[g_id];

	// the 4 random points
	// points[0-3].xyzw is a 4x4 matrix
	__private float3 points[4] = {indices.x, indices.y, indices.z, indices.w};

	// temp matrix and vector
	__private float A[4 * 4];
	__private float f[4]; 

	// filling up the A matrix
	int cnt = 0;
	for(int i = 0; i < 4; i++)
	{
		A[cnt * i]	   = 2 * points[i].x;
		A[cnt * i + 1] = 2 * points[i].y;
		A[cnt * i + 2] = 2 * points[i].z;
		A[cnt * i + 3] = 1;
		cnt += 4;
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
    float r = sqrt(f[3] + x * x + y * y + z * z);

	result[g_id] = (float4)(x, y, z, r);
}