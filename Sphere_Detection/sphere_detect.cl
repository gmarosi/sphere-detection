const int CLOUD_SIZE = 14976;
const float EPSILON = 0.05;

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


	// QR decomposition using Householder reflections
	float QT[4*4] = {0};

    // 1st iteration
    float A1[4*4] = {0}, Q1[4*4];

    {
        float4 x = (float4)(A[0], A[4], A[8], A[12]);
	    float alpha = length(x);
	    alpha = copysign(alpha, x.x);
        float4 v_ = normalize(x + (float4)(alpha, 0, 0, 0));
        float v[4] = {v_.x, v_.y, v_.z, v_.w};

	    for(int i = 0; i < 4; i++)
	    {
	    	for(int j = 0; j < 4; j++)
	    	{
	    		float vvt = 2 * v[i] * v[j];
	    		Q1[i * 4 + j] = i == j ? 1 - vvt : -vvt;
	    	}
	    }

	    for(int i = 0; i < 4; i++)
	    {
	    	for(int k = 0; k < 4; k++)
	    	{
	    		for(int j = 0; j < 4; j++)
	    		{
	    			A1[i * 4 + j] += Q1[i * 4 + k] * A[k * 4 + j];
	    		}
	    	}
	    }
    }


    // 2nd iteration
    float A2[4*4] = {0}, Q2[4*4] = {0}, Qtemp[4*4] = {0};

    {
        float3 x = (float3)(A1[4 + 1], A1[8 + 1], A1[12 + 1]);
        float alpha = length(x);
	    alpha = copysign(alpha, x.x);
        float3 v_ = normalize(x - (float3)(alpha, 0, 0));
        float v[3] = {v_.x, v_.y, v_.z};

        for(int i = 1; i < 4; i++)
	    {
	    	for(int j = 1; j < 4; j++)
	    	{
	    		float vvt = 2 * v[i - 1] * v[j - 1];
	    		Q2[i * 4 + j] = i == j ? 1 - vvt : -vvt;
	    	}
	    }
        Q2[0] = 1;

	    for(int i = 0; i < 4; i++)
	    {
	    	for(int k = 0; k < 4; k++)
	    	{
	    		for(int j = 0; j < 4; j++)
	    		{
	    			A2[i * 4 + j] += Q2[i * 4 + k] * A1[k * 4 + j];
                    Qtemp[i * 4 + j] += Q2[i * 4 + k] * Q1[k * 4 + j];
	    		}
	    	}
	    }
    }

    // 3rd iteration
    float R[4*4] = {0}, Q3[4*4] = {0};

    {
        float2 x = (float2)(A2[8 + 2], A2[12 + 2]);
        float alpha = length(x);
	    alpha = copysign(alpha, x.x);
        float2 v_ = normalize(x - (float2)(alpha, 0));
        float v[2] = {v_.x, v_.y};

        for(int i = 2; i < 4; i++)
	    {
	    	for(int j = 2; j < 4; j++)
	    	{
	    		float vvt = 2 * v[i - 2] * v[j - 2];
	    		Q2[i * 4 + j] = i == j ? 1 - vvt : -vvt;
	    	}
	    }
        Q3[0] = 1;
        Q3[5] = 1;

	    for(int i = 0; i < 4; i++)
	    {
	    	for(int k = 0; k < 4; k++)
	    	{
	    		for(int j = 0; j < 4; j++)
	    		{
	    			R[i * 4 + j]  += Q3[i * 4 + k] * A2[k * 4 + j];
                    QT[i * 4 + j] += Q3[i * 4 + k] * Qtemp[k * 4 + j];
	    		}
	    	}
	    }
    }

	// Ax = f -> Rx = QTf
	// calculating QTf
	float final[4] = {0};

	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			final[i] += QT[i * 4 + j] * f[j];
		}
	}

	// solving Rx = QTf by substitution
	final[3] = R[15] == 0 ? 0 : final[3] / R[15];

	final[2] = final[2] - R[11] * final[3];
	final[2] = R[10] == 0 ? final[2] : final[2] / R[10];

	final[1] = final[1] - R[6] * final[2] - R[7] * final[3];
	final[1] = R[5] == 0 ? final[1] : final[1] / R[5];

	final[0] = final[0] - R[1] * final[1] - R[2] * final[2] - R[3] * final[3];
	final[0] = R[0] == 0 ? final[0] : final[0] / R[0];

	float x = final[0];
    float y = final[2];
    float z = final[1];
    float r = sqrt(final[3] + x * x + y * y + z * z);

	result[g_id] = (float4)(x, y, z, r);
}

__kernel void fitSphere(
	__global float4* data,
	__global float4* spheres,
	__global float*  result)
{
	int g_id = get_global_id(0);
	float4 sphere = spheres[g_id];

	float sum = 0;
	// naive implementation
	for(int i = 0; i < CLOUD_SIZE; i++)
	{
		float dist = distance(data[i].xyz, sphere.xyz);
		sum += fabs(sphere.w - dist) < EPSILON ? 1 : 0;
	}
	result[g_id] = sum / CLOUD_SIZE;
}

__kernel void reduce(
	__global float*  inliers,
	__global float4* spheres,
	int offset)
{
	int g_id = get_global_id(0) * offset * 2;
	int id_2 = g_id + offset;

	// idea is to also swap sphere coordinates along with inlier ratios
	if(inliers[g_id] < inliers[id_2])
	{
		inliers[g_id] = inliers[id_2];
		spheres[g_id] = spheres[id_2];
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