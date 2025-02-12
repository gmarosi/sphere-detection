const int CLOUD_SIZE = 14976;
const float EPSILON = 0.1;

__kernel void calcPlane(
	__global float4* data,
	__global int3*	 idx,
	__global float3* points,
	__global float3* normals)
{
	int g_id = get_global_id(0);
	int3 ids = idx[g_id];

	// get the 3 assigned random points
	float3 p1 = data[ids.x].xyz;
	float3 p2 = data[ids.y].xyz;
	float3 p3 = data[ids.z].xyz;

	// calculate normal of plane defined by the 3 points
	float3 v1 = p2 - p1;
	float3 v2 = p3 - p1;
	float3 norm = cross(v1, v2);

	// store plane data
	points[g_id] = p1;
	normals[g_id] = norm;
}

__kernel void fitPlane(
	__global float4* data,
	__global float3* points,
	__global float3* normals,
	__global int*	 inliers,
	int ITER_NUM)
{
	int g_id = get_global_id(0);
	float3 point = data[g_id].xyz;

	for(int i = 0; i < ITER_NUM; i++)
	{
		float3 p = points[i];
		float3 n = normals[i];

		// point is on plane if |<normal, point - plane_point>| < EPSILON
		if(fabs(dot(n, point - p)) < EPSILON)
		{
			atomic_inc(&inliers[i]);
		}
	}
}

__kernel void reducePlane(
	__global int*    inliers,
	__global float3* points,
	__global float3* normals,
	__local  int*    scratch,   // local for inlier values
	__local  int*	 idx)	    // local for plane global idx
{
	int l_id = get_local_id(0);
	int g_id = get_global_id(0);
	scratch[l_id] = inliers[g_id];
	idx[l_id]	  = g_id;

	barrier(CLK_LOCAL_MEM_FENCE);

	// local reduction
	// swap inlier values along with plane idx
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
		points[get_group_id(0)] = points[idx[0]];
		normals[get_group_id(0)] = normals[idx[0]];
	}
}

__kernel void fillPlane(
	__global float4* data,
	__global float3* points,
	__global float3* normals)
{
	int g_id = get_global_id(0);
	float3 point = data[g_id].xyz;

	float3 p = points[0];
	float3 n = normals[0];
		
	// point is on plane if |<normal, point - plane_point>| < EPSILON
	if(fabs(dot(n, point - p)) < EPSILON)
	{
		data[g_id].w = 1;
	}
}