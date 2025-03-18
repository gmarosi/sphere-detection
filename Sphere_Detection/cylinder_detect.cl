const int CLOUD_SIZE = 14976;
const float EPSILON = 0.12; // primary epsilon value
const float EPS_2 = 0.07; // secondary epsilon value

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
	float3 norm = normalize(cross(v1, v2));

	// filter out nullvectors
	if(norm.x == 0 && norm.y == 0 && norm.z == 0)
	{
		norm = (float3)(0, 1, 0);
	};

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
	data[g_id].w = fabs(dot(n, point - p)) < EPSILON ? 0.25 : 0;
}

// fit cylinder by finding a circle on the ground plane
__kernel void calcCylinder(
	__global int3*   rand,
	__global float3* data,
	__global float3* normals,
	__global float4* cylinders)
{
	int g_id = get_global_id(0);
	float3 p1 = data[rand[g_id].x];
	float3 p2 = data[rand[g_id].y];
	float3 p3 = data[rand[g_id].z];
	float3 n  = normals[0];

	// orthogonal unit vectors spanning the plane
	float3 u = normalize(p2 - p1);
	float3 v = cross(n, u);

	// projecting onto plane
	float bx = dot(p2 - p1, u) * 0.5;
	float cx = dot(p3 - p1, u);
	float cy = dot(p3 - p1, v);

	float h = ((cx - bx) * (cx - bx) + cy * cy - bx * bx) / (2 * cy);

	// centerpoint of circle
	float3 k = p1 + bx * u + h * v;
	float  r = distance(k, p1);

	cylinders[g_id] = (float4)(k, r);
}

__kernel void fitCylinder(
	__global float4* data,
	__global float3* normals,
	__global float4* cylinders,
	__global int*	 inliers)
{
	int g_id0 = get_global_id(0);
	int g_id1 = get_global_id(1);
	float4 cylinder = cylinders[g_id0];
	float3 p = data[g_id1].xyz;
	float3 n = normals[0];

	// project point onto the plane of the circle,
	float dist = dot(p - cylinder.xyz, n);
	float3 p_proj = p - dist * n;

	// check distance of p_proj and cylinder centerline
	if(fabs(distance(p_proj, cylinder.xyz) - cylinder.w) < EPS_2)
	{
		atomic_inc(&inliers[g_id0]);
	}
}

__kernel void reduceCylinder(
	__global int*    inliers,
	__global float4* cylinders,
	__local  int*    scratch,   // local for inlier values
	__local  int*	 idx)	    // local for plane global idx
{
	int l_id = get_local_id(0);
	int g_id = get_global_id(0);
	scratch[l_id] = inliers[g_id];
	idx[l_id]	  = g_id;

	barrier(CLK_LOCAL_MEM_FENCE);

	// local reduction
	// swap inlier values along with cylinder idx
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
		cylinders[get_group_id(0)] = cylinders[idx[0]];
	}
}

__kernel void colorCylinder(
	__global float4* data,
	__global float3* normals,
	__global float4* cylinders)
{
	int g_id = get_global_id(0);
	float3 p = data[g_id].xyz;
	float3 n = normals[0];
	float4 cylinder = cylinders[0];

	// project point onto the plane of the circle,
	float dist = dot(p - cylinder.xyz, n);
	float3 p_proj = p - dist * n;
	
	data[g_id].w += fabs(distance(p_proj, cylinder.xyz) - cylinder.w) < EPS_2 ? 0.75 : 0;
}