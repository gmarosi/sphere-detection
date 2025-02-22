const int CLOUD_SIZE = 14976;
const float EPSILON = 0.05;

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
// center: intersection of the perpendicular bisectors of 2 chords
__kernel void calcCylinder(
	__global int3*   rand,
	__global float3* data,
	__global float3* normals,
	__global float3* cylinders)
{
	int g_id = get_global_id(0);
	float3 p1 = data[rand[g_id].x];
	float3 p2 = data[rand[g_id].y];
	float3 p3 = data[rand[g_id].z];
	float3 n  = normals[0];

	// midpoints of the chords (in the plane)
	float2 k1 = (float2)((p1.x + p2.x)/2.0, (p1.z + p2.z)/2.0);
	float2 k2 = (float2)((p1.x + p3.x)/2.0, (p1.z + p3.z)/2.0);

	// vectors perpendicular to the chords (in the plane)
	float2 v1 = cross(p2 - p1, n).xz;
	float2 v2 = cross(p3 - p1, n).xz;

	// solve: alpha * v1 + k1 = beta * v2 + k2 = c
	float alpha = ( v2.x * (k1.y - k2.y) + v2.y * (k2.x - k1.x) ) / ( v1.x * v2.y - v1.y * v2.x );
	float2 c = alpha * v1 + k1;
	float r = distance(c, p1.xz);

	cylinders[g_id] = (float3)(c, r);
}

__kernel void fitCylinder(
	__global float4* data,
	__global float3* cylinders,
	__global int*	 inliers,
	int ITER_NUM)
{
	int g_id = get_global_id(0);
	float2 point = data[g_id].xz;

	for(int i = 0; i < ITER_NUM; i++)
	{
		float3 cylinder = cylinders[i];

		if(fabs(distance(point, cylinder.xy) - cylinder.z) < EPSILON)
		{
			atomic_inc(&inliers[i]);
		}
	}
}

__kernel void reduceCylinder(
	__global int*    inliers,
	__global float3* cylinders,
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
	__global float3* cylinders)
{
	int g_id = get_global_id(0);
	float2 point = data[g_id].xz;

	float3 cylinder = cylinders[0];
	
	data[g_id].w += fabs(distance(point, cylinder.xy) - cylinder.z) < EPSILON ? 0.75 : 0;
}