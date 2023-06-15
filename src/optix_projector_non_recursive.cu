/*
    This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
    Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.

    This program is free software: you can redistribute it and/or modify  
    it under the terms of the GNU General Public License as published by  
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <optix_device.h>
#include <cuda_runtime_api.h>

#include "../include/optix_projector_data_structs.h"
#include "../include/cuda/vec_utils.cuh"
#include "../include/cuda/phys_utils.cuh"
#include "../include/cuda/array_index.cuh"

#include "../include/cuda/ProjectionGeometryPolicies.cuh"

#include "../include/cuda/stack.cuh"
#include "../include/cuda/optix_device_functions.cuh"

extern "C" {
__constant__ COptixLaunchParams launch_params;
}

const float trace_epsilon = 0; //1e-7f;  // TODO(pavel): allow user choosing scene epsilon

extern "C" __global__ void __raygen__project_non_recursive()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenSBTData* sbtData = (RayGenSBTData*)optixGetSbtDataPointer();

    // printf("In __raygen__project_non_recursive, nviews = %u...", launch_params.proj_geom_devptr->nViews);

    // Progection view returned by geometry Policy 
    ProjectionView pv;
    optixDirectCall<void, CProjectionGeometry const&, uint32_t, ProjectionView &>
        (
            launch_params.geometry_policy_index,
            *(launch_params.proj_geom_devptr),
            idx.z,
            pv
        );

    Ray init_ray;

    if(launch_params.det_geometry_policy_index == OptiXCylinderPolicy)
    {
        generate_ray_multi_per_pixel_cylinder(pv, idx, launch_params.proj_geom_devptr->det_radius, launch_params.proj_geom_devptr->det_row_count, launch_params.proj_geom_devptr->det_col_count, dim.x, dim.y, init_ray);
    }
    else
    {
        generate_ray_multi_per_pixel(pv, idx, launch_params.proj_geom_devptr->det_row_count, launch_params.proj_geom_devptr->det_col_count, dim.x, dim.y, init_ray);
    }
    
    // TODO: put everything related to tracing into a function
    // Start ray tracing:
    float init_refractive_index = sbtData->media_refractive_index;

    // This material stuck won't work in non-recursive version
    // TODO(pavel): process nested meshed using hierarchical list of meshes from the user
    LocalMemStack<CMaterial> material_stack;
    material_stack.push(CMaterial(0.f, init_refractive_index));
    
    Ray     miss_ray_data;
    float   total_attenuation = 0.0f;

    uint32_t ray_depth=0;           // Dummy variable - depth is always 1
    bool     reached_miss = false;  // Dummy variable - only one trace call, no tracking of miss event 

    float ray_time = 0.f;
    if(launch_params.geometry_policy_index == 2)    // TODO: change to enum value (Mesh transformation geometry)
        ray_time = launch_params.proj_geom_devptr->mtg.view_keys[idx.z]; 

    // DEBUG_PRINTLN("Starting for ray time %.2f", ray_time);

    CHitMeshPolygonIndex hit_mesh_polygon_id;
    CHitMeshPolygonIndex prev_hit_mesh_polygon_id;
    uint32_t reflected = 0;

    trace_project(launch_params.handle, trace_epsilon, ray_time, init_ray.origin, init_ray.direction,
                    miss_ray_data, total_attenuation, ray_depth, reached_miss, hit_mesh_polygon_id, material_stack, prev_hit_mesh_polygon_id, reflected);

    float t;
    float2 hitpoint_projected;

    if(launch_params.det_geometry_policy_index == OptiXCylinderPolicy)
    {
        find_cylinder_detector_hitpoint(
            pv,
            launch_params.proj_geom_devptr->det_row_count,
            launch_params.proj_geom_devptr->det_col_count,
            miss_ray_data,
            t,
            hitpoint_projected,
            launch_params.proj_geom_devptr->det_radius);
    }
    else
    {
        find_detector_hitpoint(pv, launch_params.proj_geom_devptr->det_row_count, launch_params.proj_geom_devptr->det_col_count, init_ray, t, hitpoint_projected);
    }

    float3 det_hp = miss_ray_data.origin + miss_ray_data.direction*(fabs(t));
    
    // printf("[%.8f, %.8f, %.8f],\n", UNWRAP_FLOAT3(det_hp));

    if(t > 0.f)
    {   // TODO(pavel): replace this direct callable with the normal function that computes detected intensity
        optixDirectCall<void, float2 const &, CProjectionGeometry const &, uint32_t, uint32_t, float, float*>
            (
                launch_params.detector_policy_index, 
                hitpoint_projected,
                *(launch_params.proj_geom_devptr),
                idx.z,
                dim.z,
                total_attenuation,
                launch_params.sinogram
            );
    }
}

extern "C" __global__ void __miss__project_non_recursive()
{

}

extern "C" __global__ void __anyhit__project_non_recursive()
{
    const HitSBTData &sbtData = *(const HitSBTData*)optixGetSbtDataPointer();
    const float mesh_attenuation_coeff    = sbtData.material_attenuation;    

    uint32_t pl_total_attenuation    = optixGetPayload_0();

    const float ray_time = optixGetRayTime();
    // Pick triangle vertices:
    float3 triangle[3];
    optixGetTriangleVertexData(
        optixGetGASTraversableHandle(),
        optixGetPrimitiveIndex(),
        optixGetSbtGASIndex(), // GAS local SBT index of this primitive. Looks like optixGetSbtGASIndex() has not yet been implemented in OptiX 7.0,
        ray_time,
        &triangle[0]
    );

    // Convert triangle coordinates into world space
    // This is needed only for sample transformations mode
    const int   transform_list_size = optixGetTransformListSize();
    if(transform_list_size > 0)
    {
        float4 object_to_world[3] = 
        {
            {1.f, 0.f, 0.f, 0.f},
            {0.f, 1.f, 0.f, 0.f},
            {0.f, 0.f, 1.f, 0.f}
        };

        optix_impl::optixGetObjectToWorldTransformMatrix(object_to_world[0], object_to_world[1], object_to_world[2]);

        triangle[0] = optix_impl::optixTransformPoint(object_to_world[0], object_to_world[1], object_to_world[2], triangle[0]);
        triangle[1] = optix_impl::optixTransformPoint(object_to_world[0], object_to_world[1], object_to_world[2], triangle[1]);
        triangle[2] = optix_impl::optixTransformPoint(object_to_world[0], object_to_world[1], object_to_world[2], triangle[2]);
    }

    float3 AB = triangle[1] - triangle[0];
    float3 AC = triangle[2] - triangle[0];
    
    float3 face_normal = normalize(cross(AB, AC));

    // Hit point from barycentric coordinates gives higher precision than using t-parameter from the ray equaition
    // float3 hit_point = optixGetWorldRayOrigin() + scale(optixGetRayTmax(), optixGetWorldRayDirection());
    const float2 barycentrics = optixGetTriangleBarycentrics();
    float3 hit_point = triangle[0]*(1-barycentrics.x-barycentrics.y) + triangle[1]*barycentrics.x + triangle[2]*barycentrics.y;
   
    float current_attenuation = uint_as_float(pl_total_attenuation);

    float length = norm(hit_point - optixGetWorldRayOrigin());
    float norm_dot_ray_sign = signbit(dot(face_normal, optixGetWorldRayDirection())) == 0? 1.0f : -1.0f;
    current_attenuation +=  (length * mesh_attenuation_coeff) * norm_dot_ray_sign;
    // current_attenuation +=  1;

    const uint3 idx = optixGetLaunchIndex();
    if(idx.x == 4 && idx.y == 11 && idx.z == 0)
    {
        printf("Ray [%d, %d, %d] with attenuation %.4f!... ", idx.x, idx.y, idx.z, current_attenuation);
    }

    optixSetPayload_0(float_as_uint(current_attenuation));

    optixIgnoreIntersection	();
}

extern "C" __global__ void __closesthit__project_non_recursive()
{

}