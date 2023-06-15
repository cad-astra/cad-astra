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

// Debug pring block for a ray
//
// const uint3 idx = optixGetLaunchIndex();
// if(idx.x == 0 && idx.y == 1599871 && idx.z == 0)
// {
//     DEBUG_PRINTLN("hit_point     = np.array([%.8f, %.8f, %.8f]) # is front face: %s", UNWRAP_FLOAT3(hit_point), is_front_face_hit? "true" : "false");
//     DEBUG_PRINTLN("hit_points    = np.vstack((hit_points, hit_point))");
//     DEBUG_PRINTLN("ray_direction = np.array([%.8f, %.8f, %.8f])", UNWRAP_FLOAT3(new_direction));
//     DEBUG_PRINTLN("triangle      = np.array([[%.8f, %.8f, %.8f], [%.8f, %.8f, %.8f], [%.8f, %.8f, %.8f]])", UNWRAP_FLOAT3(triangle[0]), UNWRAP_FLOAT3(triangle[1]), UNWRAP_FLOAT3(triangle[2]));
//     DEBUG_PRINTLN("poligons    = np.vstack((poligons, triangle))");
//     DEBUG_PRINTLN("barycentrics  = np.array([%.8f, %.8f, %.8f])", barycentrics.x, barycentrics.y, 1-barycentrics.x - barycentrics.y);
//     DEBUG_PRINTLN("# t_hit         = %.8f", optixGetRayTmax());
//     DEBUG_PRINTLN("hit_point-origin = np.array([%.8f, %.8f, %.8f]", UNWRAP_FLOAT3((hit_point-optixGetWorldRayOrigin())));
//     DEBUG_PRINTLN("<hit_point-origin, ray_direction> = %.8f", dot(hit_point-optixGetWorldRayOrigin(), new_direction));
// }

extern "C" {
__constant__ COptixLaunchParams launch_params;
}

#define UNWRAP_FLOAT3(P) (P).x, (P).y, (P).z

const float trace_epsilon = 0; //1e-7f;  // TODO(pavel): allow user choosing scene epsilon
// Add point into ray path array
__device__ void add_point(RayPathData &trace_data, int ray_lin_index, CIndexer3D_C ray_vertex_coord_index, const float3 point)
{
    trace_data.paths_ptr[ray_vertex_coord_index(ray_lin_index, trace_data.length, 0)] = point.x;
    trace_data.paths_ptr[ray_vertex_coord_index(ray_lin_index, trace_data.length, 1)] = point.y;
    trace_data.paths_ptr[ray_vertex_coord_index(ray_lin_index, trace_data.length, 2)] = point.z;
    trace_data.length++;
}

extern "C" __global__ void __raygen__project()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenSBTData* sbtData = (RayGenSBTData*)optixGetSbtDataPointer();

    // printf("In __raygen__project, nviews = %u...", launch_params.proj_geom_devptr->nViews);

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

    // Stack of materials along the ray path:
    // TODO(pavel): make top-level material configurable via either SBT or launch parameters
    LocalMemStack<CMaterial> material_stack;
    material_stack.push(CMaterial(0.f, init_refractive_index));
    
    Ray     miss_ray_data;
    float   total_attenuation = 0.0f;

    uint32_t ray_depth=0;
    bool     reached_miss = false;

    float ray_time = 0.f;
    if(launch_params.geometry_policy_index == 2)    // TODO: change to enum value (Mesh transformation geometry)
        ray_time = launch_params.proj_geom_devptr->mtg.view_keys[idx.z]; 

    // DEBUG_PRINTLN("Starting for ray time %.2f", ray_time);

    CHitMeshPolygonIndex hit_mesh_polygon_id;
    CHitMeshPolygonIndex prev_hit_mesh_polygon_id;
    uint32_t reflected = 0;
    while(!reached_miss)
    {
        trace_project(launch_params.handle, trace_epsilon, ray_time, init_ray.origin, init_ray.direction,
                      miss_ray_data, total_attenuation, ray_depth, reached_miss, hit_mesh_polygon_id, material_stack, prev_hit_mesh_polygon_id, reflected);
        init_ray.origin    = miss_ray_data.origin;
        init_ray.direction = miss_ray_data.direction;
    }

    if (ray_depth >= launch_params.ray_maxdepth)
    {
        atomicAdd(launch_params.pMax_depth_rays_cnt, 1);
        // printf("Warning: maximal depth of %d reached for ray [%d, %d, %d] with attenuation %.4f!\n", ray_depth, idx.x, idx.y, idx.z, total_attenuation);
    }

    atomicMax(launch_params.pMax_actual_depth, ray_depth);

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
        find_detector_hitpoint(pv, launch_params.proj_geom_devptr->det_row_count, launch_params.proj_geom_devptr->det_col_count, miss_ray_data, t, hitpoint_projected);
    }

    float3 det_hp = miss_ray_data.origin + miss_ray_data.direction*(fabs(t));
    
    // printf("[%.8f, %.8f, %.8f],\n", UNWRAP_FLOAT3(det_hp));

    // if(idx.x == 162 && idx.y == 99 && idx.z == 33) {DEBUG_PRINTLN("Pixel attenuation = %.4f", total_attenuation);}

    // if(material_stack.size() != 1)
    //     printf("MATERIAL STACK IS BROKEN for ray (%u, %u, %u), its size = %u; ", idx.x, idx.y, idx.z, material_stack.size() );

    // TODO(pavel): compute ray attenuation for the mesh-detector segment (top-level material)

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

extern "C" __global__ void __miss__project()
{
    // Unpack pointer to ray data from two payload registers
    const uint32_t u0 = optixGetPayload_1();
    const uint32_t u1 = optixGetPayload_2();
    Ray *pTrace_data = reinterpret_cast<Ray*>( unpackPointer( u0, u1 ) );

    pTrace_data->origin    = optixGetWorldRayOrigin();
    pTrace_data->direction = optixGetWorldRayDirection();

    optixSetPayload_4(1);   // Set "Ray reached miss" flag to true
}

extern "C" __global__ void __anyhit__project()
{
    uint32_t payload_prev_hit_mesh_sbt_index      = optixGetPayload_7();
    uint32_t payload_prev_hit_polygon_index       = optixGetPayload_8();
    uint32_t payload_prev_is_front_face_hit       = optixGetPayload_9();
    uint32_t payload_prev_prev_hit_polygon_index  = optixGetPayload_10();
    uint32_t payload_prev_prev_hit_mesh_sbt_index = optixGetPayload_11();
    uint32_t payload_reflected                    = optixGetPayload_12();

    uint32_t hit_polygon_index  = optixGetPrimitiveIndex();
    uint32_t hit_mesh_sbt_index = optixGetSbtGASIndex();
    bool     is_front_face_hit  = optixIsTriangleFrontFaceHit();

    const uint3 idx = optixGetLaunchIndex();

    if(payload_reflected < 1)   // Ray wasn't reflected on the last hit event
    {
        // Only single entrance/exit is allowed for the transmitted ray
        if(
            ( is_front_face_hit  == (payload_prev_is_front_face_hit == 1) ) &&
            ( hit_mesh_sbt_index ==  payload_prev_hit_mesh_sbt_index )
        )
        {
            optixIgnoreIntersection();
        }
        // No backward propagation allowed for the transmitted ray
        if(
            ( hit_polygon_index == payload_prev_prev_hit_polygon_index) &&
            ( hit_mesh_sbt_index == payload_prev_prev_hit_mesh_sbt_index)
          )
        {
            if(idx.x == 0 && idx.y == 3211839 && idx.z == 0)
                printf("ignored intersection: looped propagation... ");
            optixIgnoreIntersection();
        }
    }
    else    // For the reflected ray, only self-intersections are not allowed
    {
        if(
            ( hit_polygon_index  ==  payload_prev_hit_polygon_index ) &&
            ( hit_mesh_sbt_index ==  payload_prev_hit_mesh_sbt_index )
        )
        {
            optixIgnoreIntersection();
        }
    }
}

extern "C" __global__ void __closesthit__project()
{
    /* Get Material properties from SBT record */
    const HitSBTData &sbtData = *(const HitSBTData*)optixGetSbtDataPointer();
    const float mesh_attenuation_coeff = sbtData.material_attenuation;
    const float mesh_refractive_index  = sbtData.material_refractive_index;
    const uint32_t max_depth           = sbtData.max_depth;

    /* Get ray payloads */
    uint32_t payload_total_attenuation             = optixGetPayload_0();
    uint32_t u0                                    = optixGetPayload_1();
    uint32_t u1                                    = optixGetPayload_2();
    uint32_t ray_depth                             = optixGetPayload_3();
    uint32_t material_stack_u0                     = optixGetPayload_5();
    uint32_t material_stack_u1                     = optixGetPayload_6();
    uint32_t payload_prev_hit_mesh_sbt_index       = optixGetPayload_7();
    uint32_t payload_prev_hit_polygon_index        = optixGetPayload_8();

    // Unpack the pointer to the material stack
    float current_attenuation = uint_as_float(payload_total_attenuation);
    LocalMemStack<CMaterial> *pMaterial_stack =
        reinterpret_cast<LocalMemStack<CMaterial> *>( unpackPointer( material_stack_u0, material_stack_u1 ) );

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
    
    float3 normal = normalize(cross(AB, AC));

    // Hit point from barycentric coordinates gives higher precision than using t-parameter from the ray equaition
    // float3 hit_point = optixGetWorldRayOrigin() + scale(optixGetRayTmax(), optixGetWorldRayDirection());
    const float2 barycentrics = optixGetTriangleBarycentrics();
    float3 hit_point = triangle[0]*(1-barycentrics.x-barycentrics.y) + triangle[1]*barycentrics.x + triangle[2]*barycentrics.y;

    float3 new_origin = hit_point;
    float3 new_direction = optixGetWorldRayDirection();
    uint32_t hit_polygon_index  = optixGetPrimitiveIndex();
    uint32_t hit_mesh_sbt_index = optixGetSbtGASIndex();
    bool is_front_face_hit = optixIsTriangleFrontFaceHit();
    const uint3 idx = optixGetLaunchIndex();
    ray_depth++;

    // float ray_propagation_test = dot(hit_point-optixGetWorldRayOrigin(), new_direction);

    // // DEBUG_PRINTLN("hit_point     = np.array([%.8f, %.8f, %.8f]) # is front face: %s", UNWRAP_FLOAT3(hit_point), is_front_face_hit? "true" : "false");
    // // DEBUG_PRINTLN("ray_propagation_test = %.8f", ray_propagation_test);
    bool reflected = false;

    // Account for ray attenuation and refraction
    
    // We compute the intersection length since ray direction is not normalized.
    // Non-normalized ray direction gives higher accuracy in finding hit points
    // when compared to union direction vector.
    float length = norm(hit_point - optixGetWorldRayOrigin());

    // Refractive indices:
    // n1 - incident ray media,
    // n2 - transmitted ray media
    float n1 = 1.0f;
    float n2 = 1.0f;

    // Pick material properties, i.e., refractive index and attenuation coeff.
    float material_attenuation_coeff;
    if(is_front_face_hit)
    {
        CMaterial m;
        if(pMaterial_stack->head(m) == STACK_OPERATION_SUCCESS)
        {
            material_attenuation_coeff = m.m_attenuation_coeff;
            n1 = m.m_refractive_index;
        }
        // else
        // {
        //     TODO(pavel): throw OptiX Exception if reading from stack failed - attemprt of reading from empty stack
        // }
        n2 = mesh_refractive_index;
        if(pMaterial_stack->push(CMaterial(mesh_attenuation_coeff, mesh_refractive_index, hit_mesh_sbt_index)) != STACK_OPERATION_SUCCESS)
        {
            // TODO(pavel): throw OptiX Exception if pushing to stack failed, meaning that stack reached its maximum size
        }
    }
    else // is back face hit - ray is leaving the material
    {
        CMaterial material_from;
        if(pMaterial_stack->pop(material_from) == STACK_OPERATION_SUCCESS)
        {
            if(material_from.m_mesh_sbt_index == hit_mesh_sbt_index)    // No mesh overlap
            {
                material_attenuation_coeff = material_from.m_attenuation_coeff;
                n1 = material_from.m_refractive_index;
                CMaterial material_to;
                if(pMaterial_stack->head(material_to) == STACK_OPERATION_SUCCESS)
                {
                    n2 = material_to.m_refractive_index;
                }
                // else
                // {
                //     TODO(pavel): throw OptiX Exception if reading from stack failed - attemprt of reading from empty stack
                // }
            }
            else    // Here, we consider the case of overlapping meshes
            {
                material_attenuation_coeff = material_from.m_attenuation_coeff;
                n1 = material_from.m_refractive_index;
                n2 = material_from.m_refractive_index;
                // CMaterial m_prev;
                if(pMaterial_stack->remove_last() == STACK_OPERATION_SUCCESS)
                { }
                else { /*TODO: Exception */ }
                pMaterial_stack->push(material_from);
            }
        }
        // else
        // {
        //     TODO(pavel): throw OptiX Exception if popping from stack failed, i.e., stack is empty
        // }
    }
    current_attenuation +=  length * material_attenuation_coeff;

    // if(idx.x == 1660 && idx.y == 1703 && idx.z == 7)
    // {
    //     DEBUG_PRINTLN("hit_point     = np.array([%.8f, %.8f, %.8f]) # is front face: %s", UNWRAP_FLOAT3(hit_point), is_front_face_hit? "true" : "false");
    //     DEBUG_PRINTLN("hit_points    = np.vstack((hit_points, hit_point))");
    //     DEBUG_PRINTLN("ray_direction = np.array([%.8f, %.8f, %.8f])", UNWRAP_FLOAT3(new_direction));
    //     DEBUG_PRINTLN("triangle      = np.array([[%.8f, %.8f, %.8f], [%.8f, %.8f, %.8f], [%.8f, %.8f, %.8f]])", UNWRAP_FLOAT3(triangle[0]), UNWRAP_FLOAT3(triangle[1]), UNWRAP_FLOAT3(triangle[2]));
    //     DEBUG_PRINTLN("poligons      = np.vstack((poligons, triangle))");
    //     DEBUG_PRINTLN("# barycentrics  = np.array([%.8f, %.8f, %.8f])", barycentrics.x, barycentrics.y, 1-barycentrics.x - barycentrics.y);
    //     DEBUG_PRINTLN("# t_hit         = %.8f", optixGetRayTmax());
    //     DEBUG_PRINTLN("# hit_point-origin = np.array([%.8f, %.8f, %.8f]", UNWRAP_FLOAT3((hit_point-optixGetWorldRayOrigin())));
    //     DEBUG_PRINTLN("# <hit_point-origin, ray_direction> = %.8f", dot(hit_point-optixGetWorldRayOrigin(), new_direction));
    // }
    // ////////////////////////////////////////////
    if(n1 != n2)
    {
        reflected = !bend_ray(new_direction, normal, n1 / n2);
        // TODO(pavel): check if non-normalized direction gives higher precision
        if(reflected)
        {
            if(!is_front_face_hit)  // is back face hit
            {
                // Push current material back into stack in case of total internal reflection
                pMaterial_stack->push(CMaterial(mesh_attenuation_coeff, mesh_refractive_index));
            }
            else
            {
                // Remove current material in case of total external reflection (this can happen when n < 1.0)
                pMaterial_stack->remove_last();
            }
        }
    }
    // if(idx.x == 0 && idx.y == 1400508 && idx.z == 0)
    // {
    //     DEBUG_PRINTLN("time=%.1f, sbt_attenuation_coeff = %.2f, n1 = %.2f, n2 = %.2f, material stack size: %lu, curr_atten: %.2f, polyg. id: %d, is_front_face: %s", ray_time, material_attenuation_coeff, n1, n2, pMaterial_stack->size(), current_attenuation, hit_polygon_index, is_front_face_hit? "true":"false");
    // }

    uint32_t payload_is_front_face_hit = is_front_face_hit? 1 : 0;
    // if(idx.x == 0 && idx.y == 3211839 && idx.z == 0 && hit_polygon_index == 1319178)
    //     printf("in closest_hit: payload_is_front_face_hit=%d...", payload_is_front_face_hit);

    if (ray_depth == max_depth)
    {
        // Remember tracing state for the next optixTrace call
        Ray *pTrace_data = reinterpret_cast<Ray*>( unpackPointer( u0, u1 ) );

        pTrace_data->origin    = new_origin;
        pTrace_data->direction = new_direction;

        optixSetPayload_0(float_as_uint(current_attenuation));
        optixSetPayload_3(ray_depth);
        optixSetPayload_7(hit_mesh_sbt_index);
        optixSetPayload_8(hit_polygon_index);
        optixSetPayload_9(payload_is_front_face_hit);
        optixSetPayload_10(payload_prev_hit_polygon_index);
        optixSetPayload_11(payload_prev_hit_mesh_sbt_index);
        optixSetPayload_12(reflected? 1 : 0);

        // printf("Depth of %d reached: %d\n", max_depth);
    }
    else
    {
        // Start ray tracing for the transmitted ray:
        unsigned int payload_total_attenuation = float_as_uint(0.f);
        unsigned int payload_reached_miss = 0;

        uint32_t payload_reflected = reflected? 1 : 0;

        packPointer(pMaterial_stack, material_stack_u0, material_stack_u1);

        optixTrace( launch_params.handle,
                    new_origin, new_direction, trace_epsilon /* tmin */, 1e16f /* tmax */, ray_time /* rayTime */,
                    OptixVisibilityMask( 1 ),   // TODO: check what this parameter means
                    OPTIX_RAY_FLAG_NONE,
                    0,                   // SBT offset
                    1,                   // SBT stride
                    0,                   // missSBTIndex
                    payload_total_attenuation,
                    u0, u1, ray_depth,
                    payload_reached_miss,
                    material_stack_u0, material_stack_u1,
                    hit_mesh_sbt_index, hit_polygon_index,
                    payload_is_front_face_hit,        // Front face hit from the current hit event
                    payload_prev_hit_polygon_index,   // Polygon ID from the previous hit event
                    payload_prev_hit_mesh_sbt_index,  // Mesh ID from the previous hit event
                    payload_reflected
                    );

        float total_attenuation = uint_as_float(payload_total_attenuation);
        optixSetPayload_0(float_as_uint(current_attenuation + total_attenuation));
        optixSetPayload_3(ray_depth);
        optixSetPayload_4(payload_reached_miss);

        optixSetPayload_7(hit_mesh_sbt_index);
        optixSetPayload_8(hit_polygon_index);
        optixSetPayload_9(payload_is_front_face_hit);
        optixSetPayload_10(payload_prev_hit_polygon_index);
        optixSetPayload_11(payload_prev_hit_mesh_sbt_index);
        optixSetPayload_12(reflected? 1 : 0);
    }
}