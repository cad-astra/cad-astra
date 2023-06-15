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

#pragma once

#include <cuda_runtime_api.h>
#include <optix_device.h>

#include "ProjectionGeometryPolicies.cuh"
#include "stack.cuh"
#include "CMaterial.cuh"

#define UNWRAP_FLOAT3(P) (P).x, (P).y, (P).z
struct Ray
{
    float3 origin;
    float3 direction;
};

// This index is used to detect polygon self-intersecions and neighboring polygon hits
struct CHitMeshPolygonIndex
{
    __device__ CHitMeshPolygonIndex() : m_mesh_sbt_index(-1), m_polygon_index(-1), m_is_front_face(false) {}
    int32_t m_mesh_sbt_index;
    int32_t m_polygon_index;
    bool    m_is_front_face;
};

// TODO: try passing hit data to optixTrace using the struct, check performance
// struct CHitData
// {
//     __device__ CHitData() {}
//     bool is_front_face_hit;
//     int32_t m_mesh_sbt_index;
//     int32_t m_polygon_index;
// }

/**
 * @brief Generate a ray for the specified projection view aiming to the center of the detector pixel.
 * 
 * @param[in]  pv Projection view structure.
 * @param[in]  idx 3D index of the ray.
 * @param[in]  det_row_count Number of detector rows.
 * @param[in]  det_col_count Number of detector columns.
 * @param[out] ray Generated ray. Ray direction is not a unit vector.
 */
__forceinline__ __device__ void generate_ray_single_per_pixel
(
    const ProjectionView &pv,
    uint3 idx,
    uint32_t det_row_count, uint32_t det_col_count,
    Ray &ray
)
{
    float3 det_topleft_corner = pv.det_center - (pv.det_u * det_col_count + pv.det_v * det_row_count)*0.5f;

    ray.origin    = pv.source;
    ray.direction = det_topleft_corner + scale(idx.y, pv.det_u)  + scale(idx.x, pv.det_v) - pv.source + (pv.det_u + pv.det_v)*0.5f;

    // printf("ray.direction = (%.8f, %.8f, %.8f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
}

/**
 * @brief Generate ray for the projection view.
 * 
 * @param[in]  pv Projection view structure.
 * @param[in]  idx 3D index of the ray. 
 * @param[in]  det_row_count Number of detector rows.
 * @param[in]  det_col_count Number of detector columns.
 * @param[in]  rays_rows Number of ray rows.
 * @param[in]  rays_cols Nuber of ray columns.
 * @param[out] ray Generated ray. Ray direction is not a unit vector.
 */
__forceinline__ __device__ void generate_ray_multi_per_pixel
(
    const ProjectionView &pv,
    uint3 idx,
    uint32_t det_row_count, uint32_t det_col_count,
    uint32_t rays_rows, uint32_t rays_cols,
    Ray &ray
)
{
    float3 ray_u = pv.det_u * (static_cast<float>(det_col_count)/static_cast<float>(rays_cols));
    float3 ray_v = pv.det_v * (static_cast<float>(det_row_count)/static_cast<float>(rays_rows));

    float3 ray_topleft_corner = pv.det_center - (ray_u * rays_cols + ray_v * rays_rows)*0.5f;

    ray.origin    = pv.source;
    ray.direction = ray_topleft_corner + scale(idx.y, ray_u)  + scale(idx.x, ray_v) - pv.source + (ray_u + ray_v)*0.5f;

    // printf("ray.direction = (%.8f, %.8f, %.8f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
}

__forceinline__ __device__ void generate_ray_multi_per_pixel_cylinder
(
    const ProjectionView &pv,
    uint3 idx,
    float det_radius,   // Distance to the cylinder center along the detector plane vector
    uint32_t det_row_count, uint32_t det_col_count,
    uint32_t rays_rows, uint32_t rays_cols,
    Ray &ray
)
{
    //TODO: precompute as many detector/projection geometry parameters as possible

    // Use plane as a "front" face

    // Try scaling down the projector (distance and detector size) to improve floating point precision
    // float3 proj_dir = pv.det_center - pv.source;
    // float dist = norm(proj_dir);

    float3 det_topleft_corner = pv.det_center - 0.5f*(pv.det_u * (det_col_count) + pv.det_v * (det_row_count));
    
    // TODO: compute only ray_angle_step, reduce det_col_count from the formula
    float det_angle_2 = asinf(norm(pv.det_u)*det_col_count*0.5f / det_radius);
    float det_angle_step = 2.f * det_angle_2 / static_cast<float>(det_col_count);
    float ray_angle_step = det_angle_step * (static_cast<float>(det_col_count)/static_cast<float>(rays_cols));

    float3 ray_v   = pv.det_v * (static_cast<float>(det_row_count)/static_cast<float>(rays_rows));

    // start rotation from the top left "front" face corner
    // rotate around det_u x (det_u x det_v)
    float3 q = cross(pv.det_u, pv.det_v);
    if(dot(q, det_topleft_corner) > 0)
        q = -q; // this inverts rotation axis for the cylinder pixels, i.e, nornal vector for the dector top side

    float3 det_top_normal = cross(pv.det_u, q);
    
    float det_width_2 = norm(pv.det_u)*det_col_count*0.5f;
    float3 cylinder_center = pv.det_center - 0.5f*pv.det_v * (det_row_count) + q*(1/norm(q)) * sqrtf(det_radius*det_radius -det_width_2*det_width_2 );

    float3 ray_end = rotate_v(det_topleft_corner - cylinder_center, (idx.y+0.5)*ray_angle_step, det_top_normal) + scale(idx.x+0.5f, ray_v) + cylinder_center;

    // printf("[%.8f, %.8f, %.8f],\n", UNWRAP_FLOAT3(ray_end));

    ray.origin    = pv.source;
    ray.direction = ray_end - pv.source;

    // printf("ray.direction = (%.8f, %.8f, %.8f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
}

// ------------------------------------------------------------------------------------------
// This function is taken from the "Ray Tracing Gems: High-Quality and Real-Time Rendering with DXR and Other APIs",
// CHAPTER 6 A Fast and Robust Method for Avoiding Self-Intersection, 2019
constexpr __device__ float origin() { return 1.0f / 32.0f; }
constexpr __device__ float float_scale() { return 1.0f / 65536.0f; }
constexpr __device__ float int_scale() { return 256.0f; }
// Normal points outward for rays exiting the surface, else is flipped.
__forceinline__ __device__ float3 offset_ray(const float3 p, const float3 n, bool is_codirect)
{
    float3 offset_dir = is_codirect? n : -n;
    int3 of_i;
    of_i.x = int_scale() * offset_dir.x;
    of_i.y = int_scale() * offset_dir.y;
    of_i.z = int_scale() * offset_dir.z;
    float3 p_i;
    p_i.x = int_as_float(float_as_int(p.x)+((p.x < 0) ? -of_i.x : of_i.x));
    p_i.y = int_as_float(float_as_int(p.y)+((p.y < 0) ? -of_i.y : of_i.y));
    p_i.z = int_as_float(float_as_int(p.z)+((p.z < 0) ? -of_i.z : of_i.z));
    return {fabsf(p.x) < origin() ? p.x+ float_scale()*n.x : p_i.x,
            fabsf(p.y) < origin() ? p.y+ float_scale()*n.y : p_i.y,
            fabsf(p.z) < origin() ? p.z+ float_scale()*n.z : p_i.z };
}
// ------------------------------------------------------------------------------------------
// These two functions are taken from OptiX 7 SDK example optixCutouts
static __forceinline__ __device__ void* unpackPointer( uint32_t i0, uint32_t i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}
// ------------------------------------------------------------------------------------------
__forceinline__ __device__ void trace_project
(
    OptixTraversableHandle    handle,               // IN
    float                     trace_epsilon,        // IN
    float                     ray_time,             // IN
    const float3             &source,               // IN
    const float3              direction,            // IN
    Ray                      &miss_ray_data,        // OUT
    float                    &total_attenuation,    // OUT
    uint32_t                 &ray_depth,            // OUT
    bool                     &reached_miss,         // OUT
    CHitMeshPolygonIndex     &hit_mesh_polygon_id,  // IN OUT
    LocalMemStack<CMaterial> &material_stack,        // IN OUT
    CHitMeshPolygonIndex     &prev_hit_mesh_polygon_id,  // IN OUT
    uint32_t                 &reflected
)
{
    // Start ray tracing:
    uint32_t payload_attenuation  = float_as_uint(total_attenuation);
    uint32_t payload_reached_miss = 0;

    // These payload registers are used to pack the pointer to miss ray structure
    uint32_t u0, u1;
    packPointer( &miss_ray_data, u0, u1 );

    // Pack the pointer to the head of materials stack:
    uint32_t material_stack_u0, material_stack_u1;
    packPointer( &material_stack, material_stack_u0, material_stack_u1);

    uint32_t payload_ray_depth = 0;

    uint32_t payload_mesh_sbt_index    = hit_mesh_polygon_id.m_mesh_sbt_index;  // Casting int to uint -> -1 turns into max(uint32_t)
    uint32_t payload_polygon_index     = hit_mesh_polygon_id.m_polygon_index;
    uint32_t payload_is_front_face_hit = hit_mesh_polygon_id.m_is_front_face? 1: 0;

    uint32_t payload_prev_hit_mesh_sbt_index    = prev_hit_mesh_polygon_id.m_mesh_sbt_index;
    uint32_t payload_prev_hit_polygon_index     = prev_hit_mesh_polygon_id.m_polygon_index;

    optixTrace( handle,
                source, direction, trace_epsilon /* tmin */, 1e16f /* tmax */, ray_time /* rayTime */,
                OptixVisibilityMask( 1 ),   // TODO: check what this parameter means
                OPTIX_RAY_FLAG_NONE,
                0,                   // SBT offset
                1,                   // SBT stride
                0,                   // missSBTIndex
                payload_attenuation,
                u0, u1, payload_ray_depth,
                payload_reached_miss,
                material_stack_u0, material_stack_u1,
                payload_mesh_sbt_index, payload_polygon_index,
                payload_is_front_face_hit,
                payload_prev_hit_polygon_index,   // Polygon ID from the previous hit event
                payload_prev_hit_mesh_sbt_index,  // Mesh ID from the previous hit event
                reflected
              );

    total_attenuation = uint_as_float(payload_attenuation);

    ray_depth += payload_ray_depth;

    reached_miss = payload_reached_miss;

    hit_mesh_polygon_id.m_mesh_sbt_index = payload_mesh_sbt_index;
    hit_mesh_polygon_id.m_polygon_index  = payload_polygon_index ;
    hit_mesh_polygon_id.m_is_front_face  = payload_is_front_face_hit==1? true:false;

    // const uint3 idx = optixGetLaunchIndex();
    // if(idx.x == 0 && idx.y == 3211839 && idx.z == 0)
    // {
    //     printf("payload_is_front_face_hit from trace_project for poly. id %u: %u... ", payload_polygon_index, payload_is_front_face_hit);
    // }

    prev_hit_mesh_polygon_id.m_mesh_sbt_index = payload_prev_hit_polygon_index;
    prev_hit_mesh_polygon_id.m_polygon_index  = payload_prev_hit_mesh_sbt_index;
}

/**
 * @brief Find ray-detector hit point in detector basis.
 * 
 * @param[in] pv Projection view structure.
 * @param[in] det_row_count Number of detector rows.
 * @param[in] det_col_count Number of detector columns.
 * @param[in] ray Incident ray.
 * @param[out] t Distance parameter along the incident ray.
 * @param[out] hitpoint_proj Hit point in detector plane basis.
 */
__forceinline__ __device__ void find_detector_hitpoint
(
    const ProjectionView &pv,
    uint32_t              det_row_count,
    uint32_t              det_col_count,
    const Ray            &ray,
    float                &t,
    float2               &hitpoint_proj
)
{
    float3 det_topleft_corner = pv.det_center - pv.det_u * (det_col_count*0.5f) - pv.det_v * (det_row_count*0.5f);
    hitpoint_proj = project_point(ray.origin, ray.direction, det_topleft_corner, pv.det_u, pv.det_v, t);

    // float3 hp_uvq = {hitpoint_proj.x, hitpoint_proj.y, 0};
    // float3 Q_uvq = hp_uvq ;
    // printf("[%.8f, %.8f, %.8f],\n", UNWRAP_FLOAT3(Q_uvq));
}

__forceinline__ __device__ float sqSolve(float a, float b, float c)
{
    float q = 1;

    // printf("b*b - 4 * a*c = %.2f\n", b*b - 4 * a*c);

    if(b > 0)
        q = -0.5 * (b + sqrtf(b*b - 4 * a*c));
    else
        q = -0.5 * (b - sqrtf(b*b - 4 * a*c));

    float t1 = q/a;
    float t2 = c/q;
    
    if(t1 > t2)
        return t1;
    else
        return t2;
}

__forceinline__ __device__ void find_cylinder_detector_hitpoint
(
    const ProjectionView &pv,
    uint32_t              det_row_count,
    uint32_t              det_col_count,
    const Ray            &ray,
    float                &t,
    float2               &hitpoint_proj,
    float                det_radius
)
{
    float3 det_topleft_corner = pv.det_center - 0.5f*(pv.det_u * (det_col_count) + pv.det_v * (det_row_count));

    float2 hp_uv = project_point(ray.origin, ray.direction, det_topleft_corner, pv.det_u, pv.det_v, t);

    float3 det_q = cross(pv.det_u, pv.det_v);
    if(dot(det_q, det_topleft_corner) > 0)
        det_q = -det_q; //

    float det_width_2 = norm(pv.det_u)*det_col_count*0.5f;
    float3 cylinder_center = pv.det_center - 0.5f*pv.det_v * (det_row_count) + det_q*(1/norm(det_q)) * sqrtf(det_radius*det_radius -det_width_2*det_width_2 );

    float2 hp_uq =  {hp_uv.x, 0};

    float3 cylinder_center_uvq = switch_basis(cylinder_center-det_topleft_corner, pv.det_u, pv.det_v, det_q);

    float2 cylinder_center_uq = {cylinder_center_uvq.x, cylinder_center_uvq.z};
    float2 L_uq = hp_uq - cylinder_center_uq;

    float  R = norm(cylinder_center_uq);

    float3 d_uvq = switch_basis(ray.direction, pv.det_u, pv.det_v, det_q);
    float2 d_uq = {d_uvq.x, d_uvq.z};

    float ellipse_a = det_radius / norm(pv.det_u);
    float ellipse_b = det_radius / norm(det_q); //R - cylinder_center_uq.y;

    float ellipse_a_2 = ellipse_a * ellipse_a;
    float ellipse_b_2 = ellipse_b * ellipse_b;

    float a = ellipse_b_2*d_uq.x*d_uq.x + ellipse_a_2*d_uq.y*d_uq.y;                           //   dot(d_uq, d_uq);
    float b = 2*(d_uq.x*ellipse_b_2*L_uq.x + d_uq.y*ellipse_a_2*L_uq.y);                       // 2*dot(d_uq, L_uq);
    float c = L_uq.x*L_uq.x*ellipse_b_2 + L_uq.y*L_uq.y*ellipse_a_2 - ellipse_a_2*ellipse_b_2; //   dot(L_uq, L_uq) - R*R;
    
    float t_uvq  = sqSolve(a, b, c);

    float3 hp_uvq = {hp_uv.x, hp_uv.y, 0};
    float3 Q_uvq = hp_uvq + t_uvq*d_uvq;

    float3 Txyz_0 = {pv.det_u.x, pv.det_v.x, det_q.x};
    float3 Txyz_1 = {pv.det_u.y, pv.det_v.y, det_q.y};
    float3 Txyz_2 = {pv.det_u.z, pv.det_v.z, det_q.z};

    float3 Q_xyz = {dot(Txyz_0, Q_uvq), dot(Txyz_1, Q_uvq), dot(Txyz_2, Q_uvq)};

    Q_xyz = Q_xyz + det_topleft_corner;

    // printf("[%.8f, %.8f, %.8f],\n", UNWRAP_FLOAT3(Q_xyz));

    float3 det_curve_center = cylinder_center - det_q*(1/norm(det_q))*det_radius;
    float3 cylinder_center_radius_vector  = det_curve_center   - cylinder_center;

    float3 cylinder_topleft_radius_vector = det_topleft_corner - cylinder_center;
    float3 hit_point_radius_vector = Q_xyz - pv.det_v*Q_uvq.y - cylinder_center;

    float det_hit_angle = acosf(dot(cylinder_topleft_radius_vector, hit_point_radius_vector) / (norm(cylinder_topleft_radius_vector)*norm(hit_point_radius_vector)));

    // TODO: precompute detector angle step
    float det_angle_2 = asinf(norm(pv.det_u)*det_col_count*0.5f / det_radius);
    float det_angle_step = 2.f * det_angle_2 / static_cast<float>(det_col_count);

    hitpoint_proj = {det_hit_angle/det_angle_step, Q_uvq.y};

    // printf("[%.8f, %.8f],\n", hitpoint_proj.x, hitpoint_proj.y);
}