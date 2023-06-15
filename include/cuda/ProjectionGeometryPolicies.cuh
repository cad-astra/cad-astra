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

#include "common.cuh"

#include "vec_utils.cuh"
#include "../proj_geometry.h"
#include "optix_projector_policies.cuh"

/* 
This struct replicates ASTRA projetion view:
source : source position

d : the center of the detector

u : the vector from detector pixel (0,0) to (0,1)

v : the vector from detector pixel (0,0) to (1,0)
*/
struct ProjectionView
{
    float3 source, det_center, det_u, det_v;
};

// rot_x = lambda x, theta: np.array([x[0], x[1]*np.cos(theta)-x[2]*np.sin(theta), x[1]*np.sin(theta)+x[2]*np.cos(theta)])
// rot_y = lambda x, theta: np.array([x[0]*np.cos(theta)-x[2]*np.sin(theta), x[1], x[0]*np.sin(theta)+x[2]*np.cos(theta)])
// rot_z = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

__forceinline__ __device__ float3 rotate_x(const float3 &point, float angle)
{
    /* Rotate a point around the x-axis */
    float sin_angle, cos_angle; 
    sincosf(angle, &sin_angle, &cos_angle);
    return float3
    {
        point.x,
        cos_angle*point.y - sin_angle*point.z,
        sin_angle*point.y + cos_angle*point.z,
    };
}

__forceinline__ __device__ float3 rotate_y(const float3 &point, float angle)
{
    /* Rotate a point around the y-axis */
    float sin_angle, cos_angle; 
    sincosf(angle, &sin_angle, &cos_angle);
    return float3
    {
        cos_angle*point.x - sin_angle*point.z,
        point.y,
        sin_angle*point.x + cos_angle*point.z
    };
}

__forceinline__ __device__ float3 rotate_z(const float3 &point, float angle)
{
    /* Rotate a point around the z-axis */
    float sin_angle, cos_angle; 
    sincosf(angle, &sin_angle, &cos_angle);
    return float3
    {
        cos_angle*point.x - sin_angle*point.y,
        sin_angle*point.x + cos_angle*point.y,
        point.z
    };
}

__forceinline__ __device__ float3 rotate_v(const float3 &point, float angle, const float3 &v)
{
    float r = norm(v);
    float theta = asinf(v.z / r);  // same as (pi/2 -theta)
    float phi   = atan2(v.y, v.x);

    // printf("theta=%.4f, phi=%.4f\n", theta, phi);

    float3 point_rotated 
                  = rotate_z(point, -phi);
    point_rotated = rotate_y(point_rotated,-theta);
    
    point_rotated = rotate_x(point_rotated, angle);

    point_rotated = rotate_y(point_rotated, theta);
    point_rotated = rotate_z(point_rotated, phi);

    // printf("point_rotated=[%.8f, %.8f, %.8f],\n", point_rotated.x, point_rotated.y, point_rotated.z);

    return point_rotated;
}

template<typename T_3DIndex>
struct CRotationGeometryPolicy
{
    __forceinline__ __device__ static void get_projection_view(const CProjectionGeometry &pg,
                                                               uint32_t i_view,
                                                               ProjectionView &pv)
    {
        // In this implementation, we rotate only projection vector
        // float3 proj_direction = normalize(rotate_z(float3{0.0,  pg.rg.origin_det,    0.0} - float3{0.0, -pg.rg.source_origin, 0.0}, pg.rg.angles[i_view]));
        // pv.source     = proj_direction * (-pg.rg.source_origin);
        // pv.det_center = proj_direction *   pg.rg.origin_det;
        pv.source     = rotate_z(float3{0.0, -pg.rg.source_origin, 0.0}, pg.rg.angles[i_view]);
        pv.det_center = rotate_z(float3{0.0,  pg.rg.origin_det,    0.0}, pg.rg.angles[i_view]);
        pv.det_u      = rotate_z(float3{pg.rg.det_width,      0.0, 0.0}, pg.rg.angles[i_view]);
        pv.det_v      = rotate_z(float3{0.0, 0.0,     pg.rg.det_height}, pg.rg.angles[i_view]);
    }
    __forceinline__ __host__ static OptiXGeometryPolicyIndex get_optix_policy_idx()
    {
        return OptiXRotationPolicy;
    }

    static std::string type;
};

template<class T_3DIndex>
std::string CRotationGeometryPolicy<T_3DIndex>::type = "rotation_geom";

template<typename T_3DIndex>
struct CVecGeometryPolicy
{
    __forceinline__ __device__ static  void get_projection_view(const CProjectionGeometry &pg,
                                                                uint32_t i_view,
                                                                ProjectionView &pv)
    {
        // Unpack views array, store projection view into pv
        T_3DIndex view_lin_index(pg.nViews, 12);
        pv.source     = UNPACK_3D_POINT(pg.vg.views, view_lin_index, i_view, 0);
        pv.det_center = UNPACK_3D_POINT(pg.vg.views, view_lin_index, i_view, 1*3);
        pv.det_u      = UNPACK_3D_POINT(pg.vg.views, view_lin_index, i_view, 2*3);
        pv.det_v      = UNPACK_3D_POINT(pg.vg.views, view_lin_index, i_view, 3*3);
    }
    __forceinline__ __host__ static OptiXGeometryPolicyIndex get_optix_policy_idx()
    {
        return OptiXVectorPolicy;
    }

    static std::string type;
};

template<class T_3DIndex>
std::string CVecGeometryPolicy<T_3DIndex>::type = "vec_geom";

template<typename T_3DIndex>
struct CMeshTransformGeometryPolicy
{
    __forceinline__ __device__ static  void get_projection_view(const CProjectionGeometry &pg,
                                                                uint32_t i_view,
                                                                ProjectionView &pv)
    {
        T_3DIndex view_lin_index(1, 12);    // Only one projector view is available at the moment...
        pv.source     = UNPACK_3D_POINT(pg.mtg.projector_view, view_lin_index, 0, 0);
        pv.det_center = UNPACK_3D_POINT(pg.mtg.projector_view, view_lin_index, 0, 1*3);
        pv.det_u      = UNPACK_3D_POINT(pg.mtg.projector_view, view_lin_index, 0, 2*3);
        pv.det_v      = UNPACK_3D_POINT(pg.mtg.projector_view, view_lin_index, 0, 3*3);
    }
    __forceinline__ __host__ static OptiXGeometryPolicyIndex get_optix_policy_idx()
    {
        return OptiXMeshTransformPolicy;
    }

    static std::string type;
};

template<class T_3DIndex>
std::string CMeshTransformGeometryPolicy<T_3DIndex>::type = "mesh_transform";