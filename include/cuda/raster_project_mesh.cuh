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

#include <iostream>

#include <cuda_runtime_api.h>
#include "vec_utils.cuh"
#include "array_index.cuh"
#include "common.cuh"
#include "ProjectionGeometryPolicies.cuh"
#include "MeshRepresentationPolicies.cuh"

struct DetectorBoundingBox
{
    int xmin, xmax, ymin, ymax;
};

__device__ float max3(float a, float b, float c)
{
    float max_ab = (a > b? a : b);
    return max_ab > c? max_ab : c;
}

__device__ float min3(float a, float b, float c)
{
    float min_ab = (a < b? a : b);
    return min_ab < c? min_ab : c;
}

__device__ __forceinline__ DetectorBoundingBox detector_bounding_box(float2 A_proj, float2 B_proj, float2 C_proj)
{
    /* Find the bounding box of a triangles projection onto the detector */

    float  x_min = floorf(min3(A_proj.x, B_proj.x, C_proj.x));
    float  x_max = ceilf(max3(A_proj.x, B_proj.x, C_proj.x));
    float  y_min = floorf(min3(A_proj.y, B_proj.y, C_proj.y));
    float  y_max = ceilf(max3(A_proj.y, B_proj.y, C_proj.y));

    return DetectorBoundingBox{static_cast<int>(x_min), static_cast<int>(x_max), static_cast<int>(y_min), static_cast<int>(y_max)};
}

__device__ bool is_top_left_edge(const float2 &e, float n)
{
    if(n < 0.f)    // CLOCKWISE
        return (e.y > 0.f) || ((e.y == 0.f) && (e.x > 0.f));
    else            // COUNTERCLOCKWISE
        return (e.y < 0.f) || ((e.y == 0.f) && (e.x < 0.f));
}

__device__ bool point_on_same_side(const float2 &start, const float2 &end, const float2 &pixel_pos, float n)
{
    float t1 = cross_z(end - start, pixel_pos - start) *   n ;  // Test edge itself
    float t2 = cross_z(start - end, pixel_pos -   end) *   n ;  // Test shared edge for the triangle facing the same side
    // float t3 = cross_z(start - end, pixel_pos -   end) * (-n);
    float t3 = -t2;                                             // Test shared edge for the triangle facing opposite side

    if((t1 == 0.f) || ((t1>=0.f) == (t2>=0.f)) || ((t1>=0.f) != (t3>=0.f))) // Edge hit - can't rely on the sign test
    {
        if(is_top_left_edge(end - start, n))    // Top-left rule
            return true;
        else
            return false;
    }
    else if(t1 > 0.f)
        return true;
    else
        return false;
}

__device__ bool point_in_triangle_2d(float2 A_proj, float2 B_proj, float2 C_proj, float2 det_pos)
{
    // TODO: normal "vector" has only z-coordinate 1 or -1, optimize by changing to bool
    float n = cross_z(B_proj-A_proj, C_proj-A_proj) > 0.f? 1.f : -1.f;

    return (point_on_same_side(A_proj, B_proj, det_pos, n) &&
            point_on_same_side(B_proj, C_proj, det_pos, n) &&
            point_on_same_side(C_proj, A_proj, det_pos, n));
}

__device__ float get_sign(float value)
{
    // TODO: use signbit(float a)
    return value >= 0.0f? 1.f : -1.f;
}

// TODO(pavel): use Face and ProjectionView structs
template<typename T_3DIndex>
__device__ void project_face(float3 A, float3 B, float3 C, float3 normal,
                             float3 source,
                             float3 det_center, float3 det_u, float3 det_v,
                             size_t det_rows, size_t det_cols, size_t nViews,
                             size_t view_index,
                             float *sino,
                             float  attenuation)
{
    float3 det_topleft_corner = det_center - det_u * (det_cols*0.5f) - det_v * (det_rows*0.5f);
    float t_proj_A, t_proj_B, t_proj_C;
    float2 A_proj = project_point_cone(A, source, det_topleft_corner, det_u, det_v, t_proj_A);  // x_A, y_A
    float2 B_proj = project_point_cone(B, source, det_topleft_corner, det_u, det_v, t_proj_B);  // x_B, y_B
    float2 C_proj = project_point_cone(C, source, det_topleft_corner, det_u, det_v, t_proj_C);  // x_C, y_C

    // projection sanity check: projection backwards is not allowed!
    if(t_proj_A <0 || t_proj_B < 0 || t_proj_C < 0)
        return;

    DetectorBoundingBox dbb = detector_bounding_box(A_proj, B_proj, C_proj);

    T_3DIndex sino_index(nViews, det_rows, det_cols);

    float3 n_pixel = normalize(cross(det_u, det_v));

    // TODO: substitude with parallel code (implement in cuda kernel)
    for(int i = max(int(floorf(dbb.ymin)), 0); i < min(int(floorf(dbb.ymax)), int(det_rows)); i++)
    {
        // int y = i - int(det_rows)/2;
        // int i = y + det_rows/2;
        // if(i < 0 || i >= det_rows) continue;

        for(int j = max(int(floorf(dbb.xmin)), 0); j < min(int(floorf(dbb.xmax)), int(det_cols)); j++)
        {
            // int x = j - int(det_cols)/2;
            // int j = x + det_cols/2;
            // if(j < 0 || j >= det_cols) continue;

            // u - along columns, v - along raws
            float3 pixel = det_topleft_corner + scale(j+0.5f, det_u) + scale(i+0.5f, det_v);
            float3 v = pixel - source;
            v = normalize(v);
            float2 det_pos = scale(j+0.5f, float2{1.f, 0.f}) + scale(i+0.5f, float2{0.f, 1.f});

            if(point_in_triangle_2d(A_proj, B_proj, C_proj, det_pos))
            {
                float n_dot_v = dot(normal, v);
                float t = dot(normal, A - source) / n_dot_v;

                float t_pixel = dot(n_pixel, pixel - source) / dot(n_pixel, v);

                // Projection sanity check: can't project a mesh that is behind either source or detector 
                if(t >=0 && t <= t_pixel)
                {
                    // Here, we assume that the sino value is 0
                    size_t SinoIndex = sino_index(view_index, i, j);
                    atomicAdd(&sino[SinoIndex], get_sign(n_dot_v) * (t * attenuation)); // TODO(tim): use atomicExch instead?
                }
            }
        }
    }
}

/**
 * @brief CUDA kernel that projects mesh with specified geometry.
 * 
 * @tparam GeometryPolicy 
 * @tparam MeshRepresentation 
 * @tparam T_3DIndex 
 * @param pg 
 * @param mesh 
 * @param sino 
 * @param attenuation 
 */
template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
         >
__global__ void project_mesh_kernel(CProjectionGeometry pg,
                                    CMesh mesh,
                                    float *sino)
{
    size_t view_index = blockIdx.y * blockDim.y + threadIdx.y;
    size_t face_index  = blockIdx.x * blockDim.x + threadIdx.x; // same as normal_index

    size_t view_stride = blockDim.y * gridDim.y;
    size_t face_stride = blockDim.x * gridDim.x;

    for(size_t i_view = view_index; i_view < pg.nViews; i_view +=view_stride)
    {
        ProjectionView pv;
        GeometryPolicy<T_3DIndex>::get_projection_view(pg, i_view, pv);
        for(size_t j_face = face_index; j_face < mesh.nFaces; j_face +=face_stride)
        {
            Face face;
            CMeshRepresentation<T_3DIndex>::get_face(mesh, j_face, face);
            project_face<T_3DIndex>(face.A, face.B, face.C, face.normal,
                                    pv.source, pv.det_center, pv.det_u, pv.det_v,
                                    pg.det_row_count, pg.det_col_count, pg.nViews,
                                    i_view, sino, mesh.attenuation);
        }
    }
}