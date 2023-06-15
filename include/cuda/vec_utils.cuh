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

// TODO: check if all of these small functions are inlined
__forceinline__ __device__ float dot(const float2 &a, const float2 &b)
{
    return a.x*b.x + a.y*b.y;
}

__forceinline__ __device__ float dot(const float3 &a, const float3 &b)
{
    // TODO: check if fmaf call is more effective
    // return a.x*b.x + a.y*b.y + a.z*b.z;
    return fmaf(a.x,b.x, fmaf(a.y,b.y, a.z*b.z));
}

__forceinline__ __device__ float3 cross(const float3 &a, const float3 &b)
{
    float3 product;
    // product.x = a.y*b.z - a.z * b.y;
    // product.y = a.z*b.x - a.x * b.z;
    // product.z = a.x*b.y - a.y * b.x;
    // TODO: remove fmaf implementation if it doesn't give better berformance
    product.x = fmaf(a.y, b.z, -(a.z * b.y));
    product.y = fmaf(a.z, b.x, -(a.x * b.z));
    product.z = fmaf(a.x, b.y, -(a.y * b.x));

    return product;
}

__forceinline__ __device__ float cross_z(const float2 &a, const float2 &b)
{
    // TODO: test fmaf more
    // return a.x*b.y - a.y * b.x;
    return fmaf(a.x, b.y, -(a.y*b.x));
}

__forceinline__ __device__ float norm(const float2 &v)
{
    return sqrtf(v.x*v.x + v.y*v.y);
}

__forceinline__ __device__ float norm(const float3 &v)
{
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__forceinline__ __device__ float3 normalize(const float3 &v)
{
    float vnorm = norm(v);
    return {v.x / vnorm, v.y / vnorm, v.z / vnorm};
}

__forceinline__ __device__ void normalize_inplace(float3 &v)
{
    float vnorm = norm(v);
    v.x /= vnorm; v.y /= vnorm; v.z /= vnorm;
}

__forceinline__ __device__ void scale_inplace(float s, float3 &v)
{
    v.x *= s; v.y *= s; v.z *= s;
}

__forceinline__ __device__ float3 scale(float s, const float3 &v)
{
    return {v.x * s, v.y * s, v.z * s};
}

__forceinline__ __device__ float2 scale(float s, const float2 &v)
{
    return {v.x * s, v.y * s};
}

__forceinline__ __device__ float3 operator+(const float3 &lhs, const float3 &rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

__forceinline__ __device__ float3 operator-(const float3 &lhs, const float3 &rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

__forceinline__ __device__ float3 operator-(const float3 &point)
{
    return {-point.x, -point.y, -point.z};
}

__forceinline__ __device__ float2 operator+(const float2 &lhs, const float2 &rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}

__forceinline__ __device__ float2 operator-(const float2 &lhs, const float2 &rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}

__forceinline__ __device__ float2 operator-(const float2 &v)
{
    return {-v.x, -v.y};
}

__forceinline__ __device__ float3 operator*(const float3 &lhs, float rhs)
{
    return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

__forceinline__ __device__ float3 operator*(float lhs, const float3 &rhs)
{
    return {rhs.x * lhs, rhs.y * lhs, rhs.z * lhs};
}

__forceinline__ __device__ bool   operator!=(const float3 &lhs, const float3 &rhs)
{
    return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z);
}

__forceinline__ __device__ float3 switch_basis(   const float3 &point,
                                                  const float3 &e1,
                                                  const float3 &e2,
                                                  const float3 &e3)
{
    // TODO: precompute trasnformations
    // TODO: consider more computationally effective version on https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
    float detA = dot(e1, cross(e2, e3));

    // Inverse of the transformation matrix:
    const float inv_detA = (1/detA);
    float3 T_0 = inv_detA * cross(e2, e3);
    float3 T_1 = inv_detA * cross(e3, e1);
    float3 T_2 = inv_detA * cross(e1, e2);

    return {dot(T_0, point), dot(T_1, point), dot(T_2, point)};
}

/**
 * @brief Project a point onto the detector plane along the specifind direction.
 * The detector is decribed by its top-left corner and 2 basis vectors.
 * The projected point is returned as coefficients in this basis (i. e. in detector pixel coordinates).
 * 
 * @param[in] point A 3D point to project.
 * @param[in] direction Direction of the projection.
 * @param[in] det_topleft Position of top-left corner of detector.
 * @param[in] det_u The vector from detector pixel (0,0) to (0,1).
 * @param[in] det_v The vector from detector pixel (0,0) to (1,0).
 * @param[in,out] t Distance parameter along the source - point line, with origin at point itself.
 * @return float2 Projected point in detector coordinates, i.e., using det_u and det_v as basis vectors.
 */
__forceinline__ __device__ float2 project_point(float3 point,
                                                float3 direction,
                                                float3 det_topleft,
                                                float3 det_u,
                                                float3 det_v,
                                                float &t,
                                                float3 *Q_out=nullptr)
{
    float3 n = cross(det_u, det_v);
    // Here, division by zero is possible when the source ray lays in the detector:
    t = dot(n, det_topleft - point)/dot(n, direction);
    float3 Q = point + scale(t, direction);

    if(Q_out != nullptr)
        *Q_out = Q;

    // Proof of concept implementation!
    // TODO: implement inverse of the transformation matrix in construction of the projection geometry

    // float detA = dot(det_u, cross(det_v, n));

    // // Inverse of the transformation matrix:
    // float3 T_0 = (1/detA) * cross(det_v, n);
    // float3 T_1 = (1/detA) * cross(n, det_u);
    // float3 T_2 = (1/detA) * cross(det_u, det_v);

    float3 Q_uvn = switch_basis(Q-det_topleft, det_u, det_v, n);

    return {Q_uvn.x, Q_uvn.y};

    // TODO: precompute detector size, i.e., dot(det_<axis>, det_<axis>)
    // return float2{dot(det_u, Q-det_topleft)/dot(det_u,det_u), dot(det_v, Q-det_topleft)/dot(det_v, det_v)};
}

/**
 * @brief Project a point onto the detector plane with cone-beam geometry.
 * The detector is decribed by its top-left corner and 2 basis vectors.
 * The projected point is returned as coefficients in this basis (i. e. in detector pixel coordinates).
 * 
 * @param[in] point A 3D point to project.
 * @param[in] source 3D position of a cone beam point source.
 * @param[in] det_topleft Position of top-left corner of detector.
 * @param[in] det_u The vector from detector pixel (0,0) to (0,1).
 * @param[in] det_v The vector from detector pixel (0,0) to (1,0).
 * @param[in,out] t Distance parameter along the source - point line, with origin at point itself.
 * @return float2 Projected point in detector coordinates, i.e., using det_u and det_v as basis vectors.
 */
__forceinline__ __device__ float2 project_point_cone(float3 point,
                                                     float3 source,
                                                     float3 det_topleft,
                                                     float3 det_u,
                                                     float3 det_v,
                                                     float &t)
{
    // TODO: precompute normal vector
    return project_point(point, point-source, det_topleft, det_u, det_v, t);
}

template <typename IntegerType>
__forceinline__ __host__ __device__ IntegerType roundUp(IntegerType x, IntegerType y)
{
    // Taken from OptiX 7 SDK
    return ( ( x + y - 1 ) / y ) * y;
}