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

#include "vec_utils.cuh"

#include <cuda_runtime_api.h>

/**
 * @brief Find transmitted ray direction using Snell's law.
 * 
 * @param[in]  iDirection Unit vector of incident ray direction.
 * @param[in]  normal Unit surface normal vector.
 * @param[in]  eta Rafractive index ratio n1/n2, where n1 is the refractive index of incident ray media.
 * @param[out] tDirection Not normalized transmitted ray direction.
 * @return bool true if ray was refracted, fasle in case of total internal reflection. 
 */
__device__ bool snellsLaw(const float3 &iDirection, float3 normal, float eta, float3 &tDirection)
{
    float NdotI = dot(normal, iDirection);
    if (NdotI < 0)
    {
        NdotI = -NdotI;
    }
    else
    {
        normal = -normal;
    }
    float cos_incid_ang = NdotI;
    float c_1 = NdotI;
    float k = 1 - eta * eta * (1 - cos_incid_ang * cos_incid_ang);
    if (k < 0)
    {        
        return false;
    } else {
        float c_2 = sqrtf(k);
        tDirection = iDirection * eta + (eta * c_1 - c_2) * normal;
        // normalize_inplace(tDirection);
        return true;
    }
}

/**
 * @brief Compute refracted or totally internally reflected ray direction.
 * 
 * @param[in,out] ray_direction Incident ray direction. Updated not normalized ray direction is also stored in this variable.
 * @param[in] norm_vec Normal vector assumed to be normalized.
 * @param[in] eta Rafractive index ratio n1/n2. It is assumed that n1 is the refractive index of incident ray media.
 * @return bool true if ray was refracted, fasle in case of total internal reflection. 
 */
__device__ bool bend_ray(float3 &ray_direction, const float3 &norm_vec, float eta)
{
    float3 iDirection = normalize(ray_direction);
    float3 tDirection;
    bool refracted = snellsLaw(iDirection, norm_vec, eta, tDirection);
    if(refracted)
        {ray_direction = tDirection;}
    else    // Total interal reflection
    {
        ray_direction = iDirection - 2*(dot(norm_vec, iDirection) * norm_vec);
    }
    return refracted;
}