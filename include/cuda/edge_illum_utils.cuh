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
#include "ProjectionGeometryPolicies.cuh"
#include "../optix_projector_data_structs.h"

__device__ bool ray_hits_mask(const Ray &ray,
                              const ProjectionView &pv,
                              uint32_t det_rows,
                              uint32_t det_cols,
                              float bar_width,
                              float thickness,
                              float mask_det_dist,
                              float mask_det_offset,
                              float &t_front_face,
                              float &t_back_face)
{
    const float3 source_det = pv.source - pv.det_center;
    const float  source_det_dist = norm(source_det);
    const float  inv_M = (source_det_dist-mask_det_dist) / source_det_dist;
    const float3 det_x_scaled = pv.det_x * inv_M;
    const float3 det_x_dir    = normalize(pv.det_x);
    const float3 det_y_scaled = pv.det_y * inv_M;
    const float  bar_width_normed   = bar_width / norm(det_x_scaled);
    const float  bar_width_normed_2 = 0.5f * bar_width_normed;
    const float3 inv_projection_dir = normalize(source_det);

    float3 mask_front_topleft_corner =  pv.det_center +
                                        inv_projection_dir*mask_det_dist +
                                        det_x_dir * mask_det_offset -
                                        det_x_scaled * (det_cols*0.5f) -
                                        det_y_scaled * (det_rows*0.5f);

    float3 mask_back_topleft_corner  =  mask_front_topleft_corner -
                                        inv_projection_dir*thickness;

    // printf("mask front dist %.4f, toplef: (%.4f,%.4f,%.4f)\n", mask_det_dist, mask_front_topleft_corner.x, mask_front_topleft_corner.y, mask_front_topleft_corner.z);

    // Front face hit
    float2 hitpoint_front_face = project_point(ray.origin, ray.direction, mask_front_topleft_corner, det_x_scaled, det_y_scaled, t_front_face);
    float2 hitpoint_back_face  = project_point(ray.origin, ray.direction, mask_back_topleft_corner,  det_x_scaled, det_y_scaled, t_back_face);

    bool front_face_hit = false;
    bool back_face_hit  = false;

    if((hitpoint_front_face.x >= -bar_width_normed_2) && (hitpoint_front_face.x <= det_cols+bar_width_normed_2) && 
       (hitpoint_back_face.x  >= -bar_width_normed_2) && (hitpoint_back_face.x  <= det_cols+bar_width_normed_2))
    {
        front_face_hit =  ((hitpoint_front_face.x - floorf(hitpoint_front_face.x) <= bar_width_normed_2) || 
                                (ceilf(hitpoint_front_face.x) - hitpoint_front_face.x  <= bar_width_normed_2));

        // // bar side hit
        back_face_hit  =  ((hitpoint_back_face.x - floorf(hitpoint_back_face.x) <= bar_width_normed_2) || 
                                (ceilf(hitpoint_back_face.x) - hitpoint_back_face.x  <= bar_width_normed_2));

    }
    bool bar_side_hit = fabsf(hitpoint_back_face.x - hitpoint_front_face.x) > bar_width_normed;

    // if(!(front_face_hit || back_face_hit || bar_side_hit))
    //     printf("(%.4f, %.4f, %.4f), hit_2d.x = %.4f\n",mask_front_topleft_corner.x, mask_front_topleft_corner.y,mask_front_topleft_corner.z, hitpoint_front_face.x);

    return front_face_hit || back_face_hit || bar_side_hit;
}