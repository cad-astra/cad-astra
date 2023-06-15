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

#ifndef OPTIX_PROJECTOR_DATA_STRUCTS_H
#define OPTIX_PROJECTOR_DATA_STRUCTS_H

#include <stdint.h>
#include <cuda_runtime_api.h>

#include "../include/cuda/optix_device_functions.cuh"
#include "../include/cuda/array_index.cuh"
#include "../include/proj_geometry.h"

// enum PG_IDX {
//     PG_RAYGEN=0,
//     PG_MISS=1,
//     PG_HIT=2,
//     PG_DET_ATTEN_CALLABLE =3,
//     PG_DET_INTENS_CALLABLE=4,
//     PG_GEOM_ROTATION_CALLABLE=5,
//     PG_GEOM_VECTOR_CALLABLE=6
// };

// These struct represent the data blocks of the SBT records
// We pass material properties via SBT record into the corresponding optix program 
struct RayGenSBTData
{
    float media_refractive_index;
};

struct MissData     { int some_data; /*float3 bg_color; */ };

// TODO(pavel): see if this imlementation fits sbt record per mesh approach:
struct HitSBTData
{
    float material_attenuation;
    float material_refractive_index;
    uint32_t max_depth;
    // float *vertices;
};

struct RayPathData
{
    float *paths_ptr;       // Should be (nRays x ray_maxdepth x 3)
    int    length;

    uint32_t     ray_index_linear;
    CIndexer3D_C ray_vertex_coord_index;

    Ray ray_at_endpoint;
};

struct COptixLaunchParams
{
    float*                 sinogram;

    CProjectionGeometry   *proj_geom_devptr;

    OptixTraversableHandle handle; // Handle of an AS for passing into optixTrace
    
    uint32_t  ray_maxdepth;
    uint32_t *pMax_depth_rays_cnt; // Number of rays that reached maximal trace depth
    uint32_t *pMax_actual_depth;   // Number of rays that reached maximal trace depth
    
    float *ray_paths;       // Should be (nRays x ray_maxdepth x 3)
    int   *trace_lengths;   // Should be nRays
    float *ray_intensities; // Should be nRays

    // Policy indices
    uint32_t geometry_policy_index;     // Rotation / Vector geometry
    uint32_t detector_policy_index;     // Attenuation / Intencity domain
    uint32_t det_geometry_policy_index;
    // uint32_t indexing_policy_index;    // C / Fortran order
};
#endif