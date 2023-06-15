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

// #include "../include/optix_projector_data_structs.h"
#include "../include/cuda/vec_utils.cuh"
// #include "../include/cuda/phys_utils.cuh"
#include "../include/cuda/array_index.cuh"

#include "../include/cuda/ProjectionGeometryPolicies.cuh"
#include "../include/optix_projector_data_structs.h"

/*
 * DIRECT CALLABLES that implement different projector policies
*/

// TODO(pavel): add implementation with Fortran order policies

// Implementation of AtomicMax for float.
// See https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu
__device__ void AtomicMaxFlt(float * const address, const float value)
{
	if (* address >= value)
	{
		return;
	}

	int * const address_as_i = (int *)address;
	int old = * address_as_i, assumed;

	do 
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}

extern "C" __device__ void __direct_callable__rotation_geom_policy(const CProjectionGeometry &pg,
                                                                   uint32_t i_view,
                                                                   ProjectionView &pv)
{
    CRotationGeometryPolicy<CIndexer3D_C>::get_projection_view(pg, i_view, pv);
}

extern "C" __device__ void __direct_callable__vector_geom_policy(const CProjectionGeometry &pg,
                                                                 uint32_t i_view,
                                                                 ProjectionView &pv)
{
    CVecGeometryPolicy<CIndexer3D_C>::get_projection_view(pg, i_view, pv);
}

extern "C" __device__ void __direct_callable__mesh_transform_geom_policy(const CProjectionGeometry &pg,
                                                                         uint32_t i_view,
                                                                         ProjectionView &pv)
{
    CMeshTransformGeometryPolicy<CIndexer3D_C>::get_projection_view(pg, i_view, pv);
}

// Conventional X-ray CT detector policy - ray attenuation stored in detector pixel
extern "C" __device__ void __direct_callable__max_attenuation_detector_pixel_policy
(
    const float2              &Q_projected,
    const CProjectionGeometry &proj_geom,
    uint32_t                   iView,
    uint32_t                   nViews,
    float                      attenuation,
    float                     *sinogram
)
{
    const CIndexer3D_C det_index(nViews, proj_geom.det_row_count, proj_geom.det_col_count);
    int i = int(floorf(Q_projected.y));
    int j = int(floorf(Q_projected.x));
    if(i >= 0 && i < static_cast<int>(proj_geom.det_row_count) && j >= 0 && j < static_cast<int>(proj_geom.det_col_count))
    {
        // max(x1, ..., xn) approximates logsumexp(x1, ..., xn)=log(exp(x1) + ... + exp(xn)).
        // logsumexp(x1, ..., xn)=x* + log(exp(x1-x*) + ... + exp(xn-x*)), where x* = max(x1, ..., xn).
        uint32_t sinoIndex = det_index(iView, i, j);
        AtomicMaxFlt(&sinogram[sinoIndex], attenuation);
    }
}

// Edge illumination phase contrast CT detector policy - rays intensity stored in detector pixel
extern "C" __device__ void __direct_callable__sum_intensity_detector_pixel_policy
(
    const float2              &Q_projected,
    const CProjectionGeometry &proj_geom,
    uint32_t                   iView,
    uint32_t                   nViews,
    float                      attenuation,
    float*                     sinogram
)
{
    // const float epsilon = 1e-16;
    const CIndexer3D_C det_index(nViews, proj_geom.det_row_count, proj_geom.det_col_count);
    int i = int(floorf(Q_projected.y));
    int j = int(floorf(Q_projected.x));
    if(i >= 0 && i < static_cast<int>(proj_geom.det_row_count) &&
       j >= 0 && j < static_cast<int>(proj_geom.det_col_count))
    //    fabsf(float(i)-Q_projected.y) > epsilon &&
    //    fabsf(float(j)-Q_projected.x) > epsilon)
    {
        uint32_t sinoIndex = det_index(iView, i, j);
        atomicAdd(&sinogram[sinoIndex], expf(-attenuation));
    }
}

__device__ float bilinear(const float2 &P, const float2 &O, const float2 &O_diag)
{
    // Function value at O is assumed to be 1.0, and 0.0 at O_diag
    return (O_diag.x - P.x) * (O_diag.y - P.y) / ( (O_diag.x - O.x) * (O_diag.y - O.y) );
}

__device__ float bilinear_unit_pixel(const float2 &P, const float2 &O, const float2 &O_diag)
{
    // Same as bilinear, but with assumption that pixel size is 1.0.
    // In this case, we're interested only in the denominator sign.
    // Thus, multiplication can be used as a faster version.

    // TODO: Simplify this kernel by computing the weight to the diagonal pixel and
    // removing the denominator which is equal to 1.
    return (O_diag.x - P.x) * (O_diag.y - P.y) * ( (O_diag.x - O.x) * (O_diag.y - O.y) );
}

extern "C" __device__ void __direct_callable__sum_share_intensity_detector_pixel_policy
(
    const float2              &Q_projected,
    const CProjectionGeometry &proj_geom,
    uint32_t                   iView,
    uint32_t                   nViews,
    // uint32_t       det_rows,
    // uint32_t       det_cols,
    // const CIndexer3D_C &det_index,
    float                      attenuation,
    float*                     sinogram
)
{
    const CIndexer3D_C det_index(nViews, proj_geom.det_row_count, proj_geom.det_col_count);
    int i = int(floorf(Q_projected.y));
    int j = int(floorf(Q_projected.x));
    float2 det_pos = scale(j+0.5f, float2{1.f, 0.f}) + scale(i+0.5f, float2{0.f, 1.f});
 
    // We share ray intensity between three neighbouring pixels.
    // The contribution has weight found as a bilinear interpolation of a weight function,
    // that is 1.0 at det_pos, and 0.0 at the centers of all three neighbours.
    
    int i_next = Q_projected.y - det_pos.y >= 0.0f ? i + 1 : i -1;
    int j_next = Q_projected.x - det_pos.x >= 0.0f ? j + 1 : j -1;

    int i_ind[] = {i, i_next}; int i_diag_ind[] = {i_next, i};
    int j_ind[] = {j, j_next}; int j_diag_ind[] = {j_next, j};
    for(int q=0; q<2; q++)
    {
        for(int r=0; r<2; r++)
        {
            if(i_ind[r] >= 0 && i_ind[r] < static_cast<int>(proj_geom.det_row_count) && j_ind[q] >= 0 && j_ind[q] < static_cast<int>(proj_geom.det_col_count))
            {
                float2 pixel      = scale(j_ind[q]+0.5f, float2{1.f, 0.f}) + scale(i_ind[r]+0.5f, float2{0.f, 1.f});
                float2 pixel_diag = scale(j_diag_ind[q]+0.5f, float2{1.f, 0.f}) + scale(i_diag_ind[r]+0.5f, float2{0.f, 1.f});
            
                float weight = bilinear_unit_pixel(Q_projected, pixel, pixel_diag);

                int32_t sinoIndex = det_index(iView, i_ind[r], j_ind[q]);
                atomicAdd(&sinogram[sinoIndex], expf(-attenuation) * weight);
            }
        }
    }
}