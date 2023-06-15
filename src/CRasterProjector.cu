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

#include "../include/cuda/CRasterProjector.cuh"
#include "../include/cuda/raster_project_mesh.cuh"
#include "../include/cuda/common.cuh"

#include "../include/host_logging.h"

#include <algorithm>

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
CRasterProjector<GeometryPolicy,
                 T_3DIndex>
    ::CRasterProjector()
{}
// --------------------------------------------------------------------------------
template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void CRasterProjector<GeometryPolicy,
                      T_3DIndex>
    ::set_mesh(const std::vector<CMesh> &mesh_vec)
{
    if (mesh_vec.size() > 1)
    {
        m_mesh_set = false;
        throw std::runtime_error("Error: multi-mesh sample projection is not implemented in raster projector yet");
    }
    else
    {
        m_mesh = mesh_vec[0];
        m_mesh_set = true;
    }
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void CRasterProjector<GeometryPolicy, T_3DIndex>
    ::set_outbuffers(ProjectorOutBuffers buffs)
{
    m_pSino_device = buffs.pSino_device;
    m_buffers_set = true;
}
template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void CRasterProjector<GeometryPolicy, T_3DIndex>
    ::run()
{
    if(m_buffers_set && m_mesh_set && m_geometry_set)
    {
        MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "views:   %lu",   m_proj_geom.nViews);
        MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "triangles: %lu", m_mesh.nFaces);

        unsigned int blockSize_x = 8u;
        unsigned int blockSize_y = 16u;// maximal number of threads per block should be 1024
        unsigned int numBlocks_x = std::min<unsigned int>(NUM_BLOCKS(m_mesh.nFaces, blockSize_x), (2u<<31u)-1);
        unsigned int numBlocks_y = std::min<unsigned int>(NUM_BLOCKS(m_proj_geom.nViews, blockSize_y), 65535u);
        dim3 numBlocks(numBlocks_x, numBlocks_y, 1u);
        dim3 threadsPerBlock(blockSize_x, blockSize_y, 1u);

        MeshFP::Logging::info(MESH_FP_CLASS_NAME, "Starting kernel with numBlocks = (%d, %d, %d)", numBlocks.x, numBlocks.y, numBlocks.z);

        // TODO(pavel): since kernel launch can end up with a "sticky" error, consider starting it in a new thread
        project_mesh_kernel<GeometryPolicy, T_3DIndex><<<numBlocks, threadsPerBlock>>>(m_proj_geom, m_mesh, m_pSino_device);
        CUDA_CALL(cudaDeviceSynchronize());
        MeshFP::Logging::info(MESH_FP_CLASS_NAME, "Done");
    }
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
std::string CRasterProjector<GeometryPolicy,  T_3DIndex>::type =
    "raster_" + GeometryPolicy<T_3DIndex>::type + "_" + T_3DIndex::type;
