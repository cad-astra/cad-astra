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

#include <vector>
#include <string>

#include <optix.h>
#include <cuda_runtime.h>

#include "../mesh.h"

namespace mesh_fp
{
    /**
     * @brief Build GAS from a set of meshes.
     * This function allocates memory for the GAS output buffer, then stores the AS data there.
     * 
     * @param mesh_vec Vector with CMesh objects in host memory
     * @param gas_output_buffer_devptr Pointer to the output buffer with the AS. Function allocates the memory for the buffer.
     * @param optix_context Current OptiX context.
     * @return OptixTraversableHandle Handle of a GAS traversable
     */
    __host__ OptixTraversableHandle build_gas(OptixDeviceContext &optix_context, const std::vector<CMesh> &mesh_vec, CUdeviceptr &gas_output_buffer_devptr);
    /**
     * @brief Build IAS. This function allocates memory for the IAS output buffer, then stores the AS data there.
     * 
     * @param optix_context Current OptiX context.
     * @param static_transforms_vec Vector of static transformations for every child. Every static transformation is a 12-element vector that represents 3x4 transformation matrix.
     * @param children_traversable_vec Vector of child traversable handles.
     * @param ias_output_buffer_devptr Pointer to the output buffer with the AS. Function allocates the memory for the buffer.
     * @return OptixTraversableHandle Handle of an IAS traversable
     */
    __host__ OptixTraversableHandle build_ias(OptixDeviceContext &optix_context, const std::vector<std::vector<float>> static_transforms_vec, const std::vector<OptixTraversableHandle> &children_traversable_vec, CUdeviceptr &ias_output_buffer_devptr);
    /**
     * @brief Build a motion traversable from an SRT transformation.
     * End time is equal to the SRT vector size, i.e., one motion key per projection view.
     * 
     * @param optix_context Current OptiX context.
     * @param srt_data_vec Vector with SRT data, one element per motion key. See OptixSRTData on SRT data structure.
     * @param child_traversable Traversable transformed by this transformation.
     * @param motion_transform_devptr Pointer to the device memory where traversable data will be stored.
     * @return OptixTraversableHandle Handle of a traversable 
     */
    __host__ OptixTraversableHandle build_srt_transformation_traversable(OptixDeviceContext &optix_context, const std::vector<OptixSRTData> &srt_data_vec, int begin_time, int end_time, OptixTraversableHandle child_traversable, CUdeviceptr motion_transform_devptr);
    /**
     * @brief Create a program group.
     * 
     * @param optix_context Current OptiX context. 
     * @param kind Program group kind. Can be one of the following: OPTIX_PROGRAM_GROUP_KIND_RAYGEN, OPTIX_PROGRAM_GROUP_KIND_MISS, OPTIX_PROGRAM_GROUP_KIND_HITGROUP, OPTIX_PROGRAM_GROUP_KIND_CALLABLES.
     * @param module OptiX module that contains the programs.
     * @param entry_function_name_vec Names of the OptiX programs. Every name should begin with either "__raygen__", or "__miss__", or "__closesthit__", or "__direct_callable__", or "__continuation_callable__".
     * @param log_str Log string
     * @return std::vector<OptixProgramGroup> Vector of program group objects.
     */
    __host__ std::vector<OptixProgramGroup> create_program_group(OptixDeviceContext &optix_context,
                                                                 OptixProgramGroupKind kind,
                                                                 OptixModule module,
                                                                 const std::vector<std::string> &entry_function_name_vec,
                                                                 std::string &log_str);
    /**
     * @brief Read PTX code from the file.
     * Returns an empty string in case failed to open/read PTX file.
     * 
     * @param filename PTX file name.
     * @return std::string PTX code
     */
    __host__ std::string read_ptx(const std::string& filename);
}