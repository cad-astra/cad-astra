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

#include "../include/cuda/cuManagedArray.cuh"

#include <stdexcept>
#include <iostream>

#include "../include/cuda/common.cuh"
#include "../include/host_logging.h"

template<typename T>
std::string dtype_2str()
{
    return std::string("");
}

template<>
std::string dtype_2str<int>()
{
    return std::string("int");
}

template<>
std::string dtype_2str<float>()
{
    return std::string("float");
}

template<typename T>
__global__ void init_kernel(T *array_ptr, uint32_t N, T value)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for(uint32_t i=index; i < N; i += stride)
    {
        array_ptr[i] = value;
    }
}

template<typename T>
__host__ cuManagedArray<T>::cuManagedArray(size_t size, T init_value)
    : m_dataPtr(nullptr), CBaseCudaArray(size)
{
    try
    {
        if(m_size != 0)
        {
            allocate();
            initialize(init_value);
        }
    }
    catch(...)
    {
        throw;  // TODO: add more exception information for simple stack-trace
    }
}

template<typename T>
__host__ cuManagedArray<T>::cuManagedArray(size_t size, T *in_array, bool external_allocation)
    : m_dataPtr(nullptr), CBaseCudaArray(size), m_external_allocation(external_allocation)
{
    try
    {
        if (m_external_allocation == false)
        {
            if(m_size != 0) allocate();
            CUDA_CALL(cudaMemcpy(m_dataPtr, in_array, m_size*sizeof(T), cudaMemcpyHostToDevice));
        } else {
            m_dataPtr = in_array;
        }
    }
    catch(...)
    {
        throw;  // TODO: add more exception information for simple stack-trace
    }
}

template<typename T>
__host__ cuManagedArray<T>::~cuManagedArray()
{
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Destructor");
    try
    {
        if( (m_size != 0) && (m_external_allocation == false) )
        {
            free();
        }
    }
    catch(const std::runtime_error& e)
    {
        // TODO: do proper error logging
        std::cerr << e.what() << '\n';
    }
}

template<typename T>
__host__ void cuManagedArray<T>::initialize(T value)
{
    // TODO: for debug, start <<<1, 1>>> kernel
    uint32_t blockSize = 1024;
    dim3 numBlocks(NUM_BLOCKS(m_size, blockSize), 1u, 1u);
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Starting init. kernel with <<<%d, %d >>>...", numBlocks.x, blockSize);
    init_kernel<<<numBlocks,blockSize>>>(m_dataPtr, m_size, value);
    CUDA_CALL(cudaDeviceSynchronize());
}


// alias for initialize!
template<typename T>
__host__ void cuManagedArray<T>::assignScalar(T value)
{
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "assignScalar: initialize()");
    initialize(value);
}

// an alias for uploadToDevice
template<typename T>
__host__ void cuManagedArray<T>::assignArray(T* pValues, size_t N)
{
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME,"assignArray: uploadToDevice");
    uploadToDevice(pValues, N);

    // if(N != m_size)
    // {
    //     throw std::runtime_error("Error: Number of assigned values needs to match array elements!");
    // }

    // from cuda documentation: cudaMemcpyDefault = 4 - Direction of the transfer is inferred
    // from the pointer values. requires unified virtual addressing
    // CUDA_CALL(cudaMemcpy(m_dataPtr, pValues, N*sizeof(T), cudaMemcpyDefault));
    /*std::cout << "assignArray: writing " << N << " values\n";
    for(size_t Index = 0;
        Index < N;
        Index++)
    {
        m_dataPtr[Index] = pValues[Index];
    }*/
}

template<typename T>
__host__ void cuManagedArray<T>::uploadToDevice(T *in_array, size_t N)
{
    if(N != m_size)
    {
        throw std::runtime_error("Error: N != cuManagedArray.size");
    }

    CUDA_CALL(cudaMemcpy(m_dataPtr, in_array, N*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
__host__ void cuManagedArray<T>::downloadToHost(T *out_array, size_t N)
{
    if(N != m_size)
        throw std::runtime_error("Error: N != cuManagedArray.size");
    CUDA_CALL(cudaMemcpy(out_array, m_dataPtr, N*sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
__host__ void cuManagedArray<T>::allocate()
{
    MeshFP::Logging::info(MESH_FP_CLASS_NAME, "allocating %lu bytes...", m_size*sizeof(T));
    CUDA_CALL(cudaMalloc(&m_dataPtr, m_size*sizeof(T)));
}

template<typename T>
__host__ void cuManagedArray<T>::free()
{
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Calling cudaFree");
    CUDA_CALL(cudaFree(m_dataPtr));
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Done");
}

template<typename T>
CBaseCudaArray *cuManagedArrayFactory<T>::create_object()
{
    try
    {
        if(m_data_ptr != nullptr)
            return new cuManagedArray<T>(m_data_size, m_data_ptr, m_external_allocation);
        else
            return new cuManagedArray<T>(m_data_size, m_init_value);
    }
    catch(...)
    {
        // TODO: add more exception information for stack-trace
        MeshFP::Logging::error(MESH_FP_CLASS_NAME, "create_object() :: unable to create new object!");
        throw;
    }
}
