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

#include <cuda_runtime.h>

#ifndef MAX_OPTIX_MATERIAL_STACK_SIZE
#define MAX_OPTIX_MATERIAL_STACK_SIZE 16
#endif

enum LocalMemStackOperationFailCode
{
    STACK_OPERATION_SUCCESS=0,
    STACK_PUSH_FAILED=1,
    STACK_POP_FAILED =2,
    STACK_HEAD_FAILED=3,
    STACK_REMOVE_LAST_FAILED=4
};

// TODO(pavel): throw OptiX Exception instead of printout
// DEBUG_PRINTLN("Stack operation %s failed at %s: %d", #operation_result, __FILE__, __LINE__);
// #define STACK_OPERATION_CHECK( call )                                                           \
//     do                                                                                          \
//     {                                                                                           \
//         LocalMemStackOperationFailCode result =  call;                                          \
//         if (result != STACK_OPERATION_SUCCESS)                                                  \
//         {                                                                                       \
//             optixThrowException(STACK_PUSH_FAILED);                                                        \
//         }                                                                                       \
//     } while (0)

template<typename DataType, uint32_t MaxSize=MAX_OPTIX_MATERIAL_STACK_SIZE>
class LocalMemStack
{
// TODO: try __forceinline__ instruction?
// All the methods are supposed to be inline automatically, though
public:
    __device__  LocalMemStack() : m_size(0) {}
    __device__ bool     empty() const { return m_size == 0; } 
    __device__ uint32_t size () const { return m_size; }
    __device__ uint32_t max_size() const { return MaxSize; }
    __device__ LocalMemStackOperationFailCode head(DataType &data) const
    {
        LocalMemStackOperationFailCode read_success = STACK_HEAD_FAILED; 
        if(m_size > 0)
        {
            data = m_stack_vec[m_size - 1];
            read_success = STACK_OPERATION_SUCCESS;
        }
        return read_success;
    }
    __device__ LocalMemStackOperationFailCode remove_last()
    {
        LocalMemStackOperationFailCode removed = STACK_REMOVE_LAST_FAILED;
        if(m_size > 0)
        {
            m_size--;
            removed = STACK_OPERATION_SUCCESS;
        }
        return removed;
    }    
    __device__ LocalMemStackOperationFailCode push(const DataType &data)
    {
        LocalMemStackOperationFailCode pushed = STACK_PUSH_FAILED;
        if(m_size < MaxSize)
        {
            m_stack_vec[m_size] = data;
            m_size++;
            pushed = STACK_OPERATION_SUCCESS;
        }
        return pushed;
    }
    __device__ LocalMemStackOperationFailCode pop(DataType &data)
    {
        LocalMemStackOperationFailCode popped = STACK_POP_FAILED;
        if(m_size > 0)
        {
            data = m_stack_vec[m_size - 1];
            m_size--;
            popped = STACK_OPERATION_SUCCESS;
        }
        return popped;
    }
    __device__ void clean()
    {
        m_size = 0;
    }
private:
    DataType m_stack_vec[MaxSize];
    uint32_t m_size;
};