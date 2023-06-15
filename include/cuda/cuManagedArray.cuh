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

#include "../CBaseCudaArray.h"
#include "../CClassInfo.hpp"

template<typename T>
std::string dtype_2str();

template<>
std::string dtype_2str<int>();
template<>
std::string dtype_2str<float>();

template<typename T>
class cuManagedArray : public CBaseCudaArray, public CClassInfo<cuManagedArray<T>>
{
public:
    __host__ cuManagedArray(size_t size, T init_value=0);
    __host__ cuManagedArray(size_t size, T *in_array, bool external_allocation=false);
    __host__ ~cuManagedArray() override;
    __host__ std::string dtype_str() const override {return dtype_2str<T>();}
    __host__ void initialize(T value);
    __host__ void assignScalar(T value);
    __host__ void assignArray(T* value, size_t NumElements);
    __host__ void uploadToDevice(T *in_array, size_t N);
    __host__ void downloadToHost(T *out_array, size_t N);
    __host__ T *dataPtr() {return m_dataPtr;}
private:
    T *m_dataPtr;
    bool m_external_allocation;

    __host__ void allocate();
    __host__ void free();
};

template class cuManagedArray<float>;
template class cuManagedArray<int>;

typedef cuManagedArray<float> cuManagedArrayFloat;
typedef cuManagedArray<int>   cuManagedArrayInt;

#include "../CObjectFactory.h"

// TODO(pavel): make input data pointer const
template<typename T>
class cuManagedArrayFactory : public CObjectFactory<CBaseCudaArray>, public CClassInfo<cuManagedArrayFactory<T>>
{
public:\
    cuManagedArrayFactory(size_t N, T init_value=T())
        : m_data_ptr(nullptr), m_data_size(N), m_init_value(init_value)
    {}
    cuManagedArrayFactory(size_t N, T *pData, bool m_external_allocation=false)
        : m_data_ptr(pData), m_data_size(N), m_external_allocation(m_external_allocation)
    {}
    CBaseCudaArray *create_object() override;
private:
    T *m_data_ptr;
    size_t m_data_size;
    T m_init_value;
    bool m_external_allocation;
};

template class cuManagedArrayFactory<float>;
template class cuManagedArrayFactory<int>;

typedef cuManagedArrayFactory<float> cuManagedArrayFloatFactory;
typedef cuManagedArrayFactory<int>   cuManagedArrayIntFactory;
