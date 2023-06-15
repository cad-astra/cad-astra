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

#include <cuda_runtime_api.h>

#include "../include/cuda/common.cuh"

void getDeviceCount(int32_t &device_count)
{
    CUDA_CALL( cudaGetDeviceCount( &device_count ) );
}

void setDevice(int device)
{
    CUDA_CALL( cudaSetDevice( device ) );
}

void getDeviceName(std::string &device_name, int device)
{
    cudaDeviceProp prop;
    CUDA_CALL( cudaGetDeviceProperties ( &prop, device ) );
    device_name = prop.name;
}