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

#include "../include/cuda/common.cuh"

void check(cudaError_t result, char const *const func, const char *const file,
           int const line)
{
    if (result != cudaSuccess)
    {
        std::stringstream err_ss;
        err_ss << "CUDA error at " << file << ":" << line << ", " << func << ": " << cudaGetErrorString(result);
        cudaDeviceReset();  // Make sure we call CUDA Device Reset before exiting
        throw std::runtime_error(err_ss.str());
    }
}