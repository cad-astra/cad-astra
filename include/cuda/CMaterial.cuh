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

// Material structure
struct CMaterial
{
    __forceinline__ __host__ __device__ CMaterial(float attenuation_coeff=0.f, float refractive_index=1.f, int mesh_index=-1)
        : m_attenuation_coeff(attenuation_coeff), m_refractive_index(refractive_index), m_mesh_sbt_index(mesh_index)
    {}
    float m_attenuation_coeff;
    float m_refractive_index;
    int   m_mesh_sbt_index;
};