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

#include "cuda/optix_projector_policies.cuh"

struct CProjectorConfiguration
{
    CProjectorConfiguration() :
          rays_row_count(-1),   // -1 for rays_cnt == det_pixels_cnt
          rays_col_count(-1),
          detector_policy_index(OptiXMaxAttenuationPolicy),
          tracer_index(OptiXRecursivePolicy)
    {}
    int rays_row_count;
    int rays_col_count;
    int detector_policy_index;
    OptiXTracerPolicyIndex tracer_index;
};