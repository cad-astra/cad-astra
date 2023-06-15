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

#include <cuda_runtime_api.h>

#include "array_index.cuh"

#include "../CProjector.h"
#include "ProjectionGeometryPolicies.cuh"
#include "MeshRepresentationPolicies.cuh"
#include "../typelist.h"
#include "../CClassInfo.hpp"

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
class CRasterProjector : public CProjector, public CClassInfo<CRasterProjector<GeometryPolicy, T_3DIndex>>
{
public:
    __host__ CRasterProjector();
    __host__ void set_mesh(const std::vector<CMesh> &mesh_vec) override;
    __host__ void set_outbuffers(ProjectorOutBuffers buffs) override;
    __host__ void run() override;
    static std::string type;
};

typedef CRasterProjector<CRotationGeometryPolicy,    CIndexer3D_C> CRasterRotationCIndex;
typedef CRasterProjector<CRotationGeometryPolicy,    CIndexer3D_F> CRasterRotationFortranIndex;
typedef CRasterProjector<CVecGeometryPolicy,         CIndexer3D_C> CRasterVecCIndex;
typedef CRasterProjector<CVecGeometryPolicy,         CIndexer3D_F> CRasterVecFortranIndex;

template class CRasterProjector<CRotationGeometryPolicy, CIndexer3D_C>;
template class CRasterProjector<CRotationGeometryPolicy, CIndexer3D_F>;
template class CRasterProjector<CVecGeometryPolicy,      CIndexer3D_C>;
template class CRasterProjector<CVecGeometryPolicy,      CIndexer3D_F>;

typedef TYPELIST_4 (CRasterRotationCIndex,
                    CRasterRotationFortranIndex,
                    CRasterVecCIndex,
                    CRasterVecFortranIndex) RasterProjectors;