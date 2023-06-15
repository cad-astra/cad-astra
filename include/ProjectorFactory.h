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

#include "CObjectFactory.h"
#include "CProjector.h"

#include "cuda/CRasterProjector.cuh"

#ifdef WITH_OPTIX_PROJECTOR
    #include "cuda/COptiXProjector.cuh"
    // #include "cuda/COptiXTransmissionProjector.cuh"
    typedef OptiXRecursiveProjectors OptiXProjectors;
    typedef Append<RasterProjectors, OptiXProjectors> ::Result Projectors;
#else
    typedef Append<RasterProjectors, NullType>::Result Projectors;
#endif


class CProjectorFactory : public CObjectFactory<CProjector>
{
public:
    CProjectorFactory(const std::string &type)
        : m_type_tocreate(type)
    {}
    CProjector *create_object() override
    {
        return create<CProjector, Projectors>(m_type_tocreate);
    }
private:
    std::string m_type_tocreate;
};