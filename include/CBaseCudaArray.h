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

#include <string>
#include "CBaseObject.h"

class CBaseCudaArray : public CBaseObject
{
public:
    // enum DType {INT, FLOAT, DOUBLE};
    
    CBaseCudaArray(size_t size) : CBaseObject(CBaseObject::CudaArray), m_size(size) {}

    virtual std::string dtype_str() const = 0;
    
    size_t size() const {return m_size;}

    virtual ~CBaseCudaArray()
    {
        // std::cerr << "Deleted CBaseCudaArray" << std::endl;
    }
protected:
    size_t m_size;
};