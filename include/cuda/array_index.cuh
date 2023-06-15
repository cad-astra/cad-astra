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

#ifndef ARRAY_INDEX_CUH
#define ARRAY_INDEX_CUH

#include <string>
#include <cuda_runtime.h>

class CIndexer3D
{
public:
    __device__     CIndexer3D(int N0=1, int N1=1, int N2=1) : m_N0(N0), m_N1(N1), m_N2(N2) {}
    __device__ int operator()(int n0, int n1, int n2) { return 0; }   // Dummy function to make PTX JIT compiler happy
protected:
    int m_N0;
    int m_N1;
    int m_N2;
};

class CIndexer3D_C : public CIndexer3D
{
public:
    __forceinline__ __device__   CIndexer3D_C(int N0=1, int N1=1, int N2=1) : CIndexer3D(N0, N1, N2) {}
    __forceinline__ __device__ int operator()(int n0,   int n1,   int n2=0) const
    {
        return n2 + n1*m_N2 + n0 * m_N2 * m_N1;
    }
    static std::string type;
};

class CIndexer3D_F : public CIndexer3D
{
public:
    __forceinline__ __device__   CIndexer3D_F(int N0=1, int N1=1, int N2=1) : CIndexer3D(N0, N1, N2) {}
    __forceinline__ __device__ int operator()(int n0,   int n1,   int n2=0) const
    {
        return n0 + m_N0 * n1 + m_N0 * m_N1 * n2;
    }
    static std::string type;
};

// std::string CIndexer3D_C::type = "c_index";
// std::string CIndexer3D_F::type = "f_index";

#endif