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

#include "common.cuh"
#include "vec_utils.cuh"

#include "../mesh.h"

#ifdef WITH_OPTIX_PROJECTOR
    #include <optix.h>
#endif

struct Face
{
    float3 A, B, C;
    float3 normal;
};

template<typename T_3DIndex>
struct CMeshRepresentation
{
   __device__ static void get_face(const CMesh &m, uint32_t j_face, Face &face)
    {
        if(m.vertices != nullptr && m.faces != nullptr) // Indexed mesh
        {
            T_3DIndex faces_lin_index(m.nFaces, 3);
            T_3DIndex vertices_lin_index(m.nVertices, 3);

            int3  face_vertices    = UNPACK_3D_POINT(m.faces, faces_lin_index,  j_face, 0);
            face.A = UNPACK_3D_POINT(m.vertices, vertices_lin_index, face_vertices.x, 0);
            face.B = UNPACK_3D_POINT(m.vertices, vertices_lin_index, face_vertices.y, 0);
            face.C = UNPACK_3D_POINT(m.vertices, vertices_lin_index, face_vertices.z, 0);
            if(m.normals != nullptr)
            {
                face.normal  = normalize(UNPACK_3D_POINT(m.normals, faces_lin_index, j_face, 0));
            }
            else
            {
                float3 AB = face.B - face.A;
                float3 AC = face.C - face.A;
                face.normal = normalize(cross(AB, AC));
            }
        }
        else if(m.nVertices > 0)  // Non-indexed mesh
        {
            size_t nFaces = m.nVertices / 9;
            T_3DIndex vertices_lin_index(nFaces, 9);

            face.A = UNPACK_3D_POINT(m.vertices, vertices_lin_index, j_face, 0);
            face.B = UNPACK_3D_POINT(m.vertices, vertices_lin_index, j_face, 1*3);
            face.C = UNPACK_3D_POINT(m.vertices, vertices_lin_index, j_face, 2*3);
            float3 AB = face.B - face.A;
            float3 AC = face.C - face.A;
            face.normal = normalize(cross(AB, AC));
        }
        else
        {
            // TODO(Pavel): implement error code return since exception handling is not supported in device code
            // throw std::runtime_error("Error: vertices array is empty");
        }
    }
};