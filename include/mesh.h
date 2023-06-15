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

#include <vector>

struct CMesh
{
    CMesh(float attenuation=1.0f, float refractive_index=1.0f,
          size_t nFaces=0,
          size_t nVertices=0,
          float *vertices_devptr=nullptr,
          int   *faces_devptr=nullptr,
          float *normals_devptr=nullptr)
        : attenuation(attenuation), refractive_index(refractive_index),
          nFaces(nFaces), nVertices(nVertices),
          vertices(vertices_devptr), faces(faces_devptr), normals(normals_devptr),
          begin_time(0.f), end_time(0.f)
    {}
    // Mesh material
    float attenuation;
    float refractive_index;
    
    // Mesh triangles
    size_t nFaces;
    size_t nVertices;

    float *vertices;    //! Should be either (nVertices, 3) or (nFaces, 9)
    int   *faces;       //! Should be (nFaces,    3)
    float *normals;     //! Should be (nFaces,    3)

    // Mesh transformation
    std::vector< std::vector<float> > transformation_keys;
    float begin_time;
    float end_time;
};