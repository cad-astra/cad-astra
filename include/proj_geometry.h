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

struct CRotationGeometry
{
public:
    // CRotationGeometry()
    //     : det_width(0), det_height(0), source_origin(0), origin_det(0), angles(nullptr)
    // {}
    /**
     * @brief This operator performs shallow copy.
     * 
     */
    // CRotationGeometry &operator=(const CRotationGeometry &to_copy)
    // {
    //     if(this != &to_copy)
    //     {
    //         this->det_width     = to_copy.det_width;
    //         this->det_height    = to_copy.det_height;
    //         this->source_origin = to_copy.source_origin;
    //         this->origin_det    = to_copy.origin_det;
    //         this->angles        = to_copy.angles;
    //     }
    //     return *this;
    // }
    float   det_width;
    float   det_height;
    float   source_origin;
    float   origin_det;
    float  *angles;     //! Should be an array (nViews,   )
};

struct CVecGeometry
{
public:
    // CVecGeometry()
    //     : views(nullptr)
    // {}
    float  *views;      //! Should be an array (nViews, 12)
    /**
     * @brief This operator performs shallow copy.
     * 
     */
    // CVecGeometry &operator=(const CVecGeometry &to_copy)
    // {
    //     if(this != &to_copy)
    //     {
    //         this->views = to_copy.views;
    //     }
    //     return *this;
    // }
};

struct CMeshTransformGeometry
{
public:
    float *projector_view;  //! Should be an array (1, 12)
    float *view_keys;       //! Time keys for projection views, should be an array (nViews, )
};

struct CProjectionGeometry
{
    // CProjectionGeometry()
    //     : det_row_count(0), det_col_count(0), nViews(0)
    // {}
    // Common geometry parameters
    int     det_row_count;  //! Number of rows in detector array
    int     det_col_count;  //! Number of columms in detector array
    size_t  nViews;         //! Number of views (angles)
    
    // Specific geometries
    CRotationGeometry       rg;
    CVecGeometry            vg;
    CMeshTransformGeometry mtg;

    // NOTE: Ad-hoc solution for the cylindric detector
    // TODO(pavel): either introduce a new geometry or implement Detector geometry object
    float  det_radius;
    OptiXDetectorGeometryPolicyIndex det_geometry_ind;

    // CProjectionGeometry &operator=(const CProjectionGeometry &to_copy)
    // {
    //     if(this != &to_copy)
    //     {
    //         this->det_row_count = to_copy.det_row_count;
    //         this->det_col_count = to_copy.det_col_count;
    //         this->nViews = to_copy.nViews;
    //         this->rg = to_copy.rg;
    //         this->vg = to_copy.vg;
    //     }
    //     return *this;
    // }
};