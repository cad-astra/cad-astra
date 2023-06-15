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

#include "CBaseObject.h"
#include "proj_geometry.h"
#include "mesh.h"
#include "CProjectorConfiguration.h"

struct ProjectorOutBuffers
{
    float *pSino_device;
    float *pRayPaths_device;
    float *pRayPathLengths_device;
};

class CProjector : public CBaseObject
{
public:
    CProjector() : CBaseObject(Projector), m_buffers_set(false), m_geometry_set(false), m_mesh_set(false) {}
    virtual ~CProjector() {}
    virtual void initialize() {} //! Implement specific initialization in the inherited class
    void set_geometry(const CProjectionGeometry &geom)
    {
        m_proj_geom = geom;     // shallow copy
        m_geometry_set = true;
        // printf("%f, %f, %f\n", m_proj_geom.vg.views[0], m_proj_geom.rg.det_height, m_proj_geom.nViews);
    }
    virtual void set_mesh(const std::vector<CMesh> &mesh_vec)
    {
        // To be overridden in the child class
    }
    void configure(const CProjectorConfiguration &cfg)
    {
        m_cfg = cfg;
    }
    virtual void set_outbuffers(ProjectorOutBuffers buffs) {}
    virtual void run() {}
protected:
    // Data and paramters shared by all projector classes
    float *m_pSino_device;
    CProjectionGeometry     m_proj_geom;
    CMesh                   m_mesh;
    CProjectorConfiguration m_cfg; // Only for COptiXProjector
    bool                    m_buffers_set, m_geometry_set, m_mesh_set;
};