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

#define PROJECTOR_OPTIX_MAX_TRACING_DEPTH 31  // Maximal possible depth 

#include <optix.h>
#include <vector>
#include <string>

#include "array_index.cuh"
#include "../CProjector.h"
#include "../mesh.h"
#include "ProjectionGeometryPolicies.cuh"
#include "MeshRepresentationPolicies.cuh"
#include "../typelist.h"
#include "../CClassInfo.hpp"

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
class COptiXProjector : public CProjector, public CClassInfo<COptiXProjector<GeometryPolicy, T_3DIndex>>
{
public:
    __host__ COptiXProjector()
        : m_context(nullptr), m_motion_transform_devptr(0),
          m_d_ias_output_buffer(0),
          m_module(nullptr), m_policies_module(nullptr),
          m_raygen_prog_group(nullptr), m_miss_prog_group(nullptr),
          m_hitgroup_prog_group(nullptr),
          m_pipeline(nullptr)
    {
        m_pipeline_compile_options = {};
        m_sizeof_log = sizeof(log);
        m_sbt = {};
        m_sbt.raygenRecord        = 0;
        m_sbt.missRecordBase      = 0;
        m_sbt.hitgroupRecordBase  = 0;
        m_sbt.callablesRecordBase = 0;
    }
    __host__ ~COptiXProjector() override;
    
    __host__ void initialize () override;

    __host__ void set_mesh(const std::vector<CMesh> &mesh_vec) override;

    __host__ void set_outbuffers(ProjectorOutBuffers buffs) override;
    __host__ void run() override;
    __host__ void clear();

    static std::string type;
private:
    // Set mesh and build the corresponding acceleration structure:
    __host__ void prepare_acceleration_structure();

    // Create program module from PTX file
    __host__ void create_module();
    __host__ void create_program_groups_and_pipeline_project(uint32_t max_depth=31);
    // void create_program_groups_and_pipeline_compute_path(uint32_t max_depth=31);
    __host__ void setup_sbt(float media_refractive_index=1.0f);

    // TODO(pavel): move to the parent class or make global:
    __host__ static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */);

    // TODO: change this to local std::string variable
    char log[2048]; // For error reporting from OptiX functions
    size_t m_sizeof_log;
    
    // OptiX context:
    OptixDeviceContext m_context;
    
    // Acceleration structure handles and data buffers:
    std::vector<OptixTraversableHandle> m_ias_children_vec;
    OptixTraversableHandle              m_ias_handle;

    std::vector<CUdeviceptr>            m_d_gas_output_buffer_vec;
    CUdeviceptr                         m_motion_transform_devptr;
    CUdeviceptr                         m_d_ias_output_buffer;

    OptixTraversableHandle m_top_level_traversable;
    
    // Pipeline options must be consistent for all modules used in a single pipeline
    OptixPipelineCompileOptions m_pipeline_compile_options;
    
    // OptiX program modules:
    OptixModule m_module;
    OptixModule m_policies_module;
    
    // OptiX program groups and pipeline:
    OptixProgramGroup m_raygen_prog_group;
    OptixProgramGroup m_miss_prog_group;
    OptixProgramGroup m_hitgroup_prog_group;
    std::vector<OptixProgramGroup> m_callables_prog_group_vec; //! Policies as callables
    OptixPipeline m_pipeline;

    // Shader binding table:
    OptixShaderBindingTable m_sbt;

    uint32_t m_max_depth;

    // All meshes that we're projecting
    std::vector<CMesh> m_mesh_vec;
};

typedef COptiXProjector<CRotationGeometryPolicy,      CIndexer3D_C> COptiXProjectorRotationCIndex;
typedef COptiXProjector<CRotationGeometryPolicy,      CIndexer3D_F> COptiXProjectorRotationFortranIndex;
typedef COptiXProjector<CVecGeometryPolicy,           CIndexer3D_C> COptiXProjectorVecCIndex;
typedef COptiXProjector<CVecGeometryPolicy,           CIndexer3D_F> COptiXProjectorVecFortranIndex;
typedef COptiXProjector<CMeshTransformGeometryPolicy, CIndexer3D_C> COptiXProjectorMeshTransformCIndex;
typedef COptiXProjector<CMeshTransformGeometryPolicy, CIndexer3D_F> COptiXProjectorMeshTransformFortranIndex;

template class COptiXProjector<CRotationGeometryPolicy,      CIndexer3D_C>;
template class COptiXProjector<CRotationGeometryPolicy,      CIndexer3D_F>;
template class COptiXProjector<CVecGeometryPolicy,           CIndexer3D_C>;
template class COptiXProjector<CVecGeometryPolicy,           CIndexer3D_F>;
template class COptiXProjector<CMeshTransformGeometryPolicy, CIndexer3D_C>;
template class COptiXProjector<CMeshTransformGeometryPolicy, CIndexer3D_F>;

typedef TYPELIST_6 (COptiXProjectorRotationCIndex,
                    COptiXProjectorRotationFortranIndex,
                    COptiXProjectorVecCIndex,
                    COptiXProjectorVecFortranIndex,
                    COptiXProjectorMeshTransformCIndex,
                    COptiXProjectorMeshTransformFortranIndex) OptiXRecursiveProjectors;