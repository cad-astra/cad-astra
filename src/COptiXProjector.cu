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

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "../include/cuda/COptiXProjector.cuh"
#include "../include/optix_projector_data_structs.h"
#include "../include/cuda/common.cuh"
#include "../include/cuda/optix_host_functions.cuh"
#include "../include/global_config.h"
#include "../include/host_logging.h"

#include <optix_stubs.h>
#include <sutil/Exception.h>

#include <optix_function_table_definition.h>

#include <cuda_runtime.h>

struct SbtRecordHeader
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template<typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenSBTData>   RayGenSbtRecord;
typedef SbtRecord<MissData>        MissSbtRecord;
typedef SbtRecord<HitSBTData>      HitGroupSbtRecord;

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>
    ::context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
COptiXProjector<GeometryPolicy, T_3DIndex>::~COptiXProjector()
{
    clear();
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::set_mesh(const std::vector<CMesh> &mesh_vec)
{
    m_mesh_vec = mesh_vec;
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::initialize()
{
    MeshFP::Logging::info(MESH_FP_CLASS_NAME, "Initializing OptiX projector...");
    OPTIX_CHECK( optixInit() );
        
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext cuCtx = 0;  // zero means take the current context

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = MeshFP::CGlobalConfig::get_optix_log_level();
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &m_context ) );
    prepare_acceleration_structure();
    create_module();                    // Two modules: one with raygen, closest hit, miss, the other with the direct callables
    create_program_groups_and_pipeline_project(PROJECTOR_OPTIX_MAX_TRACING_DEPTH);    // OptiX programs have different names    
    setup_sbt();
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::prepare_acceleration_structure()
{
    if(GeometryPolicy<T_3DIndex>::get_optix_policy_idx() == OptiXMeshTransformPolicy)
    {
        // Create IAS -> SRT -> GAS, one GAS per mesh.
        // If no SRT is defined for the mesh, GAS is en immediate child to IAS, i.e., IAS->GAS
        const float static_transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
        std::vector<std::vector<float>> static_transforms_vec(m_mesh_vec.size());
    
        m_d_gas_output_buffer_vec.resize(m_mesh_vec.size());
        for(size_t i=0; i < m_mesh_vec.size(); i++)
        {
            std::vector<CMesh> dummy_mesh_vec(1);
            dummy_mesh_vec[0] = m_mesh_vec[i];
            OptixTraversableHandle gas_handle = mesh_fp::build_gas(m_context, dummy_mesh_vec, m_d_gas_output_buffer_vec[i]);

            if(m_mesh_vec[i].transformation_keys.size() > 0)
            {
                MeshFP::Logging::info(MESH_FP_CLASS_NAME, "Preparing SRT motion traversable for mesh [%lu]...", i);
                std::vector<OptixSRTData> srt_data_vec(m_mesh_vec[i].transformation_keys.size());
                
                for(size_t k=0; k < srt_data_vec.size(); k++)
                {
                    memcpy( &srt_data_vec[k], m_mesh_vec[i].transformation_keys[k].data(), 16 * sizeof(float) );
                }

                OptixTraversableHandle srt_transform_handle =
                    mesh_fp::build_srt_transformation_traversable(m_context, srt_data_vec, m_mesh_vec[i].begin_time, m_mesh_vec[i].end_time, gas_handle, m_motion_transform_devptr);
                m_ias_children_vec.push_back(srt_transform_handle);
            }
            else
            {
                MeshFP::Logging::info(MESH_FP_CLASS_NAME, "Only GAS traversable for mesh [%lu]...", i);
                m_ias_children_vec.push_back(gas_handle);
            }
            static_transforms_vec[i].resize(12);
            memcpy( static_transforms_vec[i].data(), &static_transform[0], 12 * sizeof(float) );
        }

        m_ias_handle = mesh_fp::build_ias(m_context, static_transforms_vec, m_ias_children_vec, m_d_ias_output_buffer);
        m_top_level_traversable = m_ias_handle;
    }
    else
    {
        // Create single GAS that includes all meshes
        m_d_gas_output_buffer_vec.resize(1);
        OptixTraversableHandle gas_handle = mesh_fp::build_gas(m_context, m_mesh_vec, m_d_gas_output_buffer_vec[0]);
        m_top_level_traversable = gas_handle;
    }
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::create_module()
{
    const std::string ptx_dir = MeshFP::CGlobalConfig::get_ptx_dir();

    // --------------------------------------------------------------------------------------------------------------------------------
    // TODO: The following examples of logging should work:
    // MeshFP::Logging::debug("My logging message"); // module="COptiXProjector", msg="My logging message"
    // MeshFP::Logging::debug("Variable value: %d", var); // module="COptiXProjector", msg="Variable value: %d", var
    // MeshFP::Logging::debug("OptiX Projector", "Variable value: %d", var); // module="OptiX Projector", msg="Variable value: %d", var
    // MeshFP::Logging::debug("OptiX Projector", "My logging message"); // module="OptiX Projector", msg="My logging message"
    // --------------------------------------------------------------------------------------------------------------------------------

    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "ptx directory: %s", ptx_dir.c_str());

    const std::string ptx_string_optix_projector = 
        mesh_fp::read_ptx
        (
            ptx_dir + std::string("/") + std::string
            (
                m_cfg.tracer_index==OptiXRecursivePolicy? "optix_projector_generated.cu.ptx" : "optix_projector_non_recursive_generated.cu.ptx"
            )
        );
    const std::string ptx_string_optix_projector_policies =
        mesh_fp::read_ptx(ptx_dir + std::string("/") + std::string("optix_projector_policies_generated.cu.ptx"));

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    // module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    m_pipeline_compile_options.usesMotionBlur        = GeometryPolicy<T_3DIndex>::get_optix_policy_idx() == OptiXMeshTransformPolicy?
        true : false;
    m_pipeline_compile_options.traversableGraphFlags = GeometryPolicy<T_3DIndex>::get_optix_policy_idx() == OptiXMeshTransformPolicy?
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    m_pipeline_compile_options.numPayloadValues      = 13;
    m_pipeline_compile_options.numAttributeValues    = 2;
    m_pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params"; // This is the name of the param struct variable in our device code

    size_t actual_size_of_log = m_sizeof_log;

    OPTIX_CHECK( optixModuleCreateFromPTX(
                    m_context,
                    &module_compile_options,
                    &m_pipeline_compile_options,
                    ptx_string_optix_projector.c_str(),
                    ptx_string_optix_projector.size(),   // Except the null character
                    log,
                    &actual_size_of_log,
                    &m_module
                    ) );

    actual_size_of_log = m_sizeof_log;
    OPTIX_CHECK( optixModuleCreateFromPTX(
                    m_context,
                    &module_compile_options,
                    &m_pipeline_compile_options,
                    ptx_string_optix_projector_policies.c_str(),
                    ptx_string_optix_projector_policies.size(),   // Except the null character
                    log,
                    &actual_size_of_log,
                    &m_policies_module
                    ) );
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>
    ::create_program_groups_and_pipeline_project(uint32_t max_depth)
{
    m_max_depth = max_depth;

    std::vector<OptixProgramGroup> program_groups;
    std::string log_str;
    log_str.resize(2048);

    std::vector<std::string> raygen_prog = {m_cfg.tracer_index==OptiXRecursivePolicy? "__raygen__project" : "__raygen__project_non_recursive"};
    std::vector<std::string> miss_prog   = {m_cfg.tracer_index==OptiXRecursivePolicy? "__miss__project" : "__miss__project_non_recursive"};

    std::vector<OptixProgramGroup> raygen_progrm_group = mesh_fp::create_program_group(m_context, OPTIX_PROGRAM_GROUP_KIND_RAYGEN, m_module, raygen_prog, log_str);
    program_groups.insert(program_groups.end(), raygen_progrm_group.begin(), raygen_progrm_group.end());
    m_raygen_prog_group = raygen_progrm_group[0];   // There's actually only one raygen program group

    std::vector<OptixProgramGroup> miss_prog_group = mesh_fp::create_program_group(m_context, OPTIX_PROGRAM_GROUP_KIND_MISS, m_module, miss_prog, log_str);
    program_groups.insert(program_groups.end(), miss_prog_group.begin(), miss_prog_group.end());
    m_miss_prog_group = miss_prog_group[0];     // There's actually only one miss program group

    // TODO: fix create_program_group for the special case of hit group

    // std::vector<OptixProgramGroup> hitgroup_prog_group = mesh_fp::create_program_group(m_context, OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_module, m_cfg.tracer_index==OptiXRecursivePolicy?{"__closesthit__project",m_cfg.tracer_index==OptiXRecursivePolicy? "__anyhit__project"}, log_str);
    // m_hitgroup_prog_group = hitgroup_prog_group[0]; // There's actually only one hit program group

    OptixProgramGroupOptions programGroupOptions = {};
    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH =m_cfg.tracer_index==OptiXRecursivePolicy? "__closesthit__project" : "__closesthit__project_non_recursive";
    hit_prog_group_desc.hitgroup.moduleAH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH =m_cfg.tracer_index==OptiXRecursivePolicy? "__anyhit__project" : "__anyhit__project_non_recursive";
    size_t actual_sizeof_log = log_str.size();
    OPTIX_CHECK( optixProgramGroupCreate( m_context,
                                          &hit_prog_group_desc,
                                          1,  // num program groups
                                          &programGroupOptions,
                                          &log_str[0],
                                          &actual_sizeof_log,
                                          &m_hitgroup_prog_group ) );
    program_groups.push_back(m_hitgroup_prog_group);

    // Callables
    std::vector<std::string> callables = 
        {
            "__direct_callable__rotation_geom_policy",
            "__direct_callable__vector_geom_policy",
            "__direct_callable__mesh_transform_geom_policy",
            "__direct_callable__max_attenuation_detector_pixel_policy",
            "__direct_callable__sum_intensity_detector_pixel_policy",
            "__direct_callable__sum_share_intensity_detector_pixel_policy"
        };
    m_callables_prog_group_vec = mesh_fp::create_program_group(m_context, OPTIX_PROGRAM_GROUP_KIND_CALLABLES, m_policies_module, callables, log_str);
    program_groups.insert(program_groups.end(), m_callables_prog_group_vec.begin(), m_callables_prog_group_vec.end());

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = m_max_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    // std::cerr << __FILE__ << " : " << __LINE__ << " - program_groups: " << program_groups.size() << std::endl;

    actual_sizeof_log = log_str.size();
    // This starts the LLVM JIT-compiler:
    OPTIX_CHECK( optixPipelineCreate(   m_context,
                                        &m_pipeline_compile_options,
                                        &pipeline_link_options,
                                        program_groups.data(),
                                        program_groups.size(),
                                        &log_str[0],
                                        &actual_sizeof_log,
                                        &m_pipeline ) );
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::setup_sbt(float media_refractive_index)
{
    // RayGen SBT record
    RayGenSbtRecord rg_sbt;
    rg_sbt.data.media_refractive_index = media_refractive_index;
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
    
    OPTIX_CHECK( optixSbtRecordPackHeader( m_raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

    m_sbt.raygenRecord                 = raygen_record;
    
    // Miss SBT record
    MissSbtRecord ms_sbt;
    ms_sbt.data = { 16 };          // Dummy data
    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( m_miss_prog_group, &ms_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );

    m_sbt.missRecordBase               = miss_record;
    m_sbt.missRecordStrideInBytes      = sizeof( MissSbtRecord );
    m_sbt.missRecordCount              = 1;

    // Hit SBT record
    std::vector<HitGroupSbtRecord> hitgroupRecords;
    for(size_t i=0; i < m_mesh_vec.size(); i++)
    {
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_prog_group, &hg_sbt));
        MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Mesh[%lu]: attenuation = %.1f, refractive index = %.1f", i, m_mesh_vec[i].attenuation, m_mesh_vec[i].refractive_index);
        hg_sbt.data.material_attenuation      = m_mesh_vec[i].attenuation;
        hg_sbt.data.material_refractive_index = m_mesh_vec[i].refractive_index;
        hg_sbt.data.max_depth                 = m_max_depth;

        hitgroupRecords.push_back(hg_sbt);
    }

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord ) * hitgroupRecords.size();
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record ),
                hitgroupRecords.data(),
                hitgroup_record_size,
                cudaMemcpyHostToDevice
                ) );

    m_sbt.hitgroupRecordBase           = hitgroup_record;
    m_sbt.hitgroupRecordStrideInBytes  = sizeof( HitGroupSbtRecord );
    m_sbt.hitgroupRecordCount          = hitgroupRecords.size();

    // Callables SBT
    std::vector<SbtRecordHeader> sbtRecordCallables(m_callables_prog_group_vec.size());

    for (size_t i = 0; i < sbtRecordCallables.size(); ++i)
    {
        OPTIX_CHECK( optixSbtRecordPackHeader(m_callables_prog_group_vec[i], &sbtRecordCallables[i]) );
    }

    CUdeviceptr  d_sbtRecordCallables;
    CUDA_CHECK( cudaMalloc((void**) &d_sbtRecordCallables, sizeof(SbtRecordHeader) * sbtRecordCallables.size()) );
    CUDA_CHECK( cudaMemcpy((void*) d_sbtRecordCallables, sbtRecordCallables.data(), sizeof(SbtRecordHeader) * sbtRecordCallables.size(), cudaMemcpyHostToDevice) );

    m_sbt.callablesRecordBase          = d_sbtRecordCallables;
    m_sbt.callablesRecordStrideInBytes = sizeof(SbtRecordHeader);
    m_sbt.callablesRecordCount         = sbtRecordCallables.size();
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::set_outbuffers(ProjectorOutBuffers buffs)
{
    m_pSino_device = buffs.pSino_device;
    // m_pRayPaths_device     = pRayPaths_device;
    // m_pRayPathLengths_device = pRayPathLengths_device;
    m_buffers_set = true;
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::run()
{
    // const uint32_t max_rays_per_dim = 1024 * 1024;

    const uint32_t n_views   = m_proj_geom.nViews;

    const uint32_t rays_rows = m_cfg.rays_row_count == -1?
        static_cast<uint32_t>(m_proj_geom.det_row_count) : static_cast<uint32_t>(m_cfg.rays_row_count);

    const uint32_t rays_cols = m_cfg.rays_col_count == -1?
        static_cast<uint32_t>(m_proj_geom.det_col_count) : static_cast<uint32_t>(m_cfg.rays_col_count);

    const uint32_t total_rays_cnt = rays_rows*rays_cols*n_views;

    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );

    uint32_t max_depth_rays_cnt = 0; // Number of rays that reached maximal depth
    uint32_t max_actual_depth   = 0; // Maximal actual ray depth

    uint32_t *max_depth_rays_cnt_devptr = nullptr;
    uint32_t *max_actual_depth_devptr = nullptr;

    CUDA_CHECK( cudaMalloc(&max_depth_rays_cnt_devptr, sizeof(uint32_t)) );
    CUDA_CHECK( cudaMemcpy(max_depth_rays_cnt_devptr, &max_depth_rays_cnt, sizeof(uint32_t), cudaMemcpyHostToDevice) );

    CUDA_CHECK( cudaMalloc(&max_actual_depth_devptr, sizeof(uint32_t)) );
    CUDA_CHECK( cudaMemcpy(max_actual_depth_devptr, &max_actual_depth, sizeof(uint32_t), cudaMemcpyHostToDevice) );

    COptixLaunchParams launch_params;
    launch_params.sinogram     = m_pSino_device;

    CProjectionGeometry *proj_geom_devptr;
    CUDA_CHECK( cudaMalloc(&proj_geom_devptr, sizeof(CProjectionGeometry)) );
    CUDA_CHECK( cudaMemcpy(proj_geom_devptr, &m_proj_geom, sizeof(CProjectionGeometry), cudaMemcpyHostToDevice) );

    // Projection geometry parameters
    launch_params.proj_geom_devptr      = proj_geom_devptr;
    launch_params.geometry_policy_index = GeometryPolicy<T_3DIndex>::get_optix_policy_idx();

    // Detector shape
    // TODO: implement either new geometry or a Detector geometry class/object
    launch_params.det_geometry_policy_index = m_proj_geom.det_geometry_ind;
    
    // Acceleration structure handle
    launch_params.handle                = m_top_level_traversable;

    // Tracing options
    launch_params.ray_maxdepth          = m_max_depth;
    launch_params.pMax_depth_rays_cnt   = max_depth_rays_cnt_devptr;
    launch_params.pMax_actual_depth     = max_actual_depth_devptr;

    // Detector operator policy
    launch_params.detector_policy_index = m_cfg.detector_policy_index;

    CUdeviceptr d_param;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( COptixLaunchParams ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                &launch_params, sizeof( launch_params ),
                cudaMemcpyHostToDevice
                ) );
    MeshFP::Logging::info(MESH_FP_CLASS_NAME,"Calling optixLaunch with (%u, %u, %u) rays, detector geometry: %s...", rays_rows, rays_cols, n_views,
        launch_params.det_geometry_policy_index==OptiXPlanePolicy? "plane" : "cylinder");

    // seems like there's a limit on depth in optixLaunch around 2**16
    OPTIX_CHECK( optixLaunch( m_pipeline, stream, d_param, sizeof( COptixLaunchParams ), &m_sbt, rays_rows, rays_cols, n_views) );
    CUDA_SYNC_CHECK();

    CUDA_CHECK( cudaMemcpy(&max_depth_rays_cnt, max_depth_rays_cnt_devptr, sizeof(uint32_t), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(&max_actual_depth,   max_actual_depth_devptr,   sizeof(uint32_t), cudaMemcpyDeviceToHost) );

    if(max_depth_rays_cnt != 0)
    {
        std::cout << "Info: " << max_depth_rays_cnt << " rays "
                  << "( "
                  << std::setprecision(2)
                //   << std::setw(6)
                  << std::fixed
                  << 100*static_cast<float>(max_depth_rays_cnt) / total_rays_cnt
                  << "% ) "
                  << "reached maximal recursion depth of " << m_max_depth << ". "
                  << "Maximal actual ray depth: "<< max_actual_depth << std::endl;
    }
    CUDA_CHECK( cudaFree(max_depth_rays_cnt_devptr) );
    CUDA_CHECK( cudaFree(max_actual_depth_devptr  ) );

    CUDA_CHECK( cudaFree( proj_geom_devptr        ) );

    CUDA_CHECK( cudaFree( (void *) d_param        ) );

    CUDA_CHECK( cudaStreamDestroy( stream ) );
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
void COptiXProjector<GeometryPolicy, T_3DIndex>::clear()
{
    MeshFP::Logging::info(MESH_FP_CLASS_NAME, "Clearing up OptiX projector...");
    if(m_sbt.raygenRecord != 0)
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.raygenRecord       ) ) );
    if(m_sbt.missRecordBase != 0)
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.missRecordBase     ) ) );
    if(m_sbt.hitgroupRecordBase != 0)
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.hitgroupRecordBase ) ) );
    if(m_sbt.callablesRecordBase != 0)
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.callablesRecordBase ) ) );

    for(size_t i=0; i < m_d_gas_output_buffer_vec.size(); i++)
        if(m_d_gas_output_buffer_vec[i] != 0)
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_d_gas_output_buffer_vec[i] ) ) );

    if(m_motion_transform_devptr != 0)
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_motion_transform_devptr) ) );

    if(m_d_ias_output_buffer !=0)
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_d_ias_output_buffer) ) );

    if(m_pipeline != nullptr)
        OPTIX_CHECK( optixPipelineDestroy( m_pipeline ) );

    if(m_hitgroup_prog_group != nullptr)
        OPTIX_CHECK( optixProgramGroupDestroy( m_hitgroup_prog_group ) );
    if(m_raygen_prog_group != nullptr)
        OPTIX_CHECK( optixProgramGroupDestroy( m_raygen_prog_group ) );
    if(m_miss_prog_group != nullptr)
        OPTIX_CHECK( optixProgramGroupDestroy( m_miss_prog_group ) );
    for(size_t i=0; i < m_callables_prog_group_vec.size(); i++)
        OPTIX_CHECK( optixProgramGroupDestroy( m_callables_prog_group_vec[i] ) );

    if(m_module != nullptr)
        OPTIX_CHECK( optixModuleDestroy( m_module ) );
    if(m_policies_module != nullptr)
        OPTIX_CHECK( optixModuleDestroy( m_policies_module ) );
    if(m_context != nullptr)
        OPTIX_CHECK( optixDeviceContextDestroy(m_context) );
}

template<
         template <class T_3DIndex> typename GeometryPolicy,
         typename T_3DIndex
        >
std::string COptiXProjector<GeometryPolicy, T_3DIndex>::type = 
    "optix_" + GeometryPolicy<T_3DIndex>::type + "_" +  T_3DIndex::type;