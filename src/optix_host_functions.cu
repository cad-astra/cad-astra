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

#include <string>
#include <fstream>

#include "../include/cuda/optix_host_functions.cuh"

#include <optix_stubs.h>
#include <sutil/Exception.h>
// #include <sutil/vec_math.h>
#include "../include/cuda/vec_utils.cuh"

#include "../include/cuda/common.cuh"
#include "../include/host_logging.h"

__host__ OptixTraversableHandle mesh_fp::build_gas(OptixDeviceContext &optix_context, const std::vector<CMesh> &mesh_vec, CUdeviceptr &gas_output_buffer_devptr)
{
    std::vector<OptixBuildInput> triangle_input_vec(mesh_vec.size());
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };    // This flag may affect performance
    for(size_t i=0; i < mesh_vec.size(); i++)
    {
        triangle_input_vec[i] = {};
        triangle_input_vec[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        
        // Always C-order arrays expected!
        // Initial values in case we're dealing with non-indexed mesh        
        triangle_input_vec[i].triangleArray.indexBuffer      = reinterpret_cast<CUdeviceptr>(nullptr);
        triangle_input_vec[i].triangleArray.numIndexTriplets = static_cast<uint32_t>(0);
        if(mesh_vec[i].vertices != nullptr) // Vertices only
        {
            triangle_input_vec[i].triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            // triangle_input_vec[i].triangleArray.vertexStrideInBytes = sizeof(float) * 3;
            triangle_input_vec[i].triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>( const_cast<float **>(&mesh_vec[i].vertices) );
            triangle_input_vec[i].triangleArray.numVertices   = static_cast<uint32_t>(mesh_vec[i].nVertices); 
        }
        if(mesh_vec[i].faces != nullptr) // Indexed mesh
        {
            // Indices
            triangle_input_vec[i].triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input_vec[i].triangleArray.indexBuffer      = reinterpret_cast<CUdeviceptr>(mesh_vec[i].faces);
            triangle_input_vec[i].triangleArray.numIndexTriplets = static_cast<uint32_t>(mesh_vec[i].nFaces);
        }
        MeshFP::Logging::debug("optix_host_functions", "Mesh[%lu] vertices: %d\n", i, triangle_input_vec[i].triangleArray.numVertices);
        
        triangle_input_vec[i].triangleArray.flags                       = triangle_input_flags;
        triangle_input_vec[i].triangleArray.numSbtRecords               = 1;
        triangle_input_vec[i].triangleArray.sbtIndexOffsetBuffer        = 0; 
        triangle_input_vec[i].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
        triangle_input_vec[i].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
    }

    // BLAS/GAS setup
    OptixTraversableHandle gas_hadle;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    // accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optix_context,
                 &accelOptions,
                 triangle_input_vec.data(),
                 mesh_vec.size(),  // num_build_inputs
                 &blasBufferSizes
                 ));

    // Build BLAS/GAS
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), blasBufferSizes.tempSizeInBytes ) );
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( blasBufferSizes.outputSizeInBytes, 8ull );
    CUDA_CHECK(
                cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8)
              );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
                                optix_context,
                                0,                  // CUDA stream
                                &accelOptions,
                                triangle_input_vec.data(),
                                mesh_vec.size(),  // num build inputs
                                d_temp_buffer_gas,
                                blasBufferSizes.tempSizeInBytes,
                                d_buffer_temp_output_gas_and_compacted_size,
                                blasBufferSizes.outputSizeInBytes,
                                &gas_hadle,
                                &emitProperty,  // emitted property list
                                1               // num emitted properties
                            ) );
    CUDA_SYNC_CHECK();
    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < blasBufferSizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &gas_output_buffer_devptr ), compacted_gas_size ) );
        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( optix_context, 0, gas_hadle, gas_output_buffer_devptr, compacted_gas_size, &gas_hadle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        gas_output_buffer_devptr = d_buffer_temp_output_gas_and_compacted_size;
    }

    return gas_hadle;
}

// Default static transform (i.e., that does nothing) should be:
// const float static_transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
__host__  OptixTraversableHandle mesh_fp::build_ias
(
    OptixDeviceContext &optix_context,
    const std::vector<std::vector<float>> static_transforms_vec,   // Every element is a 12-element vector that contains a 3x4 static transform matrix stored row-wise
    const std::vector<OptixTraversableHandle> &children_traversable_vec,
    CUdeviceptr &ias_output_buffer_devptr
)
{
    // Instance Acceleration structure

    std::vector<OptixInstance> optix_instances_vec(children_traversable_vec.size());
    const size_t instance_size_in_bytes = sizeof( OptixInstance ) * optix_instances_vec.size();

    memset( optix_instances_vec.data(), 0, instance_size_in_bytes );

    for(size_t i=0; i < optix_instances_vec.size(); i++)
    {
        optix_instances_vec[i].flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instances_vec[i].instanceId        = 0;
        optix_instances_vec[i].sbtOffset         = i;
        optix_instances_vec[i].visibilityMask    = 1;
        optix_instances_vec[i].traversableHandle = children_traversable_vec[i];
        memcpy( optix_instances_vec[i].transform, static_transforms_vec[i].data(), sizeof( float ) * 12 );
    }

    CUdeviceptr instances_devptr;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &instances_devptr ), instance_size_in_bytes) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( instances_devptr ),
                optix_instances_vec.data(),
                instance_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = instances_devptr;
    instance_input.instanceArray.numInstances = optix_instances_vec.size();

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags              = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation               = OPTIX_BUILD_OPERATION_BUILD;

    // Do we need motion options for the IAS?
    // accel_options.motionOptions.numKeys   = 4;
    // accel_options.motionOptions.timeBegin = 0.0f;
    // accel_options.motionOptions.timeEnd   = 4.0f;
    // accel_options.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                optix_context,
                &accel_options,
                &instance_input,
                1, // num build inputs - one IAS
                &ias_buffer_sizes
                ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_temp_buffer ),
                ias_buffer_sizes.tempSizeInBytes
                ) );

    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &ias_output_buffer_devptr ),
                ias_buffer_sizes.outputSizeInBytes
                ) );

    OptixTraversableHandle ias_handle;
    OPTIX_CHECK( optixAccelBuild(
                 optix_context,
                 0,                  // CUDA stream
                 &accel_options,
                 &instance_input,
                 1,                  // num build inputs - one IAS
                 d_temp_buffer,
                 ias_buffer_sizes.tempSizeInBytes,
                 ias_output_buffer_devptr,
                 ias_buffer_sizes.outputSizeInBytes,
                 &ias_handle,
                 nullptr,            // emitted property list
                 0                   // num emitted properties
                ) );

    CUDA_SYNC_CHECK();

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( instances_devptr ) ) );

    return ias_handle;
}

__host__ OptixTraversableHandle mesh_fp::build_srt_transformation_traversable
(
    OptixDeviceContext &optix_context,
    const std::vector<OptixSRTData> &srt_data_vec,
    int begin_time,
    int end_time,
    OptixTraversableHandle child_traversable,
    CUdeviceptr motion_transform_devptr
)
{
    // Motion transform

    // OptixSRTData srt_data[2] = 
    // {
    //     //sx,   a,   b, pvx,  sy,   c, pvy,  sz, pvz,  qx,  qy,                  qz,                  qw, tx,   ty,  tz
    //     {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,                 0.f,                 1.f, 0.f, 0.f, 0.f},
    //     {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, std::sin(M_PI_4f32*0.5f), std::cos(M_PI_4f32*0.5f), 0.f, 0.f, 0.f}
    // };
    
    size_t N = srt_data_vec.size();

    size_t transformSizeInBytes = sizeof( OptixSRTMotionTransform ) + ( N-2 ) * sizeof( OptixSRTData );
    OptixSRTMotionTransform* srtMotionTransform = (OptixSRTMotionTransform*) malloc( transformSizeInBytes );
    memset( srtMotionTransform, 0, transformSizeInBytes );
    
    // setup other members of srtMotionTransform
    srtMotionTransform->motionOptions.numKeys   = N;
    srtMotionTransform->motionOptions.timeBegin = begin_time;
    srtMotionTransform->motionOptions.timeEnd   = end_time;
    srtMotionTransform->child = child_traversable;
    memcpy( srtMotionTransform->srtData, srt_data_vec.data(), N * sizeof( OptixSRTData ) );
        
    // Dont forget to clean up...
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &motion_transform_devptr),
                transformSizeInBytes
                ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( motion_transform_devptr ),
                srtMotionTransform,
                transformSizeInBytes,
                cudaMemcpyHostToDevice
                ) );

    OptixTraversableHandle srt_transform_handle;
    OPTIX_CHECK( optixConvertPointerToTraversableHandle(
                optix_context,
                motion_transform_devptr,
                OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM,
                &srt_transform_handle
                ) );
 
    free(srtMotionTransform);

    return srt_transform_handle;
}

/**
 * @brief Create a program group.
 * 
 * @param optix_context OptiX context. 
 * @param kind Program group kind. Can be one of the following: OPTIX_PROGRAM_GROUP_KIND_RAYGEN, OPTIX_PROGRAM_GROUP_KIND_MISS, OPTIX_PROGRAM_GROUP_KIND_HITGROUP, OPTIX_PROGRAM_GROUP_KIND_CALLABLES.
 * @param module OptiX module that contains the programs.
 * @param entry_function_name_vec Names of the OptiX programs. Every name should begin with either "__raygen__", or "__miss__", or "__closesthit__", or "__direct_callable__", or "__continuation_callable__".
 * @param log_str Log string
 * @return std::vector<OptixProgramGroup> Vector of program group objects.
 */
__host__ std::vector<OptixProgramGroup> mesh_fp::create_program_group(OptixDeviceContext &optix_context,
                                                             OptixProgramGroupKind kind,
                                                             OptixModule module,
                                                             const std::vector<std::string> &entry_function_name_vec,
                                                             std::string &log_str)
{
    std::vector<OptixProgramGroup> progr_group_vec;
    OptixProgramGroupOptions programGroupOptions = {};
    std::vector<OptixProgramGroupDesc>  prog_group_desc_vec;
    
    std::stringstream error_ss;
    OptixProgramGroupDesc pgd = {};
    pgd.kind = kind;
    for(size_t i=0; i < entry_function_name_vec.size(); i++)
    {    
        switch (kind)
        {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            pgd.raygen.module = module;
            pgd.raygen.entryFunctionName = entry_function_name_vec[i].c_str();
            break;
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            pgd.miss.module = module;
            pgd.miss.entryFunctionName = entry_function_name_vec[i].c_str();
            break;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            if(entry_function_name_vec[i].find("__closesthit__") != std::string::npos)
            {
                pgd.hitgroup.moduleCH = module;
                pgd.hitgroup.entryFunctionNameCH = entry_function_name_vec[i].c_str();
            }
            else if(entry_function_name_vec[i].find("__anyhit__") != std::string::npos)
            {
                pgd.hitgroup.moduleAH = module;
                pgd.hitgroup.entryFunctionNameAH = entry_function_name_vec[i].c_str();
            }
            else
            {
                error_ss << "Error: unknown type of OptiX program in hit group: " << entry_function_name_vec[i];
                throw std::runtime_error(error_ss.str().c_str());
            }
            break;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            if(entry_function_name_vec[i].find("__direct_callable__") != std::string::npos)
            {
                pgd.callables.moduleDC = module;
                pgd.callables.entryFunctionNameDC = entry_function_name_vec[i].c_str();
            }
            else if(entry_function_name_vec[i].find("__continuation_callable__") != std::string::npos)
            {
                pgd.callables.moduleCC = module;
                pgd.callables.entryFunctionNameCC = entry_function_name_vec[i].c_str();
            }
            else
            {
                error_ss << "Error: unknown type of OptiX program in callables group: " << entry_function_name_vec[i];
                throw std::runtime_error(error_ss.str().c_str());
            }
            break;
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            pgd.exception.module = module;
            pgd.exception.entryFunctionName = entry_function_name_vec[i].c_str();
            break;
        default:
            // No other kinds of groups exist...
            break;
        }
        prog_group_desc_vec.push_back(pgd);
    }

    progr_group_vec.resize(prog_group_desc_vec.size());

    size_t actual_sizeof_log = log_str.size();  // In-OUT variable
    OPTIX_CHECK( optixProgramGroupCreate(
                                            optix_context,
                                            prog_group_desc_vec.data(),
                                            prog_group_desc_vec.size(), // num program groups
                                            &programGroupOptions,
                                            &log_str[0],
                                            &actual_sizeof_log,
                                            progr_group_vec.data() ) );
    return progr_group_vec;
}

// This function is copied from the OptiX-Apps project
std::string mesh_fp::read_ptx(const std::string& filename)
{
    std::ifstream inputPtx(filename);

    if (!inputPtx)
    {
        std::cerr << "ERROR: readPTX() Failed to open file " << filename << '\n';
        return std::string();
    }

    std::stringstream ptx;

    ptx << inputPtx.rdbuf();

    if (inputPtx.fail())
    {
        std::cerr << "ERROR: readPTX() Failed to read file " << filename << '\n';
        return std::string();
    }
    return ptx.str();
}