#cython: language_level=3

#
# This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
# Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.
#
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from mesh_projector.projection_geometry import ProjectionGeometry
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.type_check import common_type
cimport numpy as np
from libcpp cimport bool
from libcpp cimport cast
from libcpp.string cimport string
from libcpp.vector cimport vector
from collections.abc import Iterable
import ctypes
from libc.string cimport memcpy

from mesh_projector.objects_manager cimport *

from mesh_projector import log

cdef extern from "cuda/array_index.cuh":
    cdef cppclass CIndexer3D_C

cdef extern from "cuda/optix_projector_policies.cuh":
    ctypedef enum OptiXDetectorPolicyIndex:
        OptiXMaxAttenuationPolicy
        OptiXSumIntencityPolicy
        OptiXSumShareIntencityPolicy

    ctypedef enum OptiXDetectorGeometryPolicyIndex:
        OptiXPlanePolicy
        OptiXCylinderPolicy

    ctypedef enum OptiXTracerPolicyIndex:
        OptiXRecursivePolicy
        OptiXNonRecursivePolicy

cdef extern from "cuda/cuManagedArray.cuh":
    cdef cppclass cuManagedArrayFloat:
        size_t size()
        float *dataPtr()
    cdef cppclass cuManagedArrayInt:
        size_t size()
        int *dataPtr()

cdef extern from "proj_geometry.h":
    cdef cppclass CRotationGeometry:
        float  det_width
        float  det_height
        float  source_origin
        float  origin_det
        float *angles     # Should be an array (nViews, )

    cdef cppclass CVecGeometry:
        float *views      # Should be an array (nViews, 12)
    
    cdef cppclass CMeshTransformGeometry:
        float *projector_view
        float *view_keys

    cdef cppclass CProjectionGeometry:
        int     det_row_count
        int     det_col_count   # Number detector columns
        size_t  nViews          # number of views (proejction angles)
        float det_radius    # radius for the cylindrical detector
        OptiXDetectorGeometryPolicyIndex det_geometry_ind  # Ad-hoc solution: detector geometry type
        CRotationGeometry rg
        CVecGeometry      vg
        CMeshTransformGeometry mtg

cdef extern from "mesh.h":
    cdef cppclass CMesh:
        float  attenuation
        float  refractive_index
        size_t nFaces
        size_t nVertices
        float *vertices    # Should be either (nVertices, 3) or (nFaces, 9)
        int   *faces       # Should be (nFaces,   3)
        float *normals     # Should be (nFaces,   3)
        vector[vector[float]] transformation_keys
        float begin_time
        float end_time

cdef extern from "CProjectorConfiguration.h":
    cdef cppclass CProjectorConfiguration:
        int rays_row_count
        int rays_col_count
        unsigned int detector_policy_index
        OptiXTracerPolicyIndex tracer_index

cdef extern from "CProjector.h":
    cdef cppclass ProjectorOutBuffers:
        float *pSino_device
        float *pRayPaths_device
        float *pRayPathLengths_device
    cdef cppclass CProjector:
        void initialize()
        void set_geometry( CProjectionGeometry )
        # void set_mesh( CMesh )
        void set_mesh( vector[CMesh] )
        void set_outbuffers( ProjectorOutBuffers )
        void run()
        void configure( CProjectorConfiguration )

cdef extern from "ProjectorFactory.h":
    cdef cppclass CProjectorFactory:
        CProjectorFactory(string)


from mesh_projector.objects_manager cimport *

ctypedef cuManagedArrayFloat* cuManagedArrayFloat_ptr
ctypedef cuManagedArrayInt*   cuManagedArrayInt_ptr

INVALID_ID = -1

def create_projector_c(projector_type, cfg):
    
    geom_dict = {'cone' : "rotation_geom", 'cone_vec': "vec_geom", 'cone_mesh_transform': "mesh_transform"}  # Python names to c++ names (types)

    projector_full_type = projector_type + "_" + geom_dict[cfg['geometry'].type] + '_' + 'c_index'
    
    logger = log.logging.getLogger(log.main_logger_name())
    if projector_type == "optix":
        logger.info(f'Creating projector {projector_full_type}:{cfg.get("tracer_policy", "recursive")}')
    elif projector_type == "raster":
        logger.info(f'Creating projector {projector_full_type}')
    else:
        raise ValueError(f'Unknown projector type {projector_type}')

    # Create empty projector...
    cdef GPUProjectorObjectFactory *pFactory = <GPUProjectorObjectFactory *> new CProjectorFactory(projector_full_type.encode("UTF-8"))
    
    cdef int id = pObjectManager.create_projector(pFactory)
    
    logger.info(f'Created projector with id {id}')

    # Configure projector...
    cdef CProjector *cuProjector = pObjectManager.get_projector(id)
    
    # Cython doesn't allow declaration inside 'if' block,
    # so all variables are declared at function level.

    # Rotation (cone) geometry
    cdef int angles_id
    cdef cuManagedArrayFloat *cuAngles
    # Vector (cone) geometry
    cdef int vectors_id
    cdef cuManagedArrayFloat *cuVectors
    # Mesh Transform geomtry
    cdef int projector_view_id
    cdef int view_keys_id
    cdef cuManagedArrayFloat *cuViewKeys

    cdef CProjectionGeometry proj_geom
    proj_geom.det_row_count = cfg['geometry'].det_row_count
    proj_geom.det_col_count = cfg['geometry'].det_col_count

    logger.info(f"Detector geometry: {cfg['geometry'].detector_geometry}, cylinder radius: {cfg['geometry'].detector_cylinder_radius}")
    # See OptiXDetectorGeometryPolicyIndex enum for det_geometry_ind values

    detector_geometry_policies_map = { 'plane'    : OptiXPlanePolicy,
                                       'cylinder' : OptiXCylinderPolicy}

    proj_geom.det_geometry_ind = detector_geometry_policies_map.get( cfg['geometry'].detector_geometry, 0 )
    # Cylindrical detector step
    proj_geom.det_radius =  cfg['geometry'].detector_cylinder_radius


    if cfg['geometry'].type == 'cone':
        angles_id = cfg['geometry'].angles_id
        cuAngles  = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(angles_id))

        proj_geom.rg.det_width     = cfg['geometry'].det_spacing_x
        proj_geom.rg.det_height    = cfg['geometry'].det_spacing_y
        proj_geom.rg.source_origin = cfg['geometry'].source_origin
        proj_geom.rg.origin_det    = cfg['geometry'].det_origin
        proj_geom.rg.angles        = cuAngles.dataPtr()
        proj_geom.nViews           = cuAngles.size()

    elif cfg['geometry'].type == 'cone_vec':
        vectors_id = cfg['geometry'].vectors_id
        cuVectors = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(vectors_id))

        proj_geom.nViews   =  <size_t> (cuVectors.size() / 12)
        proj_geom.vg.views =  cuVectors.dataPtr()
    elif cfg['geometry'].type == 'cone_mesh_transform':

        projector_view_id = cfg['geometry'].projector_view
        cuVectors = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(projector_view_id))

        proj_geom.mtg.projector_view = cuVectors.dataPtr()

        view_keys_id = cfg['geometry'].view_keys
        cuViewKeys = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(view_keys_id))
        proj_geom.mtg.view_keys = cuViewKeys.dataPtr()
        proj_geom.nViews = <size_t> cuViewKeys.size()
    else:
        raise RuntimeError(f"Cannot create projector of unknown geometry: {cfg['geometry'].type}")

    # Pick mesh data and 
    cdef int vertices_id
    cdef int faces_id
    cdef int normals_id
    cdef cuManagedArrayFloat *cuVertices
    cdef cuManagedArrayInt   *cuFaces
    cdef cuManagedArrayFloat *cuNormals

    cdef CMesh mesh
    cdef vector[CMesh] mesh_vec
    cdef np.ndarray[ndim=2, dtype=np.float32_t] optix_srt_data_vec

    if isinstance(cfg['mesh'], Iterable):
        logger.info(f'Preparing {len(cfg["mesh"])} meshes')
        for i in range(len(cfg['mesh'])):
            logger.info(f'Preparing mesh {i}')
            mesh.attenuation      = cfg['mesh'][i].attenuation
            mesh.refractive_index = cfg['mesh'][i].refractive_index
            mesh.transformation_keys.resize(0)
            mesh.begin_time = mesh.end_time = 0

            vertices_id = cfg['mesh'][i].vertices
            faces_id    = cfg['mesh'][i].faces
            normals_id  = cfg['mesh'][i].normals        

            if(vertices_id != -1):
                cuVertices = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(vertices_id))
                mesh.vertices  = cuVertices.dataPtr()
                mesh.nVertices = <size_t>(cuVertices.size() / 3)
                mesh.nFaces    = <size_t>(cuVertices.size() / 9 ) ### Are you sure of this?
            if(faces_id != -1):
                cuFaces    = cast.dynamic_cast[cuManagedArrayInt_ptr]   (pObjectManager.get_cuda_array(faces_id))
                mesh.faces     = cuFaces.dataPtr()
                mesh.nFaces    = <size_t>( cuFaces.size() / 3 )
            if(normals_id != -1):
                cuNormals  = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(normals_id))
                mesh.normals   = cuNormals.dataPtr()

            if(cfg['mesh'][i].transformations is not None):

                # logger.info(f"transformation for mesh [{i}]:\n{cfg['mesh'][i].transformations.optix_srt_data()}")

                mesh.begin_time = cfg['mesh'][i].transformations.begin_time
                mesh.end_time   = cfg['mesh'][i].transformations.end_time
                optix_srt_data_vec = np.array(cfg['mesh'][i].transformations.optix_srt_data(), copy=False, dtype=np.float32, order='C')
                mesh.transformation_keys.resize(cfg['mesh'][i].transformations.optix_srt_data().shape[0])
                for i in range(mesh.transformation_keys.size()):
                    mesh.transformation_keys[i].resize(16)
                    memcpy(mesh.transformation_keys[i].data(), &optix_srt_data_vec[i, 0], 16*4) # ctypes.sizeof(float)

                    # for q in range(16):
                    #     logger.info(f'srt element {q}: {mesh.transformation_keys[i][q]}')

            logger.info(f'Mesh with {mesh.transformation_keys.size()} transformation keys is ready...')
            mesh_vec.push_back(mesh)
    else:
        mesh.attenuation      = cfg['mesh'].attenuation
        mesh.refractive_index = cfg['mesh'].refractive_index
        mesh.transformation_keys.resize(0)

        vertices_id = cfg['mesh'].vertices
        faces_id    = cfg['mesh'].faces
        normals_id  = cfg['mesh'].normals

        if(vertices_id != -1):
            cuVertices = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(vertices_id))
            mesh.vertices  = cuVertices.dataPtr()
            mesh.nVertices = <size_t>(cuVertices.size() / 3)
            mesh.nFaces    = <size_t>(cuVertices.size() / 9 ) ### Are you sure of this?
            print(f"mesh.nVertices: {mesh.nVertices}")
        if(faces_id != -1):
            cuFaces    = cast.dynamic_cast[cuManagedArrayInt_ptr]   (pObjectManager.get_cuda_array(faces_id))
            mesh.faces     = cuFaces.dataPtr()
            mesh.nFaces    = <size_t>( cuFaces.size() / 3 )
        if(normals_id != -1):
            cuNormals  = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(normals_id))
            mesh.normals   = cuNormals.dataPtr()

        if(cfg['mesh'].transformations is not None):
            mesh.begin_time = cfg['mesh'].transformations.begin_time
            mesh.end_time   = cfg['mesh'].transformations.end_time
            optix_srt_data_vec = np.array(cfg['mesh'].transformations.optix_srt_data(), copy=False, dtype=np.float32, order='C')
            mesh.transformation_keys.resize(cfg['mesh'].transformations.optix_srt_data().shape[0])
            for i in range(mesh.transformation_keys.size()):
                mesh.transformation_keys[i].resize(16)
                memcpy(mesh.transformation_keys[i].data(), &optix_srt_data_vec[i, 0], 16*4)

        mesh_vec.push_back(mesh)

    cuProjector.set_geometry(proj_geom)
    # logger.info(f'Size of mesh_vec: {mesh_vec.size()}')
    cuProjector.set_mesh(mesh_vec)

    cdef int sino_id = cfg['sino']
    cdef cuManagedArrayFloat *cuSino = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(sino_id))
    cdef ProjectorOutBuffers bufs
    bufs.pSino_device = cuSino.dataPtr()


    cuProjector.set_outbuffers( bufs )
    _configure_projector(cuProjector, cfg)  # The projector HAS to be configured before being initialized
    cuProjector.initialize()
    
    return id

cdef _configure_projector(CProjector *cuProjector, cfg):
    projector_detector_policies_map = {'max_attenuation'     : OptiXMaxAttenuationPolicy,
                                       'sum_intensity'       : OptiXSumIntencityPolicy,
                                       'sum_share_intensity' : OptiXSumShareIntencityPolicy}

    projector_tracer_policies_map = {'recursive'     : OptiXRecursivePolicy,
                                     'non-recursive' : OptiXNonRecursivePolicy}

    cdef CProjectorConfiguration proj_cfg

    proj_cfg.rays_row_count = cfg.get('rays_row_count', -1)
    proj_cfg.rays_col_count = cfg.get('rays_col_count', -1)

    proj_cfg.detector_policy_index = projector_detector_policies_map.get(cfg.get('detector_pixel_policy', 'max_attenuation'), 2)
    proj_cfg.tracer_index          = projector_tracer_policies_map.get  (cfg.get('tracer_policy', 'recursive'), OptiXRecursivePolicy)
    
    cuProjector.configure(proj_cfg)

def configure_projector(id, cfg):
    if id == INVALID_ID:
        return
    
    cdef CProjector *cuProjector = pObjectManager.get_projector(id)
    _configure_projector(cuProjector, cfg)

def delete_projector_c(id):
    logger = log.logging.getLogger(log.main_logger_name())
    if id != INVALID_ID:
        pObjectManager.delete_projector(id)
    else:
        logger.info("delete_projector() :: Unable to delete projector with invalid ID -1")

def run_c(id):
    logger = log.logging.getLogger(log.main_logger_name())
    cdef CProjector *cuProjector = pObjectManager.get_projector(id)
    logger.info("Running projector...")
    cuProjector.run()