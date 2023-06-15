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


import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.type_check import common_type
cimport numpy as np
from libcpp cimport bool
from libcpp cimport cast
from libcpp.string cimport string
import ctypes

from mesh_projector.objects_manager cimport *
from mesh_projector import log

cdef extern from "CBaseCudaArray.h":
    cdef cppclass CBaseCudaArray:
        string dtype_str()
        size_t size()

cdef extern from "cuda/cuManagedArray.cuh":
    cdef cppclass cuManagedArrayFloat:
        void downloadToHost(float *, size_t) except +
        void initialize(float) except +
        size_t size()
        void assignScalar(float value) except +
        void assignArray(float*, size_t) except +
    cdef cppclass cuManagedArrayInt:
        void downloadToHost(int *, size_t) except +
        void initialize(int) except +
        size_t size()
        void assignScalar(int value) except +
        void assignArray(int*, size_t) except +
    cdef cppclass cuManagedArrayFloatFactory:
        cuManagedArrayFloatFactory(size_t, float)
        cuManagedArrayFloatFactory(size_t, float *)
        # CBaseCudaArray *create_object()
    cdef cppclass cuManagedArrayIntFactory:
        cuManagedArrayIntFactory(size_t, int)
        cuManagedArrayIntFactory(size_t, int *  )
        # CBaseCudaArray *create_object() except +

ctypedef CBaseCudaArray*  cuBaseArray_ptr
ctypedef cuManagedArrayFloat* cuManagedArrayFloat_ptr
ctypedef cuManagedArrayInt* cuManagedArrayInt_ptr

# cdef CObjectManager *pObjectManager = CObjectManager.getSingletonPtr(string(b"cuda_managed_array.pyx"))

INVALID_ID = -1

# TODO: pass numpy array as Python object, based on shape and dtype call the corresponding Factory
# TODO: use IntEnum class for exporting options for projector array types

def create_normals_c(np.ndarray[np.float32_t, ndim=2, mode='c'] nd_normals not None):
    if nd_normals.shape[1] != 3:
        raise ValueError(f"Error: shape of array of N normals must be (N, 3), but ({nd_normals.shape[0]}, {nd_normals.shape[1]}) given")

    cdef CudaArrayObjectFactory *pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(<size_t>(nd_normals.size), <float *> nd_normals.data)
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id

def create_vertices_c(np.ndarray[np.float32_t, ndim=2, mode='c'] nd_vertices not None):
    if nd_vertices.shape[1] != 3:
        raise ValueError(f"Error: shape of array of N vertices must be (N, 3), but ({nd_vertices.shape[0]}, {nd_vertices.shape[1]}) given")

    cdef CudaArrayObjectFactory *pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(<size_t>(nd_vertices.size), <float *> nd_vertices.data)
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id

def create_faces_c(np.ndarray[np.int32_t, ndim=2, mode='c'] nd_faces not None):
    if nd_faces.shape[1] != 3:
        raise ValueError(f"Error: shape of array of N faces must be (N, 3), but ({nd_faces.shape[0]}, {nd_faces.shape[1]}) given")

    cdef CudaArrayObjectFactory *pFactory = <CudaArrayObjectFactory *> new cuManagedArrayIntFactory(<size_t>(nd_faces.size), <int *> nd_faces.data)
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id

def create_angles_c(np.ndarray[np.float32_t, ndim=1, mode='c'] nd_angles not None):
    cdef CudaArrayObjectFactory *pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(<size_t>(nd_angles.size), <float *> nd_angles.data)
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id

def create_views_c(np.ndarray[np.float32_t, ndim=2, mode='c'] nd_views not None):
    if nd_views.shape[1] != 12:
        raise ValueError(f"Error: shape of array of N views must be (N, 12), but ({nd_views.shape[0]}, {nd_views.shape[1]}) given")

    cdef CudaArrayObjectFactory *pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(<size_t>(nd_views.size), <float *> nd_views.data)
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id

def create_triangles_c(np.ndarray[np.float32_t, ndim=2, mode='c'] nd_triangles not None):
    if nd_triangles.shape[1] != 9:
        raise ValueError(f"Error: shape of array of N triangles must be (N, 9), but ({nd_triangles.shape[0]}, {nd_triangles.shape[1]}) given")

    cdef CudaArrayObjectFactory *pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(<size_t>(nd_triangles.size), <float *> nd_triangles.data)
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id

def create_sino_c(shape):
    if len(shape) != 3:
        raise ValueError(f"Error: dimension of sino array must be 3, but ({shape}) given")

    cdef float init_value = 0.0
    cdef CudaArrayObjectFactory *pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(<size_t>(shape[0]*shape[1]*shape[2]), init_value)
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id


def create_arbitrary_c(shape, dtype):
    size = 1
    if shape is not None:
        for dim in shape:
            size *= dim
    else:
        raise ValueError("Error: shape cannot be None")
    cdef CudaArrayObjectFactory *pFactory
    # TODO(pavel): check int data types - should we leave only int32?
    if (dtype == np.int) or (dtype == int) or (dtype == np.int32):
        pFactory = <CudaArrayObjectFactory *> new cuManagedArrayIntFactory(<size_t>(size), 0)
    elif dtype is np.float or dtype is np.float32:
        pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(<size_t>(size), 0)
    else:
        raise ValueError(f"Error: cannot create cuda array of dtype={dtype}")
    cdef int id = pObjectManager.create_cuda_array(pFactory)
    return id

def reallocate_cuda_array_c(array_id, array_type, **kwargs):
    """
    delete an existing array object and create one of the same type with kwargs determining how
    this function returns a new id of the freshly allocated object
    """
    if array_id != INVALID_ID:
        pObjectManager.delete_cuda_array(array_id)
    return create_cuda_array_c(array_type, **kwargs)


def get_size_c(array_id):
    if array_id == INVALID_ID:
        return -1

    cdef cuManagedArrayFloat *cuArrayFloat = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(array_id))
    if cuArrayFloat:
        return cuArrayFloat.size()

    cdef cuManagedArrayInt *cuArrayInt = cast.dynamic_cast[cuManagedArrayInt_ptr] (pObjectManager.get_cuda_array(array_id))
    if cuArrayInt:
        return cuArrayInt.size()
    else:
        raise ValueError("Error: something undefined happened when trying to retrieve the array size!")

def create_cuda_array_c(type, **kwargs):
    id = INVALID_ID
    nd_array = kwargs.get('nd_array')
    shape =  kwargs.get('shape')
    dtype = kwargs.get('dtype')
    if shape is None:
        if nd_array is None:
            raise ValueError("Must either provide 'shape' or 'nd_array' for creating a new array")
        shape = nd_array.shape

    if type == 'normals':
        id = create_normals_c(nd_array)
    elif type == 'vertices':
        id = create_vertices_c(nd_array)
    elif type == 'faces':
        id = create_faces_c(nd_array)
    elif type == 'angles':
        id = create_angles_c(nd_array)
    elif type == 'views':
        id = create_views_c(nd_array)
    elif type == 'sino':
        id = create_sino_c(shape)
    elif type == 'triangles':
        id = create_triangles_c(nd_array)
    elif type == 'arbitrary':
        id = create_arbitrary_c(shape, dtype)
    else:
        raise ValueError(f"Error: unknown array type {type}")

    return id

# Done:
# TODO: implement array.store(array_id, nd.array) method without reallocation in case shape and dtype match
# ------------------------------------------------------------------------
# TODO: consider bool parameter ForceReallocation (maybe with default to True?)
# TODO: the store function needs the following features to be a bit more conenient:
#       -   identification of the array type for reallocation (sino, vertices, etc.)
#       -   reallocation on same object without changing the ID
#       -   automatic array type pointer identification, so we dont have to dynamic_cast multiple times?!
def store_c(array_id, new_elements):
    """
    Store data into an existing cuda array.

    Parameters:
    -----------
    array_id : int
        Cuda array id.
    
    new_elements : scalar or ndarray
        Data to store. If new_elements is scalar, then cuda array is filled with the scalar value.
        if new_elements is ndarray, then cuda array is overwritten with the data from new_elements.
        In case new_elements.size != cuda array size, memory is reallocated for the latter one.

    Returns:
    --------
    array_id : int
        Id of the updated cuda array. Differs from the input id only when cuda array is reallocated.
    """
    if array_id == INVALID_ID:
        raise ValueError("store() :: Invalid ID!")

    # TODO(pavel): maybe catch exception that object manager throws in case there's no object with the id specified?
    cdef CBaseCudaArray *cuArray = cast.dynamic_cast[cuBaseArray_ptr] (pObjectManager.get_cuda_array(array_id))

    cdef cuManagedArrayInt *cuArray_int
    cdef cuManagedArrayFloat *cuArray_flt
    cdef np.ndarray[ndim=1, dtype=int] nd_elements_int
    cdef np.ndarray[ndim=1, dtype=np.float32_t] nd_elements_float

    if cuArray.dtype_str() == string(b"int"):
        cuArray_int = cast.dynamic_cast[cuManagedArrayInt_ptr] (cuArray)
        if not isinstance(new_elements, np.ndarray):
            cuArray_int.assignScalar(int(new_elements))
        else:
            nd_elements_int = np.array(new_elements, copy=False, dtype=ctypes.c_int, order='C').reshape(-1)
            if new_elements.size != cuArray_int.size():
                pObjectManager.delete_cuda_array(array_id)
                array_id = create_arbitrary_c(new_elements.shape, nd_elements_int.dtype)
                cuArray_int = cast.dynamic_cast[cuManagedArrayInt_ptr] (pObjectManager.get_cuda_array(array_id))
            cuArray_int.assignArray(&nd_elements_int[0], nd_elements_int.size)

    elif cuArray.dtype_str() == string(b"float"):
        cuArray_flt = cast.dynamic_cast[cuManagedArrayFloat_ptr] (cuArray)
        if not isinstance(new_elements, np.ndarray):
            cuArray_flt.assignScalar(float(new_elements))
        else:
            nd_elements_float = np.array(new_elements, copy=False, dtype=np.float32, order='C').reshape(-1)
            if new_elements.size != cuArray_flt.size():
                pObjectManager.delete_cuda_array(array_id)
                array_id = create_arbitrary_c(new_elements.shape, nd_elements_float.dtype)
                cuArray_flt = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(array_id))
            cuArray_flt.assignArray(&nd_elements_float[0], nd_elements_float.size)

    else:
        raise RuntimeError(f"Error: could not store cuda array of data type {cuArray.dtype_str()} into host numpy array")

    return array_id

def delete_all_cuda_arrays_c():
    pObjectManager.clear_cuda_arrays()

def delete_cuda_array_c(int id):
    if id != INVALID_ID:
        pObjectManager.delete_cuda_array(id)
    else:
        logger = log.logging.getLogger(log.main_logger_name())
        logger.warning("delete_cuda_array() :: Unable to delete array with invalid ID -1")

def initialize_sino_c(int id, float value):
    if id == INVALID_ID:
        raise ValueError("initialize_sino() :: Invalid ID!")

    cdef cuManagedArrayFloat *cuArray = cast.dynamic_cast[cuManagedArrayFloat_ptr] (pObjectManager.get_cuda_array(id))
    cuArray.initialize(value)

# TODO: make get cuda array type independent with - DONE
def get_cuda_array_c(int id):
    if id == INVALID_ID:
        raise ValueError("get_cuda_array() :: Invalid ID!")

    cdef CBaseCudaArray *cuArray = cast.dynamic_cast[cuBaseArray_ptr] (pObjectManager.get_cuda_array(id))

    cdef cuManagedArrayInt *cuArray_int
    cdef cuManagedArrayFloat *cuArray_flt
    cdef np.ndarray[dtype=int, ndim=1, mode='c'] nd_sino_int = np.empty((cuArray.size(),), dtype=ctypes.c_int)
    cdef np.ndarray[dtype=np.float32_t, ndim=1, mode='c'] nd_sino_float = np.empty((cuArray.size(),), dtype=np.float32)

    if cuArray.dtype_str() == string(b"int"):
        cuArray_int = cast.dynamic_cast[cuManagedArrayInt_ptr] (cuArray)
        cuArray_int.downloadToHost(&nd_sino_int[0], cuArray_int.size())
        return nd_sino_int
    elif cuArray.dtype_str() == string(b"float"):   
        cuArray_flt = cast.dynamic_cast[cuManagedArrayFloat_ptr] (cuArray)
        cuArray_flt.downloadToHost(&nd_sino_float[0], cuArray_flt.size())
        return nd_sino_float
    else:
        raise RuntimeError(f"Error: could not store cuda array of data type {cuArray.dtype_str()} into host numpy array")
        # return np.empty((cuArray.size()))

