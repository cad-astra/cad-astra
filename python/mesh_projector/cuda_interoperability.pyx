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

import ctypes, os
from mesh_projector.objects_manager cimport *
from libc.stdint cimport intptr_t 
from libcpp cimport bool
from libcpp cimport cast
from libcpp.string cimport string
from mesh_projector import log

CUDA_ARRAY_INTERFACE_EXPORT_VERSION = int(os.environ.get('CUDA_ARRAY_INTERFACE_EXPORT_VERSION', 3)) # controls the version of the CAI


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
        float* dataPtr() except +
    cdef cppclass cuManagedArrayInt:
        void downloadToHost(int *, size_t) except +
        void initialize(int) except +
        size_t size()
        void assignScalar(int value) except +
        void assignArray(int*, size_t) except +
        int* dataPtr() except +
    cdef cppclass cuManagedArrayFloatFactory:
        cuManagedArrayFloatFactory(size_t, float)
        cuManagedArrayFloatFactory(size_t, float *, bool)
        # CBaseCudaArray *create_object()
    cdef cppclass cuManagedArrayIntFactory:
        cuManagedArrayIntFactory(size_t, int)
        cuManagedArrayIntFactory(size_t, int *, bool)
        # CBaseCudaArray *create_object() except +

ctypedef CBaseCudaArray*  cuBaseArray_ptr
ctypedef cuManagedArrayFloat* cuManagedArrayFloat_ptr
ctypedef cuManagedArrayInt* cuManagedArrayInt_ptr


cdef class cuda_interoperability_obj_c:

    cdef:
        intptr_t data_ptr
        tuple shape
        str dtype_str
        tuple strides
        bool _c_contiguous
        int obj_id
        int size

    def __cinit__(self, obj_id):
        self.obj_id = obj_id
        cdef CBaseCudaArray *cuArray = cast.dynamic_cast[cuBaseArray_ptr] (pObjectManager.get_cuda_array(obj_id))
        #TODO: support for other types
        cdef cuManagedArrayInt *cuArray_int 
        cdef cuManagedArrayFloat *cuArray_flt
        cdef intptr_t pointer 

        dtype_str_mp = cuArray.dtype_str() 
        if dtype_str_mp == string(b'int'):
            cuArray_int = cast.dynamic_cast[cuManagedArrayInt_ptr] (cuArray)
            pointer = <intptr_t>cuArray_int.dataPtr() 
            dtype_str = "<i4" #FIXME int may be 8 bytes and the ordering of the array may be different
        elif dtype_str_mp == string(b'float'): 
            cuArray_flt = cast.dynamic_cast[cuManagedArrayFloat_ptr] (cuArray)
            pointer = <intptr_t>cuArray_flt.dataPtr() 
            dtype_str = "<f4" #FIXME float may be 8 bytes and the ordering of the array may be different
        else:
            raise ValueError('Unsupported dtype')
        #TODO: add logging?

        self.data_ptr = pointer
        self.size = cuArray.size()
        self.shape = (self.size,) # size_t size_t # TODO: support for other shapes when it will be implemented in cuManagerArray
        self.dtype_str = dtype_str
        self.strides = None # TODO: support for other strides when it will be implemented in cuManagerArray


    @property
    def __cuda_array_interface__(self):
        cdef dict desc = {
            'shape': self.shape,
            'typestr': self.dtype_str,
            'descr': [('', self.dtype_str)] # we do not support structured arrays
        }
        cdef int ver = CUDA_ARRAY_INTERFACE_EXPORT_VERSION

        if ver == 3:
            desc['stream'] = None # No stream support yet in mesh_projector
        elif ver == 2:
            # (prior to CAI v3): stream sync is explicitly handled by users.
            pass
        else:
            raise ValueError('CUDA_ARRAY_INTERFACE_EXPORT_VERSION can only be set to 3 or 2')

        desc['version'] = ver
        if self._c_contiguous:
            desc['strides'] = None
        else:
            desc['strides'] = self.strides
        if self.size > 0:
            desc['data'] = (self.data_ptr, False) # second flag is for read-only (specified in CAI)
        else:
            desc['data'] = (0, False)

        return desc

def from_cuda_array_c(interface: dict):
    cdef intptr_t data_ptr = interface['data'][0]
    #cdef size_t size     = interface['shape'][0]
    cdef tuple shape    = interface['shape'] 
    dtype_str = interface['typestr']
    #cdef tuple strides  = interface['strides'] # Not yet used by cuManagedArray
    cdef int version  = interface['version']

    #TODO: support for non-contiguous arrays

    #if len(shape) != 1: # Not yet used by cuManagedArray
    #    print("Warning: cuManagedArray supports only 1D arrays. Shape is ignored.") #FIXME: how to add the warning to be printed once only?
    cdef size_t size = 1
    for i in range(len(shape)):
        size *= shape[i]

    cdef int array_id
    cdef bool alloc = True 
    # TODO: add support for different devices 
    # TODO: allow contextual copy?
    cdef CudaArrayObjectFactory *pFactory

    if dtype_str == '<f4':
        pFactory = <CudaArrayObjectFactory *> new cuManagedArrayFloatFactory(size, <float *> data_ptr, alloc)
    elif dtype_str == '<i4':
        pFactory = <CudaArrayObjectFactory *> new cuManagedArrayIntFactory(size, <int *> data_ptr, alloc)
    else:
        raise ValueError('Unsupported dtype/endian-ness yet')

    array_id = pObjectManager.create_cuda_array(pFactory)
    return array_id