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


# Declarations related to object manager

cdef extern from "CBaseCudaArray.h":
    cdef cppclass CBaseCudaArray

cdef extern from "CProjector.h":
    cdef cppclass CProjector

#cdef extern from "CObjectFactory.h":
#    cdef cppclass CObjectFactory[T]

#ctypedef CObjectFactory[CBaseCudaArray] *  CudaArrayObjectFactory
#ctypedef CObjectFactory[CProjector]     *  GPUProjectorObjectFactory

cdef extern from "object_manager.h":
    cdef cppclass CudaArrayObjectFactory
    cdef cppclass GPUProjectorObjectFactory
 
    cdef cppclass CObjectManager:
        int create_cuda_array(CudaArrayObjectFactory     *) except +
        int create_projector (GPUProjectorObjectFactory  *) except +
        CBaseCudaArray *get_cuda_array(int) except +
        CProjector     *get_projector (int) except +
        size_t size_cuda_arrays()
        size_t size_projector  ()
        void clear_cuda_arrays()
        void clear_projectors ()
        void delete_cuda_array(int) except +
        void delete_projector (int) except +
        @staticmethod
        CObjectManager *getSingletonPtr()

    cdef extern CObjectManager *pObjectManager