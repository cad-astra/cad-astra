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

from .cuda_managed_array import *

def create_cuda_array(type: str, **kwargs):

    return create_cuda_array_c(type, **kwargs)

def store(array_id: int, new_elements):
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
    return store_c(array_id, new_elements)


def delete_all_cuda_arrays():
    delete_all_cuda_arrays_c()

def delete_cuda_array(array_id: int):
    delete_cuda_array_c(array_id)
    


def initialize_sino(array_id: int, value: float):
    initialize_sino_c(array_id, value)

def reallocate_cuda_array(array_id: int, array_type, **kwargs):
    return reallocate_cuda_array_c(array_id, array_type, **kwargs)

# TODO: make get cuda array type independent with - DONE
def get_cuda_array(array_id: int):
    return get_cuda_array_c(array_id)

def get_size(array_id: int):
    return get_size_c(array_id)

from .cuda_interoperability import cuda_interoperability_obj_c, from_cuda_array_c
def get_cuda_obj(array_id: int):
    '''
    Obtain a cuda array that implements the __cuda_array_interface__ from a (mesh-projector) cuda array id.

    Parameters:
    -----------
    array_id : int
        Id of the mesh-projector cuda array.

    Returns:
    --------
    obj : 
        Cuda array that implements the __cuda_array_interface__.
    '''
    return cuda_interoperability_obj_c(array_id)


def from_cuda_array(cuda_array_obj):
    '''
    Obtain a (mesh-projector) cuda array id from a cuda array object that implements the __cuda_array_interface__.

    Parameters:
    -----------
    cuda_array_obj : 
        Cuda array that implements the __cuda_array_interface__.

    Returns:
    --------
    array_id : int
        Id of the mesh-projector cuda array. 
    '''    
    return from_cuda_array_c(cuda_array_obj.__cuda_array_interface__)