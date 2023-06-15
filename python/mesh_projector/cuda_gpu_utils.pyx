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
cimport numpy as np
from libcpp.string cimport string

from mesh_projector import log

cdef extern from "gpu_utils.h":
    void getDeviceCount(int)           except +
    void getDeviceName(string &, int ) except +
    void setDevice(int)                except +

from mesh_projector.objects_manager cimport *

def set_gpu_index_c(int gpu_idx=0):
    logger = log.logging.getLogger(log.main_logger_name())
    if pObjectManager.size_cuda_arrays() != 0:
        logger.warning("There are still CUDA arrays stored on the active GPU. GPU index will not be changed.")
        return
    
    if pObjectManager.size_projector() != 0:
        logger.warning("There are still CUDA projectors on the active GPU. GPU index will not be changed.")
        return

    cdef int device_count = 0
    
    getDeviceCount(device_count)

    logger.info(f"Total GPUs visible: {device_count}")

    cdef string device_name
    if(gpu_idx < device_count):
        getDeviceName(device_name, gpu_idx)
        setDevice(gpu_idx)
        logger.info(f"Switched to GPU [{gpu_idx}]:  {device_name}")
    else:
        error_msg = f"Could not change GPU index to {gpu_idx}: only GPUs with indices [0"
        
        if(device_count > 1):
            error_msg += "-{device_count-1}] are available"
        else:
            error_msg += "] are available"

        raise ValueError(error_msg)