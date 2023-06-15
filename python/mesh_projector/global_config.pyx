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
from numpy.lib.type_check import common_type
cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from collections.abc import Iterable
import ctypes

from mesh_projector.objects_manager cimport *

cdef extern from "global_config.h" namespace "MeshFP":
    cdef cppclass CGlobalConfig:
        @staticmethod
        void set_ptx_dir(string)
        @staticmethod
        string get_ptx_dir()
        @staticmethod
        void set_optix_log_level(int)
        @staticmethod
        int get_optix_log_level()

def set_global_ptx_dir_c(ptx_dir: str) -> None:
    CGlobalConfig.set_ptx_dir(ptx_dir.encode("UTF-8"))

def get_global_ptx_dir_c() -> str:
    return CGlobalConfig.get_ptx_dir().encode("UTF-8")

def set_global_optix_log_level_c(level: int) -> None:
    CGlobalConfig.set_optix_log_level(level)

def get_global_optix_log_level_c() -> int:
    return CGlobalConfig.get_optix_log_level()