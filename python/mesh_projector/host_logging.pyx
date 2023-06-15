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

from mesh_projector import log

cdef extern from "host_logging.h" namespace "MeshFP":
    ctypedef enum LogLevel:
        DISABLE
        FATAL
        ERROR
        WARNING
        INFO
        DEBUG

    cdef cppclass Logging:
        @staticmethod
        void set_level(LogLevel)
        @staticmethod
        void configure_format(int)

host_logging_levels_map = {
    'disable'    : DISABLE,
    'fatal'      : FATAL,
    'error'      : ERROR,
    'warning'    : WARNING,
    'info'       : INFO,
    'debug'      : DEBUG
}

def set_host_log_level_c(level: str) -> None:
    logger = log.logging.getLogger(log.main_logger_name())
    if not level in host_logging_levels_map.keys():
        logger.info(f"Ignored unknown logging level: {level}. Valid values: {set(host_logging_levels_map.keys())}.")
        return
    else:
        logger.info(f"Setting host logging level to {level}.")
        Logging.set_level(host_logging_levels_map[level])

def sef_format_c(fmt: int) -> None:
    Logging.configure_format(fmt)

# def get_host_log_level_c() -> int:
#     return CGlobalConfig.get_optix_log_level()