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

from collections.abc import Iterable

from . import log

log.setup_logger(name=log.main_logger_name())

from .projection_geometry import *
from .mesh import *
from .projector import *
from .array import *
from .transformations import *
from .cuda_interoperability import *
from . import utils
from . import global_config as _globc
from . import host_logging as _hlg

__version__ = '1.1.0dev'

# All the OptiX PTX files are in the package root directory
__init_file_dir__ = '/'.join(__file__.split('/')[ :-1])

if __init_file_dir__ == '':
    __init_file_dir__ = '.'

_globc.set_global_ptx_dir_c(__init_file_dir__)

def debug_levels_py():
    return log.levels_dict

def set_logging_level_py(level: str):
    """
    Set logging level for the python wrapper.
    Possible values are "debug", "info", "warning", "error", "critical".
    # TODO: merge this logging level into the generic host logging.
    """
    logger = log.logging.getLogger(log.main_logger_name())
    if not level in log.levels_dict.keys():
        print(f"Ignored unknown logging level: {level}. Valid values: {set(log.levels_dict.keys())}.")
        return
    else:
        logger.info(f"Setting Python log level to {level}")
        logger.setLevel(log.levels_dict[level])

def set_logging_level_optix(level: str):
    """
    Set logging level for OptiX.
    Possible values are "disabled", "fatal", "error", "warning", "print".
    """
    if not level in log.optix_levels_dict.keys():
        print(f"Ignored unknown logging level for OptiX: {level}. Valid values: {set(log.optix_levels_dict.keys())}.")
        return
    else:
        logger = log.logging.getLogger(log.main_logger_name())
        logger.info(f"Setting OptiX log level to {level}")
        _globc.set_global_optix_log_level_c(log.optix_levels_dict.get(level))

def get_logging_level_optix() -> str:
    """
    Current OptiX logging level.
    """
    return log.optix_levels_dict[_globc.get_global_optix_log_level()]

def set_logging_level_host(level: str):
    """
    Set logging level for the host code.
    Possible values are "disabled", "fatal", "error", "warning", "info", "debug".
    Note that debug messages are ignored unless the package is build in debug mode (i.e., debug_printout is set to "TRUE").
    """
    _hlg.set_host_log_level_c(level)

def set_logging_fmt_host(fmt: list or str):
    """
    Either an Iterable with "date_time", "level", "module",
    or "all", or "none", are expected.
    """
    if isinstance(fmt, list) or isinstance(fmt, tuple) or isinstance(fmt, set):
        fmt_bin = ("date_time" in fmt) | (("level" in fmt) << 1) | (("module" in fmt) << 2)
    elif fmt == "all":
        print("Logging: {fmt} == all")
        fmt_bin = 7
    elif fmt == "none":
        fmt_bin = 0
    else:
        raise ValueError("fmt must be either iterable (list, set, or tuple), or 'all', or 'none'.")
    _hlg.sef_format_c(fmt_bin)

def project(mesh, sino_id, proj_geometry: ProjectionGeometry, attn: float = 1.0, proj_type: str = 'raster'):
    """
    Project mesh with the specified projection geometry.

    Parameters:
    -----------
    mesh : any Iterable or int
        Mesh to project described by either an iterable object with length either 1 or 3,
        or an int. If mesh is Iterable and length == 1, then mesh[0] is an id of a
        cuda array 'triangles'. If mesh is Iterable and length == 3, then mesh[0] is an id
        for cuda array 'vertices', mesh[1] - id for 'faces', mesh[2] - id for normals.
        If mesh is int, then it treated as an id of a cuda array 'triangles'.

    sino_id : int
        Id of a cuda array 'sino'.

    proj_geometry : ProjectionGeometry
        Projection geometry object.

    attn:   float
        Attenuation value of the mesh to be projected. Optional (default = 1.0)
    """
    cfg = {}
    cfg['sino']     = sino_id
    if isinstance(mesh, Iterable):
        cfg['mesh'] = create_mesh(attn, 1.0, *mesh)
    else:
        cfg['mesh'] = create_mesh(attn, 1.0, mesh)
    cfg['geometry'] = proj_geometry
            
    proj_id = create_projector(proj_type, cfg)
    run(proj_id)
    delete_projector(proj_id)
    