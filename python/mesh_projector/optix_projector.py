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

from .cuda_optix_projector_routines import *
from .cuda_managed_array      import *

from .projection_geometry import *

from collections.abc import Iterable

def _project_triangles_optix(triangle_id,
                             sino_id,
                             proj_geometry: ProjectionGeometry,
                             attenuation,
                             mesh_refractive_index,
                             media_refractive_index,
                             max_depth,
                             rays_row_count,
                             rays_col_count,
                             phase_contrast_imaging=False,
                             sample_mask=None, detector_mask=None):
    if proj_geometry.geom_type != 'cone_vec':
        raise ValueError("triangle projection not (yet) supported for non cone_vec geometry")
    project_optix_triangle_mesh_cone_vec_cu(triangle_id, sino_id,
                                            proj_geometry.det_row_count, proj_geometry.det_col_count,
                                            proj_geometry.vectors,
                                            float(attenuation),
                                            float(mesh_refractive_index),
                                            float(media_refractive_index),
                                            max_depth, rays_row_count, rays_col_count,
                                            phase_contrast_imaging, sample_mask, detector_mask)

def project_optix(mesh, sino_id,
                  proj_geometry: ProjectionGeometry,
                  attenuation=1.0,
                  mesh_refractive_index=1.0,
                  media_refractive_index=1.0,
                  max_depth=31,
                  rays_row_count=-1, rays_col_count=-1,
                  phase_contrast_imaging=False,
                  sample_mask=None, detector_mask=None):
    if isinstance(mesh, Iterable):
        if len(mesh) == 1:
            _project_triangles_optix(*mesh, sino_id, proj_geometry, attenuation, mesh_refractive_index, media_refractive_index, max_depth, rays_row_count, rays_col_count, phase_contrast_imaging, sample_mask, detector_mask)
    else:
        _project_triangles_optix(mesh, sino_id, proj_geometry, attenuation, mesh_refractive_index, media_refractive_index, max_depth, rays_row_count, rays_col_count, phase_contrast_imaging, sample_mask, detector_mask)

def compute_raypath_optix(  triangle_id,
                            ray_paths_id,
                            ray_depths_id,
                            ray_intens_id,
                            proj_geometry: ProjectionGeometry,
                            attenuation=1.0,
                            mesh_refractive_index =1.0,
                            media_refractive_index=1.0,
                            rays_row_count=-1, rays_col_count=-1,
                            phase_contrast_imaging=False,
                            sample_mask=None, detector_mask=None):
    if proj_geometry.geom_type != 'cone_vec':
        raise ValueError("triangle projection not (yet) supported for non cone_vec geometry")
    
    compute_raypath_optix_triangle_mesh_cone_vec_cu(triangle_id, ray_paths_id, ray_depths_id, ray_intens_id,
                                                    proj_geometry.det_row_count, proj_geometry.det_col_count,
                                                    proj_geometry.vectors,
                                                    float(attenuation),
                                                    float(mesh_refractive_index),
                                                    float(media_refractive_index),
                                                    rays_row_count, rays_col_count,
                                                    phase_contrast_imaging, sample_mask, detector_mask)
