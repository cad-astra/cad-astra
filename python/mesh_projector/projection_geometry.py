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

from .array import *

class ProjectionGeometry:
    def __init__(self, type: str, detector_geom: str, det_radius: float):
        self.__type = type
        self.__det_geom = detector_geom
        self.__det_radius = det_radius
    
    # TODO: change this property to name
    @property
    def type(self):
        return self.__type

    @property
    def detector_geometry(self):
        return self.__det_geom

    @property
    def detector_cylinder_radius(self):
        return self.__det_radius

class ConeGeometry(ProjectionGeometry):
    def __init__(self, det_spacing_x: np.float, det_spacing_y: np.float,
                 det_row_count: np.int, det_col_count: np.int,
                 angles: np.int or np.ndarray, source_origin: np.float, origin_det: np.float,
                 detector_geom: str, detector_cylinder_radius:float):
        super().__init__(type='cone', detector_geom=detector_geom, det_radius=detector_cylinder_radius)
        self._det_spacing_x = det_spacing_x
        self._det_spacing_y = det_spacing_y
        self._det_row_count = det_row_count
        self._det_col_count = det_col_count
        if type(angles) is int:
            self._angles_id = angles
            self._angles = None
        else:
            self._angles = angles
            self._angles_id = create_cuda_array('angles', nd_array=np.asarray(angles, dtype=np.float32))

        self._source_origin = source_origin
        self._det_origin    = origin_det

    @property
    def det_spacing_x(self):
        return self._det_spacing_x

    @property
    def det_spacing_y(self):
        return self._det_spacing_y

    @property
    def det_row_count(self):
        return self._det_row_count

    @property
    def det_col_count(self):
        return self._det_col_count

    @property
    def angles(self):
        if self._angles is not None:
            return self._angles
        else:
            angles = get_cuda_array(self._angles_id)
            return angles

    @property
    def angles_id(self):
        return self._angles_id

    @property
    def source_origin(self):
        return self._source_origin

    @property
    def det_origin(self):
        return self._det_origin

    def astra_proj_geom(self):
        return {
            'type': 'cone',
            'DetectorSpacingX': self.det_spacing_x,
            'DetectorSpacingY': self.det_spacing_y,
            'DetectorRowCount': self.det_row_count,
            'DetectorColCount': self.det_col_count,
            'ProjectionAngles': self.angles,
            'DistanceOriginSource': self.source_origin,
            'DistanceOriginDetector': self.det_origin
            }

class ConeVecGeometry(ProjectionGeometry):
    def __init__(self, det_row_count, det_col_count, vectors, detector_geom: str, detector_cylinder_radius:float):
        super().__init__(type='cone_vec', detector_geom=detector_geom, det_radius=detector_cylinder_radius)
        self._det_row_count = det_row_count
        self._det_col_count = det_col_count

        # if len(vectors.shape) != 2:
        #     raise ValueError("Error: vectors.shape != 2")
        # if vectors.shape[1] != 12:
        #     raise ValueError("Error: vectors.shape[1] != 12")
        # self._vectors = vectors
        if type(vectors) is int:
            self._vectors_id = vectors
            self._vectors = None
        else:
            self._vectors = vectors
            self._vectors_id = create_cuda_array('views', nd_array=np.asarray(vectors, dtype=np.float32))

    @property
    def det_row_count(self):
        return self._det_row_count

    @property
    def det_col_count(self):
        return self._det_col_count

    @property
    def vectors(self):
        if self._vectors is not None:
            return self._vectors
        else:
            vectors = get_cuda_array(self._vectors_id)
            return vectors

    @property
    def vectors_id(self):
        return self._vectors_id

    def astra_proj_geom(self):
        return {
            'type': 'cone_vec',
            'DetectorRowCount': self.det_row_count,
            'DetectorColCount': self.det_col_count,
            'Vectors': self.vectors
            }

class ConeMeshTransform(ProjectionGeometry):
    def __init__(self, det_spacing_x: np.float, det_spacing_y: np.float,
                 det_row_count: int, det_col_count: int,
                 view_keys: int or list,
                 projector_view, detector_geom: str, detector_cylinder_radius:float):
        super().__init__(type='cone_mesh_transform', detector_geom=detector_geom, det_radius=detector_cylinder_radius)
        self.__det_spacing_x = det_spacing_x
        self.__det_spacing_y = det_spacing_y
        self.__det_row_count = det_row_count
        self.__det_col_count = det_col_count
        self.__view_keys = view_keys
        self.__projector_view   = projector_view

    @property
    def det_spacing_x(self):
        return self.__det_spacing_x

    @property
    def det_spacing_y(self):
        return self.__det_spacing_y

    @property
    def det_row_count(self):
        return self.__det_row_count

    @property
    def det_col_count(self):
        return self.__det_col_count

    @property
    def view_keys(self):
        return self.__view_keys

    @property
    def projector_view(self):
        return self.__projector_view

    def astra_proj_geom(self):
        raise RuntimeError("This projection geometry does not have equal ASTRA geometry")

def projection_geometry_types():
    return ['cone', 'cone_vec', 'cone_mesh_transform']

def create_projection_geometry(geom_type, *args, **kwargs):
    """
    Create projection geometry.

    Parameters:
    -----------
    type : string
        Projection type, either 'cone' or'cone_vec'.
    
    det_row_count : int
        Number of detector rows.

    det_col_count : int
        Number of detector columns.

    # TODO: Fix this brief description with actual parameters
    vectors: nd.array
        Array with shape (N, 12) that defines N views,
        each view consists of 4 vectors augmented into
        12-element row: (Sx, Sy, Sz, Dx, Dy, Dz, Ux, Uy, Uz, Vx, Vy, Vz).

    Returns:
    --------
        Projection geometry object. 
    """
    options = {}

    if geom_type == 'cone_vec':
        nargin = len(args)
        options['det_row_count'] = args[0] if nargin > 0 else kwargs.get('det_row_count', 1)
        options['det_col_count'] = args[1] if nargin > 1 else kwargs.get('det_col_count', 1)
        # options['vectors']       = args[2] if nargin > 2 else kwargs.get('vectors', np.array([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0., 0., 1., 1., 0., 0.]]))
        options['vectors_id']     = args[2] if nargin > 2 else kwargs.get('vectors_id', -1)
        options['vectors_id']     = args[2] if nargin > 2 else kwargs.get('vectors_id', -1)
        options['detector_geometry']     = args[3] if nargin > 3 else kwargs.get('detector_geometry', 'plane')
        #TODO: do raduis sanity check!
        options['detector_cylinder_radius'] = args[4] if nargin > 4 else kwargs.get('detector_cylinder_radius', 1)
        return ConeVecGeometry(options['det_row_count'], options['det_col_count'], options['vectors_id'], options['detector_geometry'], options['detector_cylinder_radius'])
    elif geom_type == 'cone':
        nargin = len(args)
        options['det_spacing_x'] = args[0] if nargin > 0 else kwargs.get('det_spacing_x', 1)
        options['det_spacing_y'] = args[1] if nargin > 1 else kwargs.get('det_spacing_y', 1)
        options['det_row_count'] = args[2] if nargin > 2 else kwargs.get('det_row_count', 1)
        options['det_col_count'] = args[3] if nargin > 3 else kwargs.get('det_col_count', 1)
        # options['angles']        = args[4] if nargin > 4 else kwargs.get('angles', 1)
        options['angles_id']     = args[4] if nargin > 4 else kwargs.get('angles_id', -1)
        options['source_origin'] = args[5] if nargin > 5 else kwargs.get('source_origin', 1)
        options['origin_det']    = args[6] if nargin > 6 else kwargs.get('origin_det', 1)

        options['detector_geometry']     = args[7] if nargin > 7 else kwargs.get('detector_geometry', 'plane')
        #TODO: do raduis sanity check!
        options['detector_cylinder_radius'] = args[8] if nargin > 8 else kwargs.get('detector_cylinder_radius', 1)

        return ConeGeometry(options['det_spacing_x'], options['det_spacing_y'],
                            options['det_row_count'], options['det_col_count'],
                            options['angles_id'],
                            options['source_origin'], options['origin_det'],
                            options['detector_geometry'],
                            options['detector_cylinder_radius']
                            )
    elif geom_type == 'cone_mesh_transform':
        nargin = len(args)
        options['det_spacing_x'    ] = args[0] if nargin > 0 else kwargs.get('det_spacing_x',     1)
        options['det_spacing_y'    ] = args[1] if nargin > 1 else kwargs.get('det_spacing_y',     1)
        options['det_row_count'    ] = args[2] if nargin > 2 else kwargs.get('det_row_count',     1)
        options['det_col_count'    ] = args[3] if nargin > 3 else kwargs.get('det_col_count',     1)
        options['view_keys'        ] = args[4] if nargin > 4 else kwargs.get('view_keys',         1)
        options['projector_view'   ] = args[5] if nargin > 5 else kwargs.get('projector_view',    1)

        options['detector_geometry']     = args[6] if nargin > 6 else kwargs.get('detector_geometry', 'plane')
        #TODO: do raduis sanity check!
        options['detector_cylinder_radius'] = args[7] if nargin > 7 else kwargs.get('detector_cylinder_radius', 1)

        return ConeMeshTransform(options['det_spacing_x'], options['det_spacing_y'],
                                 options['det_row_count'], options['det_col_count'],
                                 options['view_keys'],
                                 options['projector_view'],
                                 options['detector_geometry'],
                                 options['detector_cylinder_radius'],
                                 )
    else:
        raise ValueError(f"Error: geometry {geom_type} is not implemeted yet")