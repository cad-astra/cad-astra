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

class Mesh:
    def __init__(self, attenuation: np.float32,
                       refractive_index: np.float32,
                       vertices_id: np.int32,
                       faces_id: np.int32,
                       normals_id: np.int32,
                       transformations=None) -> None:
        self._attenuation = attenuation
        self._refractive_index = refractive_index
        self._vertices_id = vertices_id
        self._faces_id = faces_id
        self._normals_id = normals_id
        self._transformations = transformations
    
    @property
    def attenuation(self):
        return self._attenuation

    @property
    def refractive_index(self):
        return self._refractive_index

    @property
    def vertices(self):
        return self._vertices_id
    @property
    def faces(self):
        return self._faces_id
    @property
    def normals(self):
        return self._normals_id

    @property
    def transformations(self):
        return self._transformations

def create_mesh(*args, **kwargs):
    """
    Create a mesh container using either (vertices, faces, normals), or vertices only 
    for mesh definition. In the latter case, the vertices array shape is expected to be (nFaces, 9),
    i.e., each row containing all three vertices that define the face. Optioanlly, normal
    vectors can be provided. All arrays should be stored on GPU, only their IDs should be provided.

    Key-word arguments:
    ===================
    attenuation : float
        Attenuation coefficient of the mesh material.
    refractive_index : float
        Real part of the refractive index of the mesh material.
    vertices_id : int
        ID of the CUDA array of mesh vertices.
    faces_id : int
        ID of the CUDA array of mesh face indices.
    normals_id : int
        ID of the CUDA array of mesh normal vectors.
    """
    options = {}
    nargin = len(args)
    options['attenuation'     ] = args[0] if nargin > 0 else kwargs.get('attenuation', 1)
    options['refractive_index'] = args[1] if nargin > 1 else kwargs.get('refractive_index', 1)

    options['vertices_id'] = args[2] if nargin > 2 else kwargs.get('vertices_id', -1)
    options['transformations'] = None
    options['faces_id'     ] = -1
    options['normals_id'   ] = -1

    if nargin == 4:
        options['transformations'] = args[3] if nargin > 3 else kwargs.get('transformations', None)
        options['faces_id'     ] = -1
        options['normals_id'   ] = -1
    elif nargin > 4:
        options['faces_id'   ] = args[3] if nargin > 3 else kwargs.get('faces_id', -1)
        options['normals_id' ] = args[4] if nargin > 4 else kwargs.get('normals_id', -1)
        options['transformations'] = args[5] if nargin > 5 else kwargs.get('transformations', None)
    return Mesh(options['attenuation'     ],
                options['refractive_index'],
                options['vertices_id'     ],
                options['faces_id'        ],
                options['normals_id'      ],
                options['transformations'])