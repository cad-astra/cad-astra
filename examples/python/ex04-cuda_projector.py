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

"""
This example demonstrates more general approach to the projector configuration.
TODO: include refraction and other configuration fields.
"""
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from time import time

import mesh_projector as mp
# %% Read STL

py_file_dir = '/'.join(__file__.split('/')[ :-1])

if py_file_dir == '':
    py_file_dir = '.'
data_dir = py_file_dir + '/../data/'

print(f"Reading data from {data_dir}")
# mesh = trimesh.load(data_dir + "stanford_bunny_remeshed.stl")
mesh = trimesh.load(data_dir + "cuboid.stl")

attenuation = 1.0

vertices = np.asarray(mesh.vertices, dtype=np.float32)*180 # times 180 to scale the mesh up
faces = np.asarray(mesh.faces, dtype=np.int32)
normals = np.asarray(mesh.face_normals, dtype=np.float32)

# Non-indexed mesh
# triangles = np.empty((faces.shape[0], 9), dtype=np.float32)
# for i, face in enumerate(faces):
#     triangles[i] = np.concatenate([vertices[face[0]], vertices[face[1]], vertices[face[2]]])

# triang_id = mp.create_cuda_array('triangles', nd_array=triangles)
# mesh = mp.create_mesh('non_indexed_mesh', attenuation, 1.0, triang_id)

# Indexed mesh
vert_id = mp.create_cuda_array('vertices', nd_array=vertices)
face_id = mp.create_cuda_array('faces', nd_array=faces)
norm_id = mp.create_cuda_array('normals', nd_array=normals)

# Backward compatibility hack due to api change
# TODO: remove when ready
# mesh = mp.create_mesh('indexed_mesh', attenuation, 1.0, vert_id, face_id, norm_id)
mesh = mp.create_mesh(attenuation, 1.0, vert_id, face_id, norm_id)

# Rotation geometry
angles = np.linspace(0, np.pi, 4, dtype=np.float32)
ang_id = mp.create_cuda_array('angles', nd_array=angles)

det_rows, det_cols = 512, 512
proj_geom = mp.create_projection_geometry('cone', 1, 1, det_rows, det_cols, ang_id, 5000, 500)
# det_size_x, det_size_y = 0.125, 0.125
# proj_geom = mp.create_projection_geometry('cone', det_size_x, det_size_y, det_rows, det_cols, ang_id, 20, 20)
sino_id = mp.create_cuda_array('sino', shape=(angles.shape[0], det_rows, det_cols))

# Vector geometry 
# views = np.array([
#                     [-20, 0, 0, 20, 0, 0, 0, 0, -0.125, 0, -0.125, 0],
#                     # [0, 0, -5000, 0, 0, 200, 1, 0,  0, 0,  1, 0],
#                     # [-1000, 0, 0, 500, 0, 0, 0, 0, -1, 0,  -1, 0],
#                     # [-1000, 0, 0, 500, 0, 0, 0, 0, -2.0, 0,  -0.5, 0]
#                   ], dtype=np.float32)

# views_id = mp.create_cuda_array('views', nd_array=views)
# det_rows = 512
# det_cols = 512
# proj_geom = mp.create_projection_geometry('cone_vec', det_rows, det_cols, views_id)
# sino_id = mp.create_cuda_array('sino', shape=(views.shape[0], det_rows, det_cols))
# mp.initialize_sino(sino_id, 0)

# %% Create projector

cfg = {}
cfg['mesh']     = mesh
cfg['geometry'] = proj_geom
cfg['sino']     = sino_id
# cfg['rays_row_count'] = 1024
# cfg['rays_col_count'] = 1024
# cfg['detector_pixel_policy'] = 'sum_intensity'

# CUDA rasterizer
# proj_id = mp.create_projector('raster', cfg)
# OptiX ray tracer
proj_id = mp.create_projector('optix', cfg)

# %% Run projector
t0 = time()
mp.run(proj_id)
print('Done in {} sec.'.format(time() - t0))

# %% plot
sino = mp.get_cuda_array(sino_id)
sino = sino.reshape((angles.shape[0], det_rows, det_cols))
for i in range(angles.shape[0]):
    plt.figure()
    plt.imshow(sino[i], cmap="gray")
    plt.colorbar()
plt.show()
# %% Clean up
mp.delete_all_cuda_arrays()
mp.delete_projector(proj_id)