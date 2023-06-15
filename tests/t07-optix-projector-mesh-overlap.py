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
Test overlaping meshes.
Expected behavoir - attenuation for rays travelling through cubes equals to 1*1 + 2*0.5 = 2.0
"""
import numpy as np
from numpy.lib.twodim_base import tri
import trimesh
import matplotlib.pyplot as plt
from time import time

import mesh_projector as mp
from trimesh import triangles

def vertices_faces_2triangles(vertices: np.ndarray, faces: np.ndarray):
    # TODO(pavel): this should be in utilities
    triangles = np.empty((faces.shape[0], 9), dtype=np.float32)
    for i, face in enumerate(faces):
        triangles[i] = np.concatenate([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
    return triangles

# %% Read STL
py_file_dir = '/'.join(__file__.split('/')[ :-1])

if py_file_dir == '':
    py_file_dir = '.'
data_dir = py_file_dir + '/../data/'

print(f"Reading data from {data_dir}")
sample_mesh = trimesh.load(data_dir + "cuboid.stl")

sample_mask_mesh = trimesh.load(data_dir + "cuboid.stl")

scale_mesh = 1

vertices = scale_mesh * np.asarray(sample_mesh.vertices, dtype=np.float32)
faces = np.asarray(sample_mesh.faces, dtype=np.int32)

sample_triangles = vertices_faces_2triangles(vertices, faces)
print(f'Sample mesh triangles shape: {sample_triangles.shape}')

scale_mask = 1.0

sample_mask_vertices  = scale_mask*np.asarray(sample_mask_mesh.vertices, dtype=np.float32) + [-0.5, 0, 0]
sample_mask_triangles = vertices_faces_2triangles(sample_mask_vertices,
                                                  np.asarray(sample_mask_mesh.faces,    dtype=np.int32  ))

print(f'Sample MASK triangles shape: {sample_mask_triangles.shape}')

sample_mesh_vert_id = mp.create_cuda_array('triangles', nd_array=sample_triangles)
sample_mask_mesh_vert_id = mp.create_cuda_array('triangles', nd_array=sample_mask_triangles)

attenuation = 1
mesh_refractive_index = 1.0
cu_sample_mesh = mp.create_mesh( attenuation, mesh_refractive_index, sample_mesh_vert_id )
cu_sample_mask_mesh = mp.create_mesh( 2.0, 1.0, sample_mask_mesh_vert_id )

rot = lambda x, theta: [x[0]*np.cos(theta)-x[1]*np.sin(theta),x[0]*np.sin(theta)+x[1]*np.cos(theta),x[2]]
# angles = np.linspace(0,182,182,endpoint=False)
angles = np.array([0], dtype=np.float32)
start_view = [-10, 0, 0, 10, 0, 0, 0, 0, -0.075, 0, -0.075, 0]
views = []
for ii in angles:
    views.append(np.concatenate((rot(start_view[0:3],np.deg2rad(ii)),rot(start_view[3:6],np.deg2rad(ii)),
                 rot(start_view[6:9],np.deg2rad(ii)),rot(start_view[9:12],np.deg2rad(ii)))))
views = np.array(views,dtype=np.float32)

det_rows = 128
det_cols = 128

rays_rows =  det_rows*1 # (1024**2)
rays_cols =  det_cols*1

views_id = mp.create_cuda_array('views', nd_array=views)

proj_geom = mp.create_projection_geometry('cone_vec', det_rows, det_cols, views_id)

sino_id = mp.create_cuda_array('sino', shape=(views.shape[0], det_rows, det_cols))
mp.initialize_sino(sino_id, 0)

mask_bar_width = 0.030
sd_d = np.linalg.norm(views[0, :3]- views[0, 3:6])

cfg = {}
cfg['mesh']          = [cu_sample_mesh, cu_sample_mask_mesh]    # A list of meshes that represent a sample
cfg['geometry']      = proj_geom
cfg['sino']          = sino_id
cfg['rays_row_count'] = rays_rows
cfg['rays_col_count'] = rays_cols

# cfg['sample_mask_mesh'  ] = sample_mask_mesh
# cfg['detector_mask_mesh'] = detector_mask_mesh

# cfg['detector_pixel_policy'] = 'sum_intensity'

proj_id = mp.create_projector('optix', cfg)

try:
    mp.utils.plot_mesh_mayavi(vertices, faces)
    mp.utils.plot_mesh_mayavi(sample_mask_vertices, np.asarray(sample_mask_mesh.faces))
    mp.utils.plot_projector_mayavi(views, det_rows, det_cols, plot_det_grid=False)
except ImportError:
    print("! Could not import mlab from mayavi, mesh plotting is skipped.")
except Exception as err:
    print(f"! An exception cought when plotting the mesh with mayavi: {err}")

# %% Run the projector to compute the ray paths
#
t0 = time()
print('Starting OptiX projector for computing the projections...')

mp.run(proj_id)

print('Done in {} sec.'.format(time() - t0))

# %% plot
sino = mp.get_cuda_array(sino_id)
sino = sino.reshape((views.shape[0], det_rows, det_cols))
for i in [0]:
    plt.figure()
    plt.imshow(sino[i], cmap="gray")
    plt.colorbar()
plt.show()
# %% Clear all
mp.delete_all_cuda_arrays()
mp.delete_projector(proj_id)