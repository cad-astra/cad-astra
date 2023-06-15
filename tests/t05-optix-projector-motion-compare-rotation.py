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
Compare sinograms produced from rotation and mesh motion geometries.
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


def plot_radiographs(sinogram, slices=None, plt_show=False, title=''):
    if slices is None:
        slices = (0, int(sinogram.shape[0]/2), sinogram.shape[0]-1)
    for i in slices:
        plt.figure()
        plt.imshow(sinogram[i], cmap="gray")
        plt.colorbar()
        plt.title(title + f'[{i}]')

    if plt_show:
        plt.show()

# %% Read STL and create mesh objects
py_file_dir = '/'.join(__file__.split('/')[ :-1])

if py_file_dir == '':
    py_file_dir = '.'
data_dir = py_file_dir + '/../data/'

print(f"Reading data from {data_dir}")
# sample_mesh = trimesh.load(data_dir + "cube.stl")
sample_mesh = trimesh.load(data_dir + "stanford_bunny_remeshed.stl")
# sample_mesh = trimesh.load(data_dir + "shifted_cube_1.stl")

sample_mask_mesh = trimesh.load(data_dir + "cuboid.stl")

scale_mesh = 1

vertices = scale_mesh * np.asarray(sample_mesh.vertices, dtype=np.float32)
faces = np.asarray(sample_mesh.faces, dtype=np.int32)

sample_triangles = vertices_faces_2triangles(vertices, faces)
print(f'Sample mesh triangles shape: {sample_triangles.shape}')

scale_mask = 1

sample_mask_vertices  = scale_mask*np.asarray(sample_mask_mesh.vertices, dtype=np.float32) + [0, 2, 0]
sample_mask_triangles = vertices_faces_2triangles(sample_mask_vertices,
                                                  np.asarray(sample_mask_mesh.faces,    dtype=np.int32  ))

print(f'Sample MASK triangles shape: {sample_mask_triangles.shape}')

sample_mesh_vert_id = mp.create_cuda_array('triangles', nd_array=sample_triangles)
sample_mask_mesh_vert_id = mp.create_cuda_array('triangles', nd_array=sample_mask_triangles)

attenuation = 1
mesh_refractive_index = 1.0

# %% Projector setup and mesh transformations
# We will add rotation of a sample mesh and translation of the mask mesh 

# Sample rotation angles
n_angles = 180
angles   = np.linspace(0, np.pi/2, n_angles, endpoint=True, dtype=np.float32)

# Time syncronizes all the transformations during the projector work
begin_time = 0
end_time   = angles.shape[0]-1

# Mesh transformation matrices
trans1 = mp.create_transformation('rotation',    begin_time, end_time, angles, 'z')

cu_sample_mesh = mp.create_mesh( attenuation, mesh_refractive_index, sample_mesh_vert_id, trans1 )

ang_id = mp.create_cuda_array('angles', nd_array=-angles)

# Projector setup
det_size_x, det_size_y = 0.025, 0.025
det_rows, det_cols = 256, 256

views_time_keys = np.linspace(begin_time, end_time, angles.shape[0], dtype=np.float32)
print(f'Time keys for the projection views: {views_time_keys}')
cu_views_time_keys_id = mp.create_cuda_array('arbitrary', shape=views_time_keys.shape, dtype=np.float32)
mp.store(cu_views_time_keys_id, views_time_keys)

source_origin = 5
detect_origin = 5

view = np.array([[0, -source_origin, 0, 0, detect_origin, 0, det_size_x, 0, 0, 0, 0, det_size_y]], dtype=np.float32)
cu_view_id = mp.create_cuda_array('views', nd_array=view)

proj_geom_transform = mp.create_projection_geometry('cone_mesh_transform',
                                          det_size_x, det_size_y,
                                          det_rows, det_cols,
                                          view_keys=cu_views_time_keys_id,  # TODO: should be just an array of views time keys
                                          projector_view=cu_view_id)

proj_geom_rotation  = mp.create_projection_geometry('cone', det_size_x, det_size_y, det_rows, det_cols, ang_id, source_origin, detect_origin)

sino_id = mp.create_cuda_array('sino', shape=(views_time_keys.shape[0], det_rows, det_cols))
mp.initialize_sino(sino_id, 0)

cfg = {}
cfg['mesh']          = [cu_sample_mesh]    # A list of meshes that represent a sample
cfg['geometry']      = proj_geom_transform
cfg['sino']          = sino_id
# cfg['detector_pixel_policy'] = 'sum_intensity'
proj_transform_id = mp.create_projector('optix', cfg)

cfg['geometry']      = proj_geom_rotation
proj_rotation_id  = mp.create_projector('optix', cfg)

try:
    mp.utils.plot_mesh_mayavi(vertices, faces)
    # mp.utils.plot_mesh_mayavi(sample_mask_vertices, np.asarray(sample_mask_mesh.faces))
    mp.utils.plot_projector_mayavi(view, det_rows, det_cols, plot_det_grid=False)
except ImportError:
    print("! Could not import mlab from mayavi, mesh plotting is skipped.")
except Exception as err:
    print(f"! An exception cought when plotting the mesh with mayavi: {err}")

# %% Run the projectors
print('Running cone_mesh_transform geometry...')
t0 = time()
mp.run(proj_transform_id)
print('Done in {} sec.'.format(time() - t0))

sino_transform = mp.get_cuda_array(sino_id)
sino_transform = sino_transform.reshape((angles.shape[0], det_rows, det_cols))

mp.initialize_sino(sino_id, 0)
print('Running cone geometry...')
t0 = time()
mp.run(proj_rotation_id)
print('Done in {} sec.'.format(time() - t0))

sino_rotation = mp.get_cuda_array(sino_id)
sino_rotation = sino_rotation.reshape((angles.shape[0], det_rows, det_cols))

sino_diff = np.abs(sino_transform - sino_rotation)
# %% plot
# slices = [0, int(angles.shape[0]/3), int(angles.shape[0]/2), int(angles.shape[0]*2/3), angles.shape[0]-1]
slices = [int(np.argmax(sino_diff) / (sino_diff.shape[2] * sino_diff.shape[1]))]

plot_radiographs(sino_rotation, slices, title='Rotaion geom. ')
plot_radiographs(sino_transform, slices, title='Mesh transform, geom. ')
plot_radiographs(sino_diff, slices, title='Abs. difference ')

print(f'Max.  sino difference: {np.max(sino_diff)}')
print(f'Total sino difference: {np.sum(sino_diff)}')

plt.show()
# %% Clear all
mp.delete_all_cuda_arrays()
mp.delete_projector(proj_rotation_id)
mp.delete_projector(proj_transform_id)