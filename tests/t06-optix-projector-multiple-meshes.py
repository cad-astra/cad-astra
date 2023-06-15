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
Here we test projection of multiple meshes in arranged into different configurations:
    - two meshes visible without covering each other
    - nested meshes
    - one mesh behind the other
"""
import numpy as np
from numpy.lib.twodim_base import tri
import trimesh
import matplotlib.pyplot as plt
from time import time

import mesh_projector as mp
from trimesh import triangles

mp.set_logging_level_optix("error")
mp.set_logging_level_py("error")
mp.set_logging_level_host("debug")
mp.set_logging_fmt_host("all")

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
data_dir = py_file_dir + '/../examples/data/'

print(f"Reading data from {data_dir}")
mesh_1 = trimesh.load(data_dir + "cuboid.stl")

mesh_2 = trimesh.load(data_dir + "cuboid.stl")

# This mesh remains stationary at the origin
scale_mesh = 1
mesh_1_vertices = scale_mesh * np.asarray(mesh_1.vertices, dtype=np.float32)
mesh_1_faces = np.asarray(mesh_1.faces, dtype=np.int32)
mesh_1_triangles = vertices_faces_2triangles(mesh_1_vertices, mesh_1_faces)
print(f'Mesh 1 triangles shape: {mesh_1_triangles.shape}')

# Translated mesh that i placed away from the optical axis
scale_mesh = 1
mesh_transl_vertices  = scale_mesh*np.asarray(mesh_2.vertices, dtype=np.float32) + [5, 0, 2]
mesh_transl_triangles = vertices_faces_2triangles(mesh_transl_vertices,
                                                  np.asarray(mesh_2.faces,    dtype=np.int32  ))

# Scaled down and nested into mesh_1 mesh:
scale_mesh = 0.5
mesh_nested_vertices  = scale_mesh*np.asarray(mesh_2.vertices, dtype=np.float32)
mesh_nested_triangles = vertices_faces_2triangles(mesh_nested_vertices,
                                                  np.asarray(mesh_2.faces,    dtype=np.int32  ))

# This mesh is placed at the optical axis, as well as mesh_1:
scale_mesh = 2.0
mesh_behind_vertices  = scale_mesh*np.asarray(mesh_2.vertices, dtype=np.float32) + [4, 0, 0]
mesh_behind_triangles = vertices_faces_2triangles(mesh_behind_vertices,
                                                  np.asarray(mesh_2.faces,    dtype=np.int32  ))

# Prepare cuda arrays and CAD-ASTRA objects
mesh_1_vert_id = mp.create_cuda_array('triangles', nd_array=mesh_1_triangles)
mesh_transl_vert_id = mp.create_cuda_array('triangles', nd_array=mesh_transl_triangles)
mesh_nested_vert_id = mp.create_cuda_array('triangles', nd_array=mesh_nested_triangles)
mesh_behind_vert_id = mp.create_cuda_array('triangles', nd_array=mesh_behind_triangles)

attenuation = 1
mesh_refractive_index = 1.0
cu_mesh_1      = mp.create_mesh( attenuation, mesh_refractive_index, mesh_1_vert_id )
cu_mesh_transl = mp.create_mesh( 0.5, 1.0, mesh_transl_vert_id )
cu_mesh_nested = mp.create_mesh( 0.5, 1.0, mesh_nested_vert_id )
cu_mesh_behind = mp.create_mesh( 0.5, 1.0, mesh_behind_vert_id )

rot = lambda x, theta: [x[0]*np.cos(theta)-x[1]*np.sin(theta),x[0]*np.sin(theta)+x[1]*np.cos(theta),x[2]]
# angles = np.linspace(0,182,182,endpoint=False)
angles = np.array([0], dtype=np.float32)
start_view = [-10, 0, 0, 10, 0, 0, 0, 0.075, 0, 0, 0, 0.075]
views = []
for ii in angles:
    views.append(np.concatenate((rot(start_view[0:3],np.deg2rad(ii)),rot(start_view[3:6],np.deg2rad(ii)),
                 rot(start_view[6:9],np.deg2rad(ii)),rot(start_view[9:12],np.deg2rad(ii)))))
views = np.array(views,dtype=np.float32)

det_rows = 256
det_cols = 256

rays_rows =  det_rows*1 # (1024**2)
rays_cols =  det_cols*1

views_id = mp.create_cuda_array('views', nd_array=views)

proj_geom = mp.create_projection_geometry('cone_vec', det_rows, det_cols, views_id)

sino_id = mp.create_cuda_array('sino', shape=(views.shape[0], det_rows, det_cols))
mp.initialize_sino(sino_id, 0)

mask_bar_width = 0.030
sd_d = np.linalg.norm(views[0, :3]- views[0, 3:6])

cfg = {}
cfg['mesh']           = [ cu_mesh_1, cu_mesh_transl, cu_mesh_nested, cu_mesh_behind ]
cfg['geometry']       = proj_geom
cfg['sino']           = sino_id
cfg['rays_row_count'] = rays_rows
cfg['rays_col_count'] = rays_cols
# cfg["tracer_policy"]  = "non-recursive"   # Uncomment for the non-recursive tracing

# cfg['mesh_2'  ] = mesh_2
# cfg['detector_mask_mesh'] = detector_mask_mesh

# cfg['detector_pixel_policy'] = 'sum_intensity'

proj_id = mp.create_projector('optix', cfg)

try:
    mp.utils.plot_mesh_mayavi(mesh_1_vertices, mesh_1_faces)
    mp.utils.plot_mesh_mayavi(mesh_transl_vertices, np.asarray(mesh_2.faces))
    mp.utils.plot_mesh_mayavi(mesh_nested_vertices, np.asarray(mesh_2.faces))
    mp.utils.plot_mesh_mayavi(mesh_behind_vertices, np.asarray(mesh_2.faces))
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