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

import numpy as np, os
import trimesh, pathlib
import matplotlib.pyplot as plt
from time import time

import mesh_projector as mp



py_file_dir = '/'.join(__file__.split('/')[ :-1])

if py_file_dir == '':
    py_file_dir = '.'
data_dir = py_file_dir + '/../data/'

print(f"Reading data from {data_dir}")
mesh_path = pathlib.Path(data_dir + "cuboid.stl")
assert os.path.exists(mesh_path), f"Mesh file {mesh_path} does not exist"
mesh = trimesh.load(mesh_path.absolute())

det_rows = 512
det_cols = 512

vertices = np.asarray(mesh.vertices, dtype=np.float32)*180 # times 180 to scale the mesh up
vertices += np.array([0, 0, 10])
faces = np.asarray(mesh.faces, dtype=np.int32)
normals = np.asarray(mesh.face_normals, dtype=np.float32)
vert_id = mp.create_cuda_array('vertices', nd_array=vertices)
face_id = mp.create_cuda_array('faces', nd_array=faces)
norm_id = mp.create_cuda_array('normals', nd_array=normals)
mesh = mp.create_mesh(1.0, 1.0, vert_id, face_id, norm_id)

views = np.array([
                    [-5000, 0, 0, 200, 000, 0, 0, 0, -1, 0, -1, 0],
                    [0, 0, -5000, 0, 0, 200, 1, 0,  0, 0,  1, 0],
                    [-1000, 0, 0, 500, 0, 0, 0, 0, -1, 0,  -1, 0],
                    [-1000, 0, 0, 500, 0, 0, 0, 0, -2.0, 0,  -0.5, 0]
                  ], dtype=np.float32)
views_id = mp.create_cuda_array('views', nd_array=views)
proj_geom = mp.create_projection_geometry('cone_vec', det_rows, det_cols, views_id)
sino_id = mp.create_cuda_array('sino', shape=(views.shape[0], det_rows, det_cols))
cfg = {'geometry': proj_geom, 'sino': sino_id,  'mesh': mesh}

print("Starting projector...")
proj_id = mp.create_projector('raster', cfg) # fills the sino
mp.run(proj_id)
print("Done")

#sino = mp.get_cuda_array(sino_id)
#sino = sino.reshape((views.shape[0], det_rows, det_cols))

import cupy as cp #, torch
cuda_obj = mp.get_cuda_obj(sino_id)
cp_obj = cp.asarray( mp.get_cuda_obj(sino_id) )
assert cuda_obj.__cuda_array_interface__['data'][0] == cp_obj.__cuda_array_interface__['data'][0], "The data pointer should be the same"

cp_obj[0:300*300] = 0
assert cuda_obj.__cuda_array_interface__['data'][0] == cp_obj.__cuda_array_interface__['data'][0], "The data pointer should be the same"

cp_obj2 = 2*cp_obj
assert cuda_obj.__cuda_array_interface__['data'][0] != cp_obj2.__cuda_array_interface__['data'][0], "The data pointer should now be different"


sino_id2 = mp.from_cuda_array(cp_obj2)
sino2 = mp.get_cuda_array(sino_id2)
sino2_obj = mp.get_cuda_obj(sino_id2)
assert sino2_obj.__cuda_array_interface__['data'][0] == cp_obj2.__cuda_array_interface__['data'][0], "The data pointer should be the same"
assert (sino2 == cp_obj2.get()).all(), "The data should be the same"


# if we now run the projector, sino_id2 gets rewritten with the same data of "sino_id", but cp_obj2 should be following the changes
cfg['sino'] = sino_id2
mp.run(proj_id)

assert (sino2 == cp_obj2.get()).all(), "The data should be the same"


sino = mp.get_cuda_array(sino_id)

# sino = sino.reshape((views.shape[0], det_rows, det_cols))
# cp_sino = cp_obj.reshape((views.shape[0], det_rows, det_cols))
                    
# plt.figure()
# plt.imshow(sino[0, :, :])
# plt.figure()
# plt.imshow(cp_sino[0, :, :].get())
# plt.show()

# %% Clear all
mp.delete_all_cuda_arrays()