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
mesh = trimesh.load(data_dir + "cuboid.stl")

vertices = np.asarray(mesh.vertices, dtype=np.float32)*180
faces = np.asarray(mesh.faces, dtype=np.int32)
normals = np.asarray(mesh.face_normals, dtype=np.float32)

vert_id = mp.create_cuda_array('vertices', nd_array=vertices)
face_id = mp.create_cuda_array('faces', nd_array=faces)
norm_id = mp.create_cuda_array('normals', nd_array=normals)

views = np.array([
                    [-5000, 0, 0, 200, 0, 0, 0, 0, -1, 0, -1, 0],
                    [0, 0, -5000, 0, 0, 200, 1, 0,  0, 0,  1, 0],
                    [-1000, 0, 0, 500, 0, 0, 0, 0, -1, 0,  -1, 0],
                    [-1000, 0, 0, 500, 0, 0, 0, 0, -2.0, 0,  -0.5, 0]
                  ], dtype=np.float32)

views_id = mp.create_cuda_array('views', nd_array=views)

det_rows = 512
det_cols = 512
proj_geom = mp.create_projection_geometry('cone_vec', det_rows, det_cols, views_id)

sino_id = mp.create_cuda_array('sino', shape=(views.shape[0], det_rows, det_cols))

# %% run the projector with timer
print("Starting projector...")
t0 = time()
mp.project([vert_id, face_id, norm_id], sino_id, proj_geom)

print('Done in {} sec.'.format(time() - t0))

# print("sino.max = {}".format(sino.max()))

# %% plot
sino = mp.get_cuda_array(sino_id)
sino = sino.reshape((views.shape[0], det_rows, det_cols))
for i in range(views.shape[0]):
    plt.figure()
    plt.imshow(sino[i], cmap="gray")
    plt.colorbar()
# plt.figure()
# plt.imshow(sino[:, sino.shape[1]//2, :], cmap="gray")
# plt.colorbar()
plt.show()
# %% Clear all
mp.delete_all_cuda_arrays()