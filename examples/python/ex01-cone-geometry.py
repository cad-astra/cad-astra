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
# %% Read STL and create cuda arrays

py_file_dir = '/'.join(__file__.split('/')[ :-1])

if py_file_dir == '':
    py_file_dir = '.'
data_dir = py_file_dir + '/../data/'

print(f"Reading data from {data_dir}")
mesh = trimesh.load(data_dir + "cuboid.stl")

vertices = np.asarray(mesh.vertices, dtype=np.float32)*180 # times 180 to scale the mesh up
faces = np.asarray(mesh.faces, dtype=np.int32)
normals = np.asarray(mesh.face_normals, dtype=np.float32)
# angles = np.linspace(np.pi/3, 2*np.pi, 1, dtype=np.float32)
# angles = np.linspace(0, 2*np.pi, 1, dtype=np.float32)
angles = np.linspace(0, 2*np.pi, 512, dtype=np.float32)

vert_id = mp.create_cuda_array('vertices', nd_array=vertices)
face_id = mp.create_cuda_array('faces', nd_array=faces)
norm_id = mp.create_cuda_array('normals', nd_array=normals)

ang_id = mp.create_cuda_array('angles', nd_array=angles)

det_size_x, det_size_y = 1, 1
det_rows, det_cols = 512, 512
proj_geom = mp.create_projection_geometry('cone', det_size_x, det_size_y, det_rows, det_cols, ang_id, 5000, 500)

sino_id = mp.create_cuda_array('sino', shape=(angles.shape[0], det_rows, det_cols))

# %% run the projector with timer
print("Running the projector...")
t0 = time()
mp.project([vert_id, face_id, norm_id], sino_id, proj_geom, proj_type="raster")

print('Done in {} sec.'.format(time() - t0))

# %% plot
sino = mp.get_cuda_array(sino_id)
sino = sino.reshape((angles.shape[0], det_rows, det_cols))

first_ang  = angles.shape[0]//3
mid_angle  = angles.shape[0]//2
last_angle = 2*angles.shape[0]//3
ang_list = [first_ang]
if mid_angle not in ang_list:
    ang_list.append(mid_angle)
if last_angle not in ang_list:
    ang_list.append(last_angle)

for i in ang_list:
    plt.figure()
    plt.imshow(sino[i], cmap="gray")
    plt.colorbar()
plt.show()
