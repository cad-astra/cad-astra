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

def rotate_2d(X, Y, angle):
    X_rotated = X*np.cos(angle) - Y*np.sin(angle)
    Y_rotated = X*np.sin(angle) + Y*np.cos(angle)
    return X_rotated, Y_rotated

py_file_dir = '/'.join(__file__.split('/')[ :-1])

if py_file_dir == '':
    py_file_dir = '.'
data_dir = py_file_dir + '/../data/'

print(f"Reading data from {data_dir}")
mesh = trimesh.load(data_dir + "cuboid.stl")

scale_factor = 20
offset = [30, 0, 0]

vertices = np.asarray(mesh.vertices, dtype=np.float32)*scale_factor + offset
faces = np.asarray(mesh.faces, dtype=np.int32)

triangles = np.empty((faces.shape[0], 9), dtype=np.float32)
for i, face in enumerate(faces):
    triangles[i] = np.concatenate([vertices[face[0]], vertices[face[1]], vertices[face[2]]])

vert_id = mp.create_cuda_array('triangles', nd_array=triangles)

ang = np.pi/4

x_src, y_src = rotate_2d(40, 0.0, ang)
x_det, y_det = rotate_2d(-300.0, 0.0, ang)
x_basis_det_1, y_basis_det_1  = rotate_2d(0, -1, ang)

views = np.array([
                    [50, 0, 0, -300, 0, 0, 0, 0, -1, 0, -1, 0], # source is at the front face
                    [40, 0, 0, -300, 0, 0, 0, 0, -1, 0, -1, 0], # source is inside the cube
                    [x_src, y_src, 0, x_det, y_det, 0, 0, 0, -1, x_basis_det_1, y_basis_det_1, 0],
                    [ 0, 0, 0, -300, 0, 0, 0, 0, -1, 0, -1, 0], # mesh is behind the source
                    [100,0, 0,   60, 0, 0, 0, 0, -1, 0, -1, 0] # mesh is behind the detector                
                  ], dtype=np.float32)

views_id = mp.create_cuda_array('views', nd_array=views)

det_rows = 512
det_cols = 512
proj_geom = mp.create_projection_geometry('cone_vec', det_rows, det_cols, views_id)

sino_id = mp.create_cuda_array('sino', shape=(views.shape[0], det_rows, det_cols))

# %% Plot the mesh
fig = plt.Figure()
ax = plt.axes(projection="3d")
faces_inds = np.arange(0, faces.shape[0])
triangles = vertices[faces[faces_inds]]
for i in range(triangles.shape[0]):
    mp.utils.plot_triang_matplotlib(triangles[i], ax, color="red", fill=False)

for iv in range(views.shape[0]):
    # ax.plot3D(views[iv, 0], views[iv, 1], views[iv, 2], "blue", marker='x')
    ax.plot3D([views[iv, 0], views[iv, 3]], [views[iv, 1], views[iv, 4]], [views[iv, 2], views[iv, 5]], "blue", marker='x', linestyle='dashed')

ax.set_xlim([views.min(), views.max()])
ax.set_ylim([views.min(), views.max()])
ax.set_zlim([views.min(), views.max()])
ax.set_title("Projection geometry")
# %% run the projector with timer
t0 = time()
mp.project(vert_id, sino_id, proj_geom)

print('Done in {} sec.'.format(time() - t0))

# print("sino.max = {}".format(sino.max()))

# %% plot
sino = mp.get_cuda_array(sino_id)
sino = sino.reshape((views.shape[0], det_rows, det_cols))

titles = ['Source is at the face', 'Source is inside the mesh', 'Vertex behind the source',
          'Mesh is behind the source', 'Mesh is behind the detector']

fig = plt.figure(figsize=(16, 2))
for i in range(views.shape[0]):
    ax = fig.add_subplot(1, views.shape[0], i+1)
    cs = ax.imshow(sino[i], cmap="gray")
    ax.set_title(titles[i])
    # fig.colorbar(cs, ax=ax)
plt.tight_layout()
plt.show()
# %% Clear all
mp.delete_all_cuda_arrays()
