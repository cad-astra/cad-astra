#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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



# TODO(pavel): implement project choice as a command line argument

# %% Import 
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from time import time

import mesh_projector as mp

# %% Parameters
run_raster = True
run_optix  = True

# %% Define functions
def rotate_2d(X, Y, angle):
    X_rotated = X*np.cos(angle) - Y*np.sin(angle)
    Y_rotated = X*np.sin(angle) + Y*np.cos(angle)
    return X_rotated, Y_rotated

def run_test(mesh, proj_geom, nViews, sino_id, projector_type, plot=False, plot_slice_idx=[0]):

    cfg = {}
    cfg['mesh']     = mesh
    cfg['geometry'] = proj_geom
    cfg['sino']     = sino_id
    cfg["tracer_policy"]  = "non-recursive"   # Uncomment for the non-recursive tracing

    proj_id = mp.create_projector(projector_type, cfg)

    print(f"Starting test for {projector_type} projector...")
    t_run_1 = time()
    mp.initialize_sino(sino_id, float(0.0))
    mp.run(proj_id)
    t_run_2 = time()
    run_time = t_run_2 - t_run_1
    print(f"Done in {run_time} seconds\n")

    mp.delete_projector(proj_id)
    
    sino = mp.get_cuda_array(sino_id)
    sino = sino.reshape((nViews, det_rows, det_cols))
    if plot:
        for i in plot_slice_idx:
            plt.figure()
            plt.imshow(sino[i], cmap="gray")
            plt.title(f'{projector_type} projection')
            plt.colorbar()
    
    return sino

def compare(sino, sino_reference, plot=False, plot_slice_idx=[0], title='', plot_title=''):
    
    sino_diff = np.abs(sino-sino_reference)
    abs_err_sum  = np.sum(sino_diff)
    abs_err_mean = np.mean(sino_diff)
    abs_err_max  = np.max(sino_diff)
    print(title)
    print(10*'-')
    print(f'Absolute difference: {abs_err_sum}')
    print(f'Mean     difference: {abs_err_mean}')
    print(f'Max      difference: {abs_err_max}')
    print(10*'-')

    if plot:
        for i in plot_slice_idx:
            plt.figure()
            plt.imshow(sino_diff[i], cmap="gray")
            plt.title(f'Projection difference. {plot_title}')
            plt.colorbar()

# %% Read STL, create cuda arrays
# py_file_dir = '/home/pparamonov/Projects/mesh-fp-prototype/tests'
py_file_dir = '/'.join(__file__.split('/')[ :-1])

if py_file_dir == '':
    py_file_dir = '.'
data_dir = py_file_dir + '/../examples/data/'

print(f"Reading data from {data_dir}")

mesh_stl_filename = "cuboid.stl"
mesh_stl = trimesh.load(data_dir + mesh_stl_filename)

scale_k = 120

vertices = np.asarray(mesh_stl.vertices, dtype=np.float32)*scale_k*2
faces = np.asarray(mesh_stl.faces, dtype=np.int32)
normals = np.asarray(mesh_stl.face_normals, dtype=np.float32)

angles = np.linspace(0, 2*np.pi, 1, dtype=np.float32)

d_source = 10000
d_detector = 200
det_rows = 512
det_cols = 512
dy = 0.125
dx = 0.125

# %% Generate reference sinogram
# sino_ref = np.zeros(shape=(angles.shape[0], det_rows, det_cols))
pix_inds_rows = np.arange(-det_rows*0.5 + 0.5, det_rows*0.5, 1)
pix_inds_cols = np.arange(-det_cols*0.5 + 0.5, det_cols*0.5, 1)

k_front = (d_source - scale_k) / (d_source + d_detector)
k_back  = (d_source + scale_k) / (d_source + d_detector)

pix_coord_x = np.asarray(pix_inds_rows * dx, dtype=np.float32)
pix_coord_y = np.asarray(pix_inds_cols * dy, dtype=np.float32)

# pix_coord_x, pix_coord_y = rotate_2d(pix_coord_x, pix_coord_y, angles[0])

Dx, Dy, Dz = np.meshgrid(pix_coord_x, pix_coord_y, np.asarray([d_detector], dtype=np.float32), indexing='ij')

Ofront_x, Ofront_y, Ofront_z = Dx*k_front, Dy*k_front, -np.ones(shape=Dz.shape, dtype=np.float32) * np.float32(scale_k)
Oback_x,  Oback_y,  Oback_z  = Dx*k_back,  Dy*k_back,   np.ones(shape=Dz.shape, dtype=np.float32) * np.float32(scale_k)

ray_path_x, ray_path_y, ray_path_z = Oback_x - Ofront_x,  Oback_y - Ofront_y,  Oback_z - Ofront_z

ray_length = np.sqrt(ray_path_x**2 + ray_path_y**2 + ray_path_z**2)
ray_d_x, ray_d_y, ray_d_z  = ray_path_x/ray_length, ray_path_y/ray_length, ray_path_z/ray_length

sino_ref = np.squeeze(ray_length)
plt.figure()
plt.imshow(sino_ref, cmap="gray")
plt.title('Reference projection')
plt.colorbar()

# try:
#     from mayavi import mlab
#     mlab.quiver3d(Ofront_x, Ofront_y, Ofront_z, ray_d_x, ray_d_y, ray_d_z)
# except:
#     print('Could not plot ray directions with mayavi...')
# %% run the projector benchmark
vert_id = mp.create_cuda_array('vertices', nd_array=vertices)
face_id = mp.create_cuda_array('faces', nd_array=faces)
norm_id = mp.create_cuda_array('normals', nd_array=normals)
angles_id = mp.create_cuda_array('angles', nd_array=angles)
sino_id = mp.create_cuda_array('sino', shape=(angles.shape[0], det_rows, det_cols))

proj_geom = mp.create_projection_geometry('cone', dx, dy, det_rows, det_cols, angles_id, d_source, d_detector)

# Backward compatibility hack due to api change
# TODO: remove when ready
try:
    mesh = mp.create_mesh(1.0, 1.0, vert_id, face_id, norm_id)
except:
    mesh = mp.create_mesh('indexed_mesh', 1.0, 1.0, vert_id, face_id, norm_id)

sino_raster = run_test(mesh, proj_geom, angles.shape[0], sino_id, 'raster', plot=True)
sino_optix  = run_test(mesh, proj_geom, angles.shape[0], sino_id, 'optix',  plot=True)

compare(sino_raster, sino_ref, True, title='Test for raster projector',plot_title='Raster projector')
print('')
compare(sino_optix,  sino_ref, True, title='Test for OptiX projector', plot_title='OptiX projector')

print('')
compare(sino_optix,  sino_raster, True, title='Raster - OptiX comparison', plot_title='Raster - OptiX difference')

plt.show()
# %% Clear all cuda arrays
mp.delete_all_cuda_arrays()