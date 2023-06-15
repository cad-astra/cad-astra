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

from .cuda_gpu_utils import *

def set_gpu_index(idx: int) -> None:
    """
    Choose the current GPU index.
    """
    set_gpu_index_c(idx)

def vertices_faces_2triangles(vertices: np.ndarray, faces: np.ndarray):
    """
    Returns an array of mesh triangles based on arrays of vertices and faces.
    Each row in the array of triangles is a vector of shape (1, 9).

    Parameters:
    -----------
    vertices: numpy.ndarray
        Array of shape (N, 3) containing N mesh vertices.
    faces: numpy.ndarray
        Array of shape (M, 3) containing indices of vertices that compose M mesh faces.
    
    Returns:
    --------
    triangles : numpy.ndarray
        Array of shape (M, 9) containing coordinates of mesh faces.
    """
    triangles = np.empty((faces.shape[0], 9), dtype=np.float32)
    for i, face in enumerate(faces):
        triangles[i] = np.concatenate([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
    return triangles

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_triang_matplotlib(triangle, axes3d, color='green', fill=False, marker=None):
    """
    Plot a single triangle on 3D axes.

    Parameters:
    -----------
    triangle : numpy.ndarray
        Triangle vertices in array [x, y, z].
    axes3d : pyplot.Axes
        Axes object created with projection="3d" parameter, i.e., axex3d = plt.axes(projection="3d").
    color : string
        Edge color. The default is 'green'.
    fill : bool
        Fill the triangle with color. The default is False.
    marker : string
        Marker for vertices. The default is None.   
    """
    triangle_extended = np.vstack((triangle, triangle[0, :]))
    if fill:
        verts = [triangle]
        axes3d.add_collection3d(Poly3DCollection(verts, facecolor=color,
                                                 edgecolor=color))
    axes3d.plot3D(triangle_extended[:, 0], triangle_extended[:, 1],
                  triangle_extended[:, 2], color, marker=marker)

try:
    from mayavi import mlab
    # print("Imported mayavi...")
except ImportError:
    print("! Could not import mlab from mayavi, 3D plotting with mayavi is unavailable.")

def plot_mesh_mayavi(vertices, faces=None, *args, **kwargs):
    if faces is None:
        # if vertices.shape[1] == 3:
        faces = np.reshape(np.arange(vertices.shape[0]), (int(vertices.shape[0] / 3), 3))
        # elif vertices.shape[1] == 9:
    nargin = len(args)
    color  = args[0] if nargin > 0 else kwargs.get('color', None)
    line_width = args[1] if nargin > 0 else kwargs.get('line_width', 2.0)
    representation = args[2] if nargin > 0 else kwargs.get('representation', 'mesh')
    tube_radius = args[2] if nargin > 0 else kwargs.get('tube_radius', None)

    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces,
                         color=color, representation=representation,
                         line_width=line_width, tube_radius=tube_radius)

def plot_projector_mayavi(views, det_rows, det_cols, plot_det_grid=False):
    for j in range(views.shape[0]):
        source     = views[j, :3]
        det_center = views[j, 3:6]
        det_u      = views[j, 6:9]
        det_v      = views[j, 9:12]
        det_topleft   = det_center - 0.5*det_cols*det_u - 0.5*det_rows*det_v
        det_downleft  = det_center - 0.5*det_cols*det_u + 0.5*det_rows*det_v
        det_downright = det_center + 0.5*det_cols*det_u + 0.5*det_rows*det_v
        det_topright  = det_center + 0.5*det_cols*det_u - 0.5*det_rows*det_v    

        projection_cone_1 =  np.vstack(( source, det_topleft ))
        projection_cone_2 =  np.vstack(( source, det_downleft ))
        projection_cone_3 =  np.vstack(( source, det_downright ))
        projection_cone_4 =  np.vstack(( source, det_topright ))

        # Source and detector position
        mlab.points3d(source[0], source[1], source[2], [0.5], color=(1.0, 0.0, 0.0), scale_mode='scalar', scale_factor=1)
        mlab.points3d(det_center[0], det_center[1], det_center[2], [0.5], color=(1.0, 0.0, 0.0), scale_mode='scalar', scale_factor=1)

        # Plot detector grid
        if plot_det_grid:
            for i in range(det_rows+1):
                det_row = np.vstack((det_topleft + det_v*i, det_topright + det_v*i))
                mlab.plot3d(det_row[:, 0], det_row[:, 1], det_row[:, 2],
                                color=(1.0, 0.0, 0.0), tube_radius=None, line_width=3.5)

            for i in range(det_cols+1):
                det_col = np.vstack((det_topleft + det_u*i, det_downleft + det_u*i))
                mlab.plot3d(det_col[:, 0], det_col[:, 1], det_col[:, 2],
                                color=(1.0, 0.0, 0.0), tube_radius=None, line_width=3.5)
        else:
            # Plot detector edges
            detector_rect = np.vstack((det_topleft, det_downleft, det_downright, det_topright, det_topleft))
            mlab.plot3d(detector_rect[:, 0], detector_rect[:, 1], detector_rect[:, 2],
                            color=(1.0, 0.0, 0.0), tube_radius=None, line_width=3.5)


        source_det_line = np.vstack((views[j, :3], views[j, 3:6]))

        mlab.plot3d(source_det_line[:, 0], source_det_line[:, 1], source_det_line[:, 2],
                    color=(1.0, 0.0, 0.0), tube_radius=None, line_width=2.0)

        mlab.plot3d(projection_cone_1[:, 0], projection_cone_1[:, 1], projection_cone_1[:, 2],
                      color=(1.0, 0.0, 0.0), tube_radius=None, line_width=2.0)
        mlab.plot3d(projection_cone_2[:, 0], projection_cone_2[:, 1], projection_cone_2[:, 2],
                        color=(1.0, 0.0, 0.0), tube_radius=None, line_width=2.0)
        mlab.plot3d(projection_cone_3[:, 0], projection_cone_3[:, 1], projection_cone_3[:, 2],
                        color=(1.0, 0.0, 0.0), tube_radius=None, line_width=2.0)
        mlab.plot3d(projection_cone_4[:, 0], projection_cone_4[:, 1], projection_cone_4[:, 2],
                        color=(1.0, 0.0, 0.0), tube_radius=None, line_width=2.0)

def _plot_mask_face_mayavi(mask, mask_face_left_edge, det_u, det_cols):
        det_u_dir = det_u / np.linalg.norm(det_u)
        for i in range(det_cols+1):
            mask_bar_topleft   = mask_face_left_edge[0] + det_u*i - det_u_dir*mask['bar_width'] * 0.5
            mask_bar_downleft  = mask_face_left_edge[1] + det_u*i - det_u_dir*mask['bar_width'] * 0.5
            mask_bar_downright = mask_face_left_edge[1] + det_u*i + det_u_dir*mask['bar_width'] * 0.5
            mask_bar_topright  = mask_face_left_edge[0] + det_u*i + det_u_dir*mask['bar_width'] * 0.5
            
            mask_bar = np.vstack((mask_bar_topleft, mask_bar_downleft, mask_bar_downright, mask_bar_topright, mask_bar_topleft))
            mlab.plot3d(mask_bar[:, 0], mask_bar[:, 1], mask_bar[:, 2],
                        color=(0.0, 0.0, 1.0), tube_radius=None, line_width=3.5)

def plot_mask_mayavi(views, det_rows, det_cols, mask):
    for j in range(views.shape[0]):
        source     = views[j, :3]
        det_center = views[j, 3:6]
        det_u      = views[j, 6:9]
        det_v      = views[j, 9:12]
        det_u_dir = det_u / np.linalg.norm(det_u)
        M = np.linalg.norm(det_center - source) / (np.linalg.norm(det_center - source) - mask['mask_det_dist'])
        det_u_scaled = det_u / M
        det_v_scaled = det_v / M
        det_v_dir = det_v / np.linalg.norm(det_v)
        inv_projection_dir = -(det_center - source) / np.linalg.norm(det_center - source)
        mask_front_topleft_corner   = det_center + \
                                      inv_projection_dir*mask['mask_det_dist'] + \
                                      det_u_dir * mask['mask_offset'] - \
                                      0.5*det_cols*det_u_scaled - \
                                      0.5*det_rows*det_v_scaled
        mask_front_downleft         = det_center + \
                                      inv_projection_dir*mask['mask_det_dist'] + \
                                      det_u_dir * mask['mask_offset'] - \
                                      0.5*det_cols*det_u_scaled + \
                                      0.5*det_rows*det_v_scaled

        mask_back_topleft_corner   = det_center + \
                                      inv_projection_dir*mask['mask_det_dist'] - \
                                      inv_projection_dir*mask['bar_thickness'] + \
                                      det_u_dir * mask['mask_offset'] - \
                                      0.5*det_cols*det_u_scaled - \
                                      0.5*det_rows*det_v_scaled
        mask_back_downleft         = det_center + \
                                      inv_projection_dir*mask['mask_det_dist'] - \
                                      inv_projection_dir*mask['bar_thickness'] + \
                                      det_u_dir * mask['mask_offset'] - \
                                      0.5*det_cols*det_u_scaled + \
                                      0.5*det_rows*det_v_scaled

    front_face_left_edge = np.vstack((mask_front_topleft_corner, mask_front_downleft))
    _plot_mask_face_mayavi(mask, front_face_left_edge, det_u_scaled, det_cols)
    back_face_left_edge = np.vstack((mask_back_topleft_corner, mask_back_downleft))
    _plot_mask_face_mayavi(mask, back_face_left_edge, det_u_scaled, det_cols)

def plot_rays_mayavi(ray_paths: np.ndarray, ray_lengths: np.ndarray, plot_ray_points=False):
    for k in range(ray_paths.shape[0]):
        # print(f"Ray path [{k}] with length {ray_lengths[k]}:\n{ray_paths[k]}")
        # Plot rays
        mlab.plot3d(ray_paths[k , :ray_lengths[k] , 0], ray_paths[k , :ray_lengths[k] , 1], ray_paths[k , :ray_lengths[k] , 2],
                    # np.repeat(10.0, ray_paths[k, :ray_lengths[k]].shape[0]),
                    line_width = 2,
                    color=(0.0, 0.5, 0.5), tube_radius=None, representation='wireframe')
        # Plot detector hit points
        mlab.points3d(ray_paths[k , ray_lengths[k]-1 , 0], ray_paths[k , ray_lengths[k]-1 , 1], ray_paths[k , ray_lengths[k]-1, 2],
                      [0.5], color=(.0, 0.5, 0.5), scale_mode='scalar', scale_factor=1)
        if plot_ray_points:
            mlab.points3d(ray_paths[k , :ray_lengths[k] , 0], ray_paths[k , :ray_lengths[k] , 1], ray_paths[k , :ray_lengths[k] , 2],
                          np.repeat(0.5, ray_paths[k, :ray_lengths[k]].shape[0]), color=(.0, 0.6, 0.6), scale_mode='scalar', scale_factor=1)