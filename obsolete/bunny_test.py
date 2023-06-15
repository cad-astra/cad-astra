#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:59:26 2021

@author: pparamonov
"""
import numpy as np
import trimesh
import matplotlib.pyplot as plt
# %% Read STL
mesh = trimesh.load("../examples/data/Stanford_Bunny.stl")

vertices = np.asarray(mesh.vertices, dtype=np.float32)*180 # times 180 to scale the mesh up
faces = np.asarray(mesh.faces, dtype=np.int32)
normals = np.asarray(mesh.face_normals, dtype=np.float32)

# %% Plot triangles...
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def plot_triang(triangle, axes3d, color='green', asplane=False):
    triangle_extended = np.vstack((triangle, triangle[0, :]))
    if asplane:
        verts = [triangle]
        axes3d.add_collection3d(Poly3DCollection(verts, facecolor=color,
                                                 edgecolor=color))
    axes3d.plot3D(triangle_extended[:, 0], triangle_extended[:, 1],
                  triangle_extended[:, 2], color, marker='o')

fig = plt.Figure()
ax = plt.axes(projection="3d")
# faces_inds = [82307]
# faces_inds = [82170, 82155]
faces_inds = [82303, 82307, 82308, 82321, 82335]
triangles = vertices[faces[faces_inds]]
for i in range(triangles.shape[0]):
    plot_triang(triangles[i], ax, "red", asplane=False)
    triangle_center = np.mean(triangles[i], axis=0)
    ax.plot3D(triangle_center[0], triangle_center[1], triangle_center[2],
              "blue", marker='x')
    norm_coords = np.vstack((triangle_center, triangle_center+normals[faces_inds[i]]))
    ax.plot3D(norm_coords[:, 0], norm_coords[:, 1], norm_coords[:, 2], "blue")

source = np.array([-5000.00, 0.00, 0.00])
pixel  = np.array([200.00, 137.50, 80.50])
ray_coords = np.vstack((source + (pixel-source)*0.9495, pixel - (pixel-source)*0.0490))
ax.plot3D(ray_coords[:, 0], ray_coords[:, 1], ray_coords[:, 2],
          "green", linewidth=2)
# ax.set_xlim([-63, -57])
# ax.set_ylim([129, 132])
# ax.set_zlim([74, 78])
plt.show()
# %% Plot projected triangle
def cross_2d(a,b):
    return a[0]*b[1] - a[1]*b[0]

A=np.array([-128.46064758, -71.08475494], dtype=np.float32)
B=np.array([-129.70030212, -72.78107452], dtype=np.float32)
C=np.array([-129.63026428, -72.67737579], dtype=np.float32)

triang_2d = np.vstack((A, B, C, A))
plt.Figure()
plt.plot(triang_2d[:, 0], triang_2d[:, 1], marker='o', color='blue')

det_pos = np.array([-129.50, -72.50], dtype=np.float32)

plt.plot(det_pos[0], det_pos[1], marker='o', color='red')

A=np.array([-128.46064758, -71.08475494], dtype=np.float32)
B=np.array([-129.63026428, -72.67737579], dtype=np.float32)
C=np.array([-128.10983276, -71.04455566], dtype=np.float32)

triang_2d = np.vstack((A, B, C, A))
plt.plot(triang_2d[:, 0], triang_2d[:, 1], color='green', marker='o')
plt.show()