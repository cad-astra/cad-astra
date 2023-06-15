#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:28:47 2023

@author: pparamonov
"""
# %% Import
from matplotlib import pyplot as plt

import numpy as np

# %% Bilinear for unit pixel
def bilinear_unit_pixel(P, O, O_diag):
    # Same as bilinear, but with assumption that pixel size is 1.0.
    # In this case, we're interested only in the denominator sign.
    # Thus, multiplication can be used as a faster version.
    return (O_diag[0] - P[0]) * (O_diag[1] - P[1]) * ( (O_diag[0] - O[0]) * (O_diag[1] - O[1]))


N= 10
x = y = np.linspace(0.5, 1.5, N)
step = x[1]-x[0]

bilin_f = np.zeros(shape=(N,N))

for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        bilin_f[i,j] = bilinear_unit_pixel((x[i], y[j]), (0.5, 0.5), (1.5, 1.5))
        
plt.figure(figsize=(8,8))
plt.imshow(bilin_f, extent=(y[0]+step*0.5, y[-1]-step*0.5, x[-1]-step*0.5, x[0]+step*0.5))
plt.colorbar()

# %%

N= 101

pix_ij = (10, 12) # an arbitrary detector pixel...

x = np.linspace(pix_ij[1]-1, pix_ij[1]+1, N)
y = np.linspace(pix_ij[0]-1, pix_ij[0]+1, N)
step = x[1]-x[0]

def w_fun(i, j, x_q, y_q):
    f_i = i-y_q >= 0 #np.ceil(i-y_q)
    f_j = j-x_q >= 0 #np.ceil(j-x_q)
    # print(f"i={i}, j={j}, x_q={x_q}, y_q={y_q}, f_i={f_i}, f_j={f_j}")
    return np.abs( (i+((-1)**f_i) - y_q) * (j+((-1)**f_j) - x_q) )

ray_weights = np.zeros(shape=(N,N))

for i in range(y.shape[0]):
    for j in range(x.shape[0]):
        ray_weights[i,j] = w_fun(pix_ij[0], pix_ij[1], x[j], y[i])
        
plt.figure(figsize=(8,8))
plt.imshow(ray_weights, extent=(pix_ij[1]-1.0, pix_ij[1]+1.0, pix_ij[0]-1.0, pix_ij[0]+1.0), origin="lower")
plt.colorbar()
plt.grid(True)
plt.show()

 # %%

S = np.array([0, 0, 0])
P = np.array([3, 0, 0])

u = np.array([0, 0.5,  0])
v = np.array([0, 0.0, -1])
q = np.cross(u, v)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

ax.plot3D(*P, marker='o') # Detector origin
ax.plot3D(*np.vstack((P,P+u)).T, color="red", label="u") # u
ax.plot3D(*np.vstack((P,P+v)).T, color="green", label="v") # v
ax.plot3D(*np.vstack((P,P+q)).T, color="blue", label="q") # q

# V = np.array([1.0, -1, 0])

R = np.hstack((np.reshape(u, (-1,1)), np.reshape(v, (-1,1)), np.reshape(q, (-1,1))))

V_det = np.array([1, 1, 0])

V_glob = (R @ V_det) + P

ax.plot3D(*V_glob, marker='o', label="V") # Projection direction - from Source to Detector
plt.legend()

V_det_back = np.linalg.inv(R) @ (V_glob - P)