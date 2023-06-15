#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:12:21 2022

@author: pparamonov
"""

import numpy as np
from matplotlib import pyplot as plt

A=np.array([104.25322723, -26.00683784, -75.62353516])
B=np.array([99.22764587, -24.36729431, -75.62353516])
C=np.array([102.24301910, -26.22885513, -78.13638306])
normal=np.array([-0.29434374, -0.90223187, 0.31518152])
ray_origin=np.array([102.26112366, -25.54774475, -76.17214203])
ray_direction=np.array([0.60411453, -0.79318893, 0.07679018])
hit_point=np.array([102.26194000, -25.54881287, -76.17204285])
new_origin = np.array([102.26136780, -25.54925156, -76.17143250])

# A=np.array([-77.34245300, 29.00472450, -108.28996277])
# B=np.array([-77.92488098, 29.00472450, -108.52872467])
# C=np.array([-77.70564270, 27.74832916, -108.28996277])
# normal=np.array([-0.37704784, 0.10899427, 0.91975820])
# ray_origin=np.array([-77.56267548, 28.62616158, -108.33747864])
# ray_direction=np.array([0.23900206, 0.97096205, -0.01052115])
# hit_point=np.array([-77.48626709, 28.93654633, -108.34085083])
# new_origin = np.array([-77.48699951, 28.93659782, -108.33905792])

# A=np.array([-86.71970367, 100.25205994, 110.32376099])
# B=np.array([-86.71970367, 99.36317444, 109.60699463])
# C=np.array([-85.46327972, 99.92041016, 110.32376099])
# normal=np.array([-0.16346416, -0.61926854, 0.76797527])
# ray_origin=np.array([-86.47039795, 99.31868744, 110.27524567])
# ray_direction=np.array([-0.01683369, 0.99963206, 0.02126972])
# hit_point=np.array([-86.48442841, 100.15176392, 110.29296875])
# new_origin = np.array([-86.56615448, 99.84213257, 110.67695618])

# A=np.array([-114.36052704, -42.33623886, 3.52969313])
# B=np.array([-114.36052704, -41.60848618, 2.27325249])
# C=np.array([-113.73649597, -42.61015320, 3.52969313])
# normal=np.array([-0.35507774, -0.80893737, -0.46855089])
# ray_origin=np.array([-114.12065887, -42.54290390, 3.21370816])
# ray_direction=np.array([-0.02240837, 0.99974871, 0.00065741])
# hit_point=np.array([-114.12709045, -42.25578690, 3.21389699])
# new_origin = np.array([-114.30462646, -42.45801926, 3.20657611])

triangle   = np.vstack((A, B, C, A))
normal_vec = np.vstack((A, normal))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot3D(triangle[:, 0], triangle[:, 1], triangle[:, 2])
ax.plot3D(hit_point[0], hit_point[1], hit_point[2], marker='x')
ax.quiver(A[0], A[1], A[2], normal[0], normal[1], normal[2], length=0.1, normalize=True)
ax.plot3D(ray_origin[0], ray_origin[1], ray_origin[2], marker='o', color='r')
ax.quiver(ray_origin[0], ray_origin[1], ray_origin[2],
          ray_direction[0], ray_direction[1], ray_direction[2],
          length=0.1, normalize=True, color='r')
ax.plot3D(new_origin[0], new_origin[1], new_origin[2], marker='*', color='r')

# %%
import trimesh
import numpy as np
from matplotlib import pyplot as plt

def vertices_faces_2triangles(vertices: np.ndarray, faces: np.ndarray):
    triangles = np.empty((faces.shape[0], 9), dtype=np.float32)
    for i, face in enumerate(faces):
        triangles[i] = np.concatenate([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
    return triangles

mesh_dir = '/home/pparamonov/Projects/mesh-fp-prototype/examples/data/'

sample_mask_mesh = trimesh.load(mesh_dir + "sphere-midpoly.stl")

sample_mask_vertices  = np.asarray(sample_mask_mesh.vertices, dtype=np.float32)

source = np.array([-10.00000000, 0.99000001, 0.00000000])
hit_point0 = np.array([-0.01014002, 0.99899995, -0.00000000])
hit_point1 = np.array([0.94663483, 0.28388819, -0.07042767])
hit_point2 = np.array([0.55135024, -0.83343744, -0.00258491])
hit_point3 = np.array([-0.80904317, -0.51341724, 0.26308468])
at_detector = np.array([-11.61808586, 6.65641403, 1.49023771])
ray_path1 = np.vstack((source, hit_point0, hit_point1, hit_point2, hit_point3, at_detector))

source = np.array([-10.00000000, 0.94999999, 0.00000000])
hit_point0 = np.array([-0.29657164, 0.94999999, -0.00000000])
hit_point1 = np.array([0.94434601, 0.29247350, -0.06734656])
at_detector = np.array([9.99999905, -12.61081409, -0.09038536])

ray_path2 = np.vstack((source, hit_point0, hit_point1, at_detector))

source = np.array([-10.00000000, 0.99900001, 0.00000000])
hit_point0 = np.array([-0.01014002, 0.99899995, -0.00000000])
at_detector = np.array([10.00000095, 2.97081470, 0.19419385])
ray_path3 = np.vstack((source, hit_point0, at_detector))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot_trisurf(sample_mask_vertices[:, 0], sample_mask_vertices[:, 1], sample_mask_vertices[:, 2],
                triangles=sample_mask_mesh.faces, alpha=0.5)

# ax.plot3D(ray_path1[:, 0], ray_path1[:, 1], ray_path1[:, 2], color='r', marker='o')
# ax.plot3D(ray_path2[:, 0], ray_path2[:, 1], ray_path2[:, 2], color='r', marker='o')
ax.plot3D(ray_path3[:, 0], ray_path3[:, 1], ray_path3[:, 2], color='r', marker='o')

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# %%

sample_mesh = trimesh.load(mesh_dir + "sphere-midpoly.stl")

sample_vertices  = 2*np.asarray(sample_mask_mesh.vertices, dtype=np.float32)

# source = np.array([-10.00000000, 1.04999995, 0.00000000])
# hit_point0 = np.array([-1.69561708, 1.04999995, -0.00000000])
# hit_point1 = np.array([-0.19876528, 0.97813880, -0.01494161])
# hit_point2 = np.array([0.98095685, 0.13600673, 0.05733873])
# hit_point3 = np.array([-0.06581298, -0.98441958, -0.09237304])
# hit_point4 = np.array([-1.44360340, -1.34985280, -0.18476498])
# at_detector = np.array([-12.88720608, -3.82830477, -0.89227152])

# ray_path = np.vstack((source, hit_point0, hit_point1, hit_point2, hit_point3, hit_point4, at_detector))

source = np.array([-10.00000000, 0.99900001, 0.00000000])
hit_point0 = np.array([-1.72287726, 0.99899995, 0.00000000])
# material_attenuation_coeff = 0.00, n1 = 1.00, n2 = 1.01, material stack size: 2
hit_point1 = np.array([-0.34799764, 0.93299443, -0.01372408])
# material_attenuation_coeff = 1.00, n1 = 1.01, n2 = 0.90, material stack size: 2
hit_point2 = np.array([0.83718729, 1.79559410, -0.12395102])
# material_attenuation_coeff = 1.00, n1 = 1.01, n2 = 1.00, material stack size: 1
at_detector = np.array([10.00000000, 7.80259800, -0.90958416])
ray_path = np.vstack((source, hit_point0, hit_point1, hit_point2, at_detector))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot_trisurf(sample_mask_vertices[:, 0], sample_mask_vertices[:, 1], sample_mask_vertices[:, 2],
                triangles=sample_mask_mesh.faces, alpha=0.6, color='b')
ax.plot_trisurf(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2],
                triangles=sample_mask_mesh.faces, alpha=0.5, color='g')

ax.plot3D(ray_path[:, 0], ray_path[:, 1], ray_path[:, 2], color='r', marker='o')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# %%
import numpy as np
from matplotlib import pyplot as plt
import trimesh
mesh_dir = '/home/pparamonov/Projects/mesh-fp-prototype/examples/data/'
sample_mesh = trimesh.load(mesh_dir + "sample_mask.stl")

sample_vertices  = np.asarray(sample_mesh.vertices, dtype=np.float32)

triang10 = np.array([[1.00000000, -1.00000000, -1.00000000],
                     [0.00000000, -1.00000000,  1.00000000],
                     [-1.00000000, -1.00000000, -1.00000000]])
hit_point10 = np.array([[-0.25000000, -1.00000000, 0.00000000]])
origin=np.array([0.00000000, -10.00000000, 0.00000000])
direction=np.array([0.00000000, 20.00000000, 0.00000000])

# triangs = np.vstack((triang10))
# hit_points = np.vstack((hit_point10))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot3D(triang10[:, 0], triang10[:, 1], triang10[:, 2])
ax.plot3D(hit_point10[:, 0], hit_point10[:, 1], hit_point10[:, 2], marker='x')
# for i in range(1):
#     ax.plot3D(triangs[i*3:(i+1)*3, 0], triangs[i*3:(i+1)*3, 1], triangs[i*3:(i+1)*3, 2])
#     ax.plot3D(hit_points[i:(i+1), 0], hit_points[i:(i+1), 1], hit_points[i:(i+1), 2], marker='x')

# %%
import numpy as np
from matplotlib import pyplot as plt

hit_point_trans = np.array([[-0.72029054, -0.23774858, 0.18453725],
                      [-0.73331666, -0.15162544, 0.18787453],
                      [-0.73997462, -0.10760600, 0.18958028],
                      [-0.74008638, -0.10686733, 0.18960889],
                      # [-0.74009347, -0.10682022, 0.18961070],
                      [-0.74023134, -0.10599938, 0.18964411],
                      [-0.75462312, -0.01084727, 0.19333126]])

triangles_trans  = np.array([[[-0.72096401, -0.23099793, 0.19410951], [-0.72415543, -0.23127872, 0.18014954], [-0.71826601, -0.24477470, 0.18014954]],
                             [[-0.73108882, -0.14780517, 0.19410951], [-0.73616332, -0.14825162, 0.18014954], [-0.73093951, -0.16180590, 0.19410951]],
                             [[-0.73688197, -0.11328001, 0.19410951], [-0.74052382, -0.10659347, 0.18877827], [-0.73995811, -0.11302353, 0.18712963]],
                             [[-0.73805058, -0.10637588, 0.19410951], [-0.74052382, -0.10659347, 0.18877827], [-0.73688197, -0.11328001, 0.19410951]],
                             [[-0.73805058, -0.10637588, 0.19410951], [-0.74052382, -0.10659347, 0.18877827], [-0.73688197, -0.11328001, 0.19410951]],
                             [[-0.73805058, -0.10637588, 0.19410951], [-0.74113554, -0.09964035, 0.19036140], [-0.74052382, -0.10659347, 0.18877827]],
                             [[-0.75400209, -0.01668869, 0.19410951], [-0.75435555, -0.00971284, 0.19410951], [-0.75642645, -0.01490000, 0.18852188]]])

front_faces = [0, 2, 4]
back_faces = [1, 3, 5]
hit_point_trans_front = hit_point_trans[front_faces]
hit_point_trans_back  = hit_point_trans[back_faces]

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# ax.plot3D(triangles_trans[:, 0], triangles_trans[:, 1], triangles_trans[:, 2])
for triang in triangles_trans[:]:
    polygon = np.vstack((triang, triang[0]))
    ax.plot3D(polygon[:, 0], polygon[:, 1], polygon[:, 2], alpha=0.5)
    
ax.plot3D(hit_point_trans[:, 0], hit_point_trans[:, 1], hit_point_trans[:, 2], linestyle=':')
ax.plot3D(hit_point_trans_front[:, 0], hit_point_trans_front[:, 1], hit_point_trans_front[:, 2], marker='x', linestyle='none')
ax.plot3D(hit_point_trans_back[:, 0], hit_point_trans_back[:, 1], hit_point_trans_back[:, 2], marker='o', linestyle='none')
ax.set_title('Mesh transformation')

hit_point_rotate = np.array([[-0.73835546, -0.17370671, 0.18453723],
                             [-0.74378353, -0.08677281, 0.18787453],
                             [-0.74655777, -0.04234221, 0.18958017],
                             [-0.74662358, -0.04128766, 0.18962066],
                             [-0.75266409,  0.05545466, 0.19333448]])
triangles_rotate  = np.array([[[-0.73843479, -0.16692278, 0.19410951], [-0.74163854, -0.16692278, 0.18014954], [-0.73695463, -0.18088299, 0.18014954]],
                              [[-0.74122953, -0.08316277, 0.19410951], [-0.74632359, -0.08316277, 0.18014954], [-0.74230778, -0.09712271, 0.19410951]],
                              [[-0.74397457, -0.04826275, 0.19410951], [-0.74701631, -0.04128277, 0.18877827], [-0.74701631, -0.04773766, 0.18712963]],
                              [[-0.74453354, -0.04128277, 0.19410951], [-0.74701631, -0.04128277, 0.18877827], [-0.74397457, -0.04826275, 0.19410951]],
                              [[-0.75256336, 0.04945733, 0.19410951], [-0.75230408, 0.05643731, 0.19410951], [-0.75482166, 0.05145161, 0.18852188]]])

front_faces = [0, 2]
back_faces = [1, 3, 4]
hit_point_rotate_front = hit_point_rotate[front_faces]
hit_point_rotate_back  = hit_point_rotate[back_faces]

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# ax.plot3D(triangles_trans[:, 0], triangles_trans[:, 1], triangles_trans[:, 2])
for triang in triangles_rotate[:]:
    polygon = np.vstack((triang, triang[0]))
    ax.plot3D(polygon[:, 0], polygon[:, 1], polygon[:, 2], alpha=0.5)
    
ax.plot3D(hit_point_rotate[:, 0], hit_point_rotate[:, 1], hit_point_rotate[:, 2], linestyle=':')
ax.plot3D(hit_point_rotate_front[:, 0], hit_point_rotate_front[:, 1], hit_point_rotate_front[:, 2], marker='x', linestyle='none')
ax.plot3D(hit_point_rotate_back[:, 0], hit_point_rotate_back[:, 1], hit_point_rotate_back[:, 2], marker='o', linestyle='none')

ax.set_title('Projector rotation')

# %%
sample_mesh = trimesh.load(mesh_dir + "cube_loopcut.stl")

sample_vertices  = np.asarray(sample_mesh.vertices, dtype=np.float32)

# source = np.array([-10.00000000, 1.04999995, 0.00000000])
# hit_point0 = np.array([-1.69561708, 1.04999995, -0.00000000])
# hit_point1 = np.array([-0.19876528, 0.97813880, -0.01494161])
# hit_point2 = np.array([0.98095685, 0.13600673, 0.05733873])
# hit_point3 = np.array([-0.06581298, -0.98441958, -0.09237304])
# hit_point4 = np.array([-1.44360340, -1.34985280, -0.18476498])
# at_detector = np.array([-12.88720608, -3.82830477, -0.89227152])

# ray_path = np.vstack((source, hit_point0, hit_point1, hit_point2, hit_point3, hit_point4, at_detector))

source = np.array([-10.00000000, 0.99900001, 0.00000000])
hit_point0 = np.array([-1.72287726, 0.99899995, 0.00000000])
# material_attenuation_coeff = 0.00, n1 = 1.00, n2 = 1.01, material stack size: 2
hit_point1 = np.array([-0.34799764, 0.93299443, -0.01372408])
# material_attenuation_coeff = 1.00, n1 = 1.01, n2 = 0.90, material stack size: 2
hit_point2 = np.array([0.83718729, 1.79559410, -0.12395102])
# material_attenuation_coeff = 1.00, n1 = 1.01, n2 = 1.00, material stack size: 1
at_detector = np.array([10.00000000, 7.80259800, -0.90958416])
ray_path = np.vstack((source, hit_point0, hit_point1, hit_point2, at_detector))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot_trisurf(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2],
                triangles=sample_mesh.faces, alpha=0.5, color='g')

# ax.plot3D(ray_path[:, 0], ray_path[:, 1], ray_path[:, 2], color='r', marker='o')
# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
# ax.set_zlim([-10, 10])

# %%

hit_point1     = np.array([47.99081802, 0.00000005, 4.43807030]) ## is front face: 1])
hit_point2     = np.array([47.99082184, 0.00000005, 4.43807030]) #is front face: 0])

ray_direction1 = np.array([0.99999362, 0.00000000, 0.00354428])
ray_direction2 = np.array([0.99999362, 0.00000000, 0.00354428])

triangle1      = np.array([[47.99082184, -0.10000000, 4.43806934], [48.00227737, -0.10000000, 4.44779778], [47.99082184, 0.10000000, 4.43806934]])
triangle2      = np.array([[47.97786713, 0.10000000, 4.44567394], [47.99082184, -0.10000000, 4.43806934], [47.99082184, 0.10000000, 4.43806934]])

hp = np.vstack((hit_point1, hit_point2))
rd = np.vstack((ray_direction1, ray_direction2))
polygons = np.vstack((triangle1, triangle2))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')

# ax.plot3D(hp[:, 0], hp[:, 1], hp[:, 2], linestyle=':', marker='x')

ax.plot3D(hit_point1[ 0], hit_point1[ 1], hit_point1[ 2], linestyle=':', marker='x')
ax.plot3D(hit_point2[ 0], hit_point2[ 1], hit_point2[ 2], linestyle=':', marker='o')

# ax.plot3D(polygons[:,0], polygons[:,1], polygons[:, 2])

ax.quiver(hp[:,0], hp[:,1], hp[:,2],
          rd[:,0], rd[:,1], rd[:,2],
          length=1e-6, normalize=True, color='r')

# ax.set_xlim([-0.01, 0.01])
# ax.set_ylim([-0.1, 0.1])
# ax.set_zlim([4.35, 4.5])

# %%

# hit_point     = np.array([-3.32614779, -0.00000000, -17.17678452, 1])
# hit_points    = hit_point.copy()
# ray_direction = np.array([1800.00000000, 0.00000000, -25.83679199])
# triangle      = np.array([[-1.71529996, 0.10000000, -17.41573334], [-3.41408062, -0.10000000, -17.16374207], [-3.41408062, 0.10000000, -17.16374207]])
# poligons      = triangle.copy()
# # barycentrics  = np.array([0.50000000, 0.44823772, 0.05176228])
# # t_hit         = 0.66481882
# # hit_point-origin = np.array([1196.67382812, -0.00000000, -17.17678452]
# # <hit_point-origin, ray_direction> = 2154456.75000000
# hit_point     = np.array([-2.67018199, 0.00000000, -17.18620300, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989694, 0.00000000, -0.01435599])
# triangle      = np.array([[-2.67015147, -0.10000000, -17.18619537], [-2.70361423, -0.10000000, -17.19403076], [-2.67015147, 0.10000000, -17.18619537]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.00091235, 0.50000000, 0.49908763])
# # t_hit         = 0.65603346
# # hit_point-origin = np.array([0.65596581, 0.00000000, -0.00941849]
# # <hit_point-origin, ray_direction> = 0.65603340
# hit_point     = np.array([-2.66909313, -0.00000000, -17.18621826, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989694, 0.00000000, -0.01435795])
# triangle      = np.array([[-2.63579226, 0.10000000, -17.18695068], [-2.67015147, -0.10000000, -17.18619537], [-2.67015147, 0.10000000, -17.18619537]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.50000000, 0.46919951, 0.03080049])
# # t_hit         = 0.00108891
# # hit_point-origin = np.array([0.00108886, -0.00000000, -0.00001526]
# # <hit_point-origin, ray_direction> = 0.00108896
# hit_point     = np.array([-2.58962274, 0.00000000, -17.18736458, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989599, 0.00000000, -0.01442210])
# triangle      = np.array([[-2.59412694, 0.10000000, -17.16120720], [-2.58865285, -0.10000000, -17.19299698], [-2.58865285, 0.10000000, -17.19299698]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.50000000, 0.32282144, 0.17717856])
# # t_hit         = 0.07947866
# # hit_point-origin = np.array([0.07947040, 0.00000000, -0.00114632]
# # <hit_point-origin, ray_direction> = 0.07947867
# hit_point     = np.array([-2.51134276, 0.00000001, -17.18849373, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442200])
# triangle      = np.array([[-2.50948548, -0.10000000, -17.19927979], [-2.51495934, -0.10000000, -17.16749191], [-2.50948548, 0.10000000, -17.19927979]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.33931613, 0.50000006, 0.16068381])
# # t_hit         = 0.07828803
# # hit_point-origin = np.array([0.07827997, 0.00000001, -0.00112915]
# # <hit_point-origin, ray_direction> = 0.07828812
# hit_point     = np.array([-1.84142339, 0.00000001, -17.19815636, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442209])
# triangle      = np.array([[-1.84062612, -0.10000000, -17.19631004], [-1.84620237, -0.10000000, -17.20922279], [-1.84062612, 0.10000000, -17.19631004]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.14298485, 0.50000006, 0.35701507])
# # t_hit         = 0.66998893
# # hit_point-origin = np.array([0.66991937, 0.00000000, -0.00966263]
# # <hit_point-origin, ray_direction> = 0.66998911
# hit_point     = np.array([-1.81524146, 0.00000001, -17.19853401, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989599, 0.00000000, -0.01442227])
# triangle      = np.array([[-1.81395912, -0.10000000, -17.20327187], [-1.81702244, -0.10000000, -17.19195366], [-1.81395912, 0.10000000, -17.20327187]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.41860572, 0.50000006, 0.08139426])
# # t_hit         = 0.02618467
# # hit_point-origin = np.array([0.02618194, -0.00000000, -0.00037766]
# # <hit_point-origin, ray_direction> = 0.02618466
# hit_point     = np.array([-1.55268216, 0.00000001, -17.20232201, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989593, 0.00000000, -0.01442242])
# triangle      = np.array([[-1.54064000, 0.10000000, -17.18709564], [-1.55893898, -0.10000000, -17.21023178], [-1.55893898, 0.10000000, -17.21023178]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999994, 0.15808317, 0.34191689])
# # t_hit         = 0.26258653
# # hit_point-origin = np.array([0.26255929, 0.00000000, -0.00378799]
# # <hit_point-origin, ray_direction> = 0.26258659
# hit_point     = np.array([-1.43992114, 0.00000002, -17.20394897, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989599, 0.00000000, -0.01442279])
# triangle      = np.array([[-1.44897223, -0.10000000, -17.21539116], [-1.43067312, -0.10000000, -17.19225693], [-1.44897223, 0.10000000, -17.21539116]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49462011, 0.50000012, 0.00537980])
# # t_hit         = 0.11277278
# # hit_point-origin = np.array([0.11276102, 0.00000001, -0.00162697]
# # <hit_point-origin, ray_direction> = 0.11277276
# hit_point     = np.array([-1.36848843, 0.00000003, -17.20497894, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989599, 0.00000000, -0.01442242])
# triangle      = np.array([[-1.36911547, -0.10000000, -17.20355034], [-1.35889530, -0.10000000, -17.22684097], [-1.36911547, 0.10000000, -17.20355034]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.06135410, 0.50000018, 0.43864572])
# # t_hit         = 0.07144015
# # hit_point-origin = np.array([0.07143271, 0.00000001, -0.00102997]
# # <hit_point-origin, ray_direction> = 0.07144014
# hit_point     = np.array([-1.32001865, 0.00000004, -17.20567703, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989593, 0.00000000, -0.01442222])
# triangle      = np.array([[-1.33014762, 0.10000000, -17.22721481], [-1.31932294, -0.10000000, -17.20419884], [-1.31932294, 0.10000000, -17.20419884]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999982, 0.43573087, 0.06426930])
# # t_hit         = 0.04847483
# # hit_point-origin = np.array([0.04846978, 0.00000000, -0.00069809]
# # <hit_point-origin, ray_direction> = 0.04847481
# hit_point     = np.array([-0.65593779, 0.00000005, -17.21525574, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989599, 0.00000000, -0.01442200])
# triangle      = np.array([[-0.65517843, 0.10000000, -17.19354248], [-0.65593976, -0.10000000, -17.21531296], [-0.65593976, 0.10000000, -17.21531296]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999979, 0.49735662, 0.00264362])
# # t_hit         = 0.66415000
# # hit_point-origin = np.array([0.66408086, 0.00000001, -0.00957870]
# # <hit_point-origin, ray_direction> = 0.66414994
# hit_point     = np.array([-0.54294288, 0.00000005, -17.21688461, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989599, 0.00000000, -0.01442201])
# triangle      = np.array([[-0.52967238, 0.10000000, -17.23717499], [-0.54582775, -0.10000000, -17.21247482], [-0.54582775, 0.10000000, -17.21247482]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999976, 0.32143065, 0.17856959])
# # t_hit         = 0.11300666
# # hit_point-origin = np.array([0.11299491, 0.00000000, -0.00162888]
# # <hit_point-origin, ray_direction> = 0.11300665
# hit_point     = np.array([-0.47360247, 0.00000005, -17.21788597, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989599, 0.00000000, -0.01442235])
# triangle      = np.array([[-0.47065431, -0.10000000, -17.21247101], [-0.47708994, -0.10000000, -17.22428894], [-0.47065431, 0.10000000, -17.21247101]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.45809713, 0.50000024, 0.04190266])
# # t_hit         = 0.06934765
# # hit_point-origin = np.array([0.06934041, -0.00000000, -0.00100136]
# # <hit_point-origin, ray_direction> = 0.06934763
# hit_point     = np.array([-0.44801757, 0.00000005, -17.21825600, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442263])
# triangle      = np.array([[-0.44764599, -0.10000000, -17.22105408], [-0.44909987, -0.10000000, -17.21010208], [-0.44764599, 0.10000000, -17.22105408]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.25557590, 0.50000024, 0.24442387])
# # t_hit         = 0.02558757
# # hit_point-origin = np.array([0.02558491, 0.00000000, -0.00037003]
# # <hit_point-origin, ray_direction> = 0.02558758
# hit_point     = np.array([-0.39808303, 0.00000006, -17.21897697, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442270])
# triangle      = np.array([[-0.41327980, -0.10000000, -17.21245003], [-0.37901452, -0.10000000, -17.22716522], [-0.41327980, 0.10000000, -17.21245003]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.44350314, 0.50000030, 0.05649656])
# # t_hit         = 0.04993971
# # hit_point-origin = np.array([0.04993454, 0.00000001, -0.00072098]
# # <hit_point-origin, ray_direction> = 0.04993974
# hit_point     = np.array([-0.30213103, 0.00000007, -17.22036171, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442154])
# triangle      = np.array([[-0.31292424, 0.10000000, -17.23001671], [-0.29880810, -0.10000000, -17.21738815], [-0.29880810, 0.10000000, -17.21738815]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999967, 0.26460195, 0.23539841])
# # t_hit         = 0.09596199
# # hit_point-origin = np.array([0.09595200, 0.00000001, -0.00138474]
# # <hit_point-origin, ray_direction> = 0.09596200
# hit_point     = np.array([-0.13145484, 0.00000007, -17.22282410, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442101])
# triangle      = np.array([[-0.13508585, -0.10000000, -17.21690559], [-0.12258447, -0.10000000, -17.23727989], [-0.13508585, 0.10000000, -17.21690559]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.29044902, 0.50000036, 0.20955062])
# # t_hit         = 0.17069393
# # hit_point-origin = np.array([0.17067619, 0.00000000, -0.00246239]
# # <hit_point-origin, ray_direction> = 0.17069395
# hit_point     = np.array([-0.05545728, 0.00000007, -17.22391891, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01442069])
# triangle      = np.array([[-0.04702761, -0.10000000, -17.23120880], [-0.06965170, -0.10000000, -17.21164703], [-0.04702761, 0.10000000, -17.23120880]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.37259683, 0.50000036, 0.12740278])
# # t_hit         = 0.07600546
# # hit_point-origin = np.array([0.07599756, 0.00000000, -0.00109482]
# # <hit_point-origin, ray_direction> = 0.07600545
# hit_point     = np.array([-0.04231540, 0.00000008, -17.22410774, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01442124])
# triangle      = np.array([[-0.04104311, 0.10000000, -17.22099495], [-0.04291916, -0.10000000, -17.22558594], [-0.04291916, 0.10000000, -17.22558594]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999961, 0.17817838, 0.32182199])
# # t_hit         = 0.01314324
# # hit_point-origin = np.array([0.01314187, 0.00000000, -0.00018883]
# # <hit_point-origin, ray_direction> = 0.01314323
# hit_point     = np.array([0.00436914, 0.00000007, -17.22478104, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989593, 0.00000000, -0.01442141])
# triangle      = np.array([[0.00469777, -0.10000000, -17.22649002], [0.00326072, -0.10000000, -17.21901703], [0.00469777, 0.10000000, -17.22649002]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.22868367, 0.50000036, 0.27131599])
# # t_hit         = 0.04668940
# # hit_point-origin = np.array([0.04668454, -0.00000001, -0.00067329]
# # <hit_point-origin, ray_direction> = 0.04668939
# hit_point     = np.array([0.35466903, 0.00000007, -17.22983360, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442151])
# triangle      = np.array([[0.33381042, 0.10000000, -17.22088814], [0.35805407, -0.10000000, -17.23128510], [0.35805407, 0.10000000, -17.23128510]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999964, 0.36037490, 0.13962546])
# # t_hit         = 0.35033631
# # hit_point-origin = np.array([0.35029989, 0.00000000, -0.00505257]
# # <hit_point-origin, ray_direction> = 0.35033634
# hit_point     = np.array([0.40127936, 0.00000007, -17.23050499, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989605, 0.00000000, -0.01442035])
# triangle      = np.array([[0.39010572, -0.10000000, -17.23309898], [0.42137718, -0.10000000, -17.22584152], [0.39010572, 0.10000000, -17.23309898]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.35731131, 0.50000036, 0.14268833])
# # t_hit         = 0.04661518
# # hit_point-origin = np.array([0.04661033, 0.00000000, -0.00067139]
# # <hit_point-origin, ray_direction> = 0.04661516
# hit_point     = np.array([0.48316583, 0.00000007, -17.23168564, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01441849])
# triangle      = np.array([[0.48607975, 0.10000000, -17.22695160], [0.48274094, -0.10000000, -17.23237610], [0.48274094, 0.10000000, -17.23237610]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999961, 0.37274307, 0.12725729])
# # t_hit         = 0.08189497
# # hit_point-origin = np.array([0.08188647, 0.00000000, -0.00118065]
# # <hit_point-origin, ray_direction> = 0.08189499
# hit_point     = np.array([0.49697238, 0.00000007, -17.23188400, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01441876])
# triangle      = np.array([[0.49999896, -0.10000000, -17.23460770], [0.49239662, -0.10000000, -17.22776794], [0.49999896, 0.10000000, -17.23460770]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.39810854, 0.50000036, 0.10189110])
# # t_hit         = 0.01380801
# # hit_point-origin = np.array([0.01380655, -0.00000000, -0.00019836]
# # <hit_point-origin, ray_direction> = 0.01380798
# hit_point     = np.array([1.23335636, 0.00000007, -17.24250412, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01441930])
# triangle      = np.array([[1.23346233, 0.10000000, -17.23570442], [1.23335016, -0.10000000, -17.24290276], [1.23335016, 0.10000000, -17.24290276]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.49999964, 0.44450048, 0.05549988])
# # t_hit         = 0.73646051
# # hit_point-origin = np.array([0.73638397, -0.00000000, -0.01062012]
# # <hit_point-origin, ray_direction> = 0.73646063
# hit_point     = np.array([1.25363517, 0.00000006, -17.24279594, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01441930])
# triangle      = np.array([[1.25134277, -0.10000000, -17.24391174], [1.26462650, -0.10000000, -17.23744965], [1.25134277, 0.10000000, -17.24391174]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.17257342, 0.50000030, 0.32742625])
# # t_hit         = 0.02028094
# # hit_point-origin = np.array([0.02027881, -0.00000001, -0.00029182]
# # <hit_point-origin, ray_direction> = 0.02028091
# hit_point     = np.array([1.31562436, 0.00000007, -17.24368858, 0])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01441834])
# triangle      = np.array([[1.30626118, -0.10000000, -17.24100304], [1.33667779, -0.10000000, -17.24973106], [1.30626118, 0.10000000, -17.24100304]])
# poligons      = np.vstack((poligons, triangle))
# # barycentrics  = np.array([0.30783400, 0.50000036, 0.19216561])
# # t_hit         = 0.06199573
# # hit_point-origin = np.array([0.06198919, 0.00000001, -0.00089264]
# # <hit_point-origin, ray_direction> = 0.06199562
# hit_point     = np.array([1.37777197, 0.00000007, -17.24458504, 1])
# hit_points    = np.vstack((hit_points, hit_point))
# ray_direction = np.array([0.99989611, 0.00000000, -0.01441662])
# triangle      = np.array([[1.36830449, -0.10000000, -17.24868202], [1.39266706, -0.10000000, -17.23813820], [1.36830449, 0.10000000, -17.24868202]])
# poligons      = np.vstack((poligons, triangle))
# barycentrics  = np.array([0.38860548, 0.50000036, 0.11139417])
# t_hit         = 0.06215402
# hit_point-origin = np.array([0.06214762, 0.00000000, -0.00089645]
# <hit_point-origin, ray_direction> = 0.06215408
hit_point     = np.array([1.46597755, 0.00000008, -17.24585724, 0])
hit_points    = hit_point.copy() # np.vstack((hit_points, hit_point))
ray_direction = np.array([0.99989611, 0.00000000, -0.01441556])
triangle      = np.array([[1.46481860, -0.10000000, -17.24103546], [1.46837234, -0.10000000, -17.25581932], [1.46481860, 0.10000000, -17.24103546]])
poligons      = triangle.copy() #np.vstack((poligons, triangle))
# barycentrics  = np.array([0.32611516, 0.50000042, 0.17388445])
# t_hit         = 0.08821473
# hit_point-origin = np.array([0.08820558, 0.00000001, -0.00127220]
# <hit_point-origin, ray_direction> = 0.08821476
hit_point     = np.array([1.50687802, 0.00000007, -17.24644661, 1])
hit_points    = np.vstack((hit_points, hit_point))
ray_direction = np.array([0.99989617, 0.00000000, -0.01441543])
triangle      = np.array([[1.50587535, -0.10000000, -17.24732780],
                          [1.52977574, -0.10000000, -17.22632980],
                          [1.50587535, 0.10000000, -17.24732780]])
poligons      = np.vstack((poligons, triangle))
# barycentrics  = np.array([0.04195194, 0.50000036, 0.45804769])
# t_hit         = 0.04090472
# hit_point-origin = np.array([0.04090047, -0.00000001, -0.00058937]
# <hit_point-origin, ray_direction> = 0.04090472
hit_point     = np.array([1.50687838, 0.00000007, -17.24644661, 1])
hit_points    = np.vstack((hit_points, hit_point))
ray_direction = np.array([0.99989617, 0.00000000, -0.01441490])
triangle      = np.array([[1.50587535, -0.10000000, -17.24732780],
                          [1.52977574, -0.10000000, -17.22632980],
                          [1.50587535, 0.10000000, -17.24732780]])
poligons      = np.vstack((poligons, triangle))
# barycentrics  = np.array([0.04196545, 0.50000036, 0.45803422])
# t_hit         = 0.00000032
# hit_point-origin = np.array([0.00000036, -0.00000000, 0.00000000]
# <hit_point-origin, ray_direction> = 0.00000036
hit_point     = np.array([1.84559691, 0.00000007, -17.25132942, 0])
hit_points    = np.vstack((hit_points, hit_point))
ray_direction = np.array([0.99989617, 0.00000000, -0.01441490])
triangle      = np.array([[1.85833383, 0.10000000, -17.24804306], [1.83701634, -0.10000000, -17.25354385], [1.83701634, 0.10000000, -17.25354385]])
poligons      = np.vstack((poligons, triangle))
# barycentrics  = np.array([0.49999967, 0.09748660, 0.40251374])
# t_hit         = 0.33875370
# hit_point-origin = np.array([0.33871853, -0.00000000, -0.00488281]
# <hit_point-origin, ray_direction> = 0.33875376
hit_point     = np.array([1.86215854, 0.00000007, -17.25156784, 1])
hit_points    = np.vstack((hit_points, hit_point))
ray_direction = np.array([0.99989617, 0.00000000, -0.01441490])
triangle      = np.array([[1.87452281, 0.10000000, -17.26296425], [1.85833383, -0.10000000, -17.24804306], [1.85833383, 0.10000000, -17.24804306]])
poligons      = np.vstack((poligons, triangle))
# barycentrics  = np.array([0.49999967, 0.26375109, 0.23624927])
# t_hit         = 0.01656327
# hit_point-origin = np.array([0.01656163, 0.00000000, -0.00023842]
# <hit_point-origin, ray_direction> = 0.01656334
hit_point     = np.array([2.73698974, 0.00000006, -17.26417923, 0])
hit_points    = np.vstack((hit_points, hit_point))
ray_direction = np.array([0.99989617, 0.00000000, -0.01441490])
triangle      = np.array([[3.41408062, -0.10000000, -17.16374207], [1.82232547, -0.10000000, -17.39985847], [3.41408062, 0.10000000, -17.16374207]])
poligons      = np.vstack((poligons, triangle))
# barycentrics  = np.array([0.42537364, 0.50000030, 0.07462603])
# t_hit         = 0.87492222
# hit_point-origin = np.array([0.87483120, -0.00000001, -0.01261139]
# <hit_point-origin, ray_direction> = 0.87492216

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')

ax.plot3D(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], linestyle=':', marker='x')

face_colors = ['b', 'g']

for p in range(int(poligons.shape[0]/3)):
# for p in [int(poligons.shape[0]/3)-2, int(poligons.shape[0]/3)-1]:
    triang = np.vstack((poligons[p*3 : (p+1)*3], poligons[p*3]))
    ax.plot3D(triang[:, 0], triang[:, 1], triang[:, 2], color=face_colors[int(hit_points[p, 3])])

# %%

C = np.array([2, -5, 1])
R = 2

ray_origin = C + np.array([0.5, 0.3, 0])

angs = np.linspace(0, np.pi, 20, endpoint=False)
ray_directions = np.array([[.1,.1,1], [1,1,1.5], [1,1,1], [1,1, 0.5], [1, 1, 0]])

def sqSolve(a, b, c ):
    q = 1
    if b > 0:
        q = -0.5 * (b + np.sqrt(b**2 - 4 * a*c))
    else:
        q = -0.5 * (b - np.sqrt(b**2 - 4 * a*c))
    t1, t2 = q/a, c/q
    
    if t1 > t2:
        return t1
    else:
        return t2

rot = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')

ax.plot3D(C[0], C[1], C[2], marker='*')

ax.plot3D(ray_origin[0], ray_origin[1], ray_origin[2], marker='x')

for ray_direction in ray_directions:
    for ang in angs:
        D = rot(ray_direction, ang)
        L = ray_origin - C
        a = np.dot(D, D)
        b = 2*np.dot(D, L)
        c = np.dot(L, L) - R**2
        t = sqSolve(a, b, c)
        P = ray_origin + D * t
        ax.plot3D(P[0], P[1], P[2], marker='o', color='b')
        ax.plot3D(np.array([ray_origin[0], P[0]]), [ray_origin[1], P[1]], [ray_origin[2], P[2]], linestyle='-', color='g')
        # ax.quiver(ray_origin[0], ray_origin[1], ray_origin[2],
        #           D[0], D[1], D[2], length=0.2, normalize=True)
    
# %% Cylindric detector test - rays casting and detection - cone rotation geometry

rot = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

det_size=6*8    # i.e., number or pixels per projection
ray_origin = np.array([[0, -10, 0], rot([0, -10, 0], np.pi/3)])

ray_end = np.array([

])

at_detector = np.array([

])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')

colors = ['g', 'r']
for i, d in enumerate(at_detector):
    ax.plot3D([ray_origin[int(i/det_size), 0], d[0]],
              [ray_origin[int(i/det_size), 1], d[1]],
              [ray_origin[int(i/det_size), 2], d[2]],  marker='o', color=colors[int(i/det_size)])
    ax.plot3D(at_detector [i, 0], at_detector [i, 1], at_detector[i, 2],  marker='x')

ax.set_xlim(np.min(ray_end), np.max(ray_end))
ax.set_ylim(np.min(ray_end), np.max(ray_end))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# %% Rotation around arbitrary axis
rot_x = lambda x, theta: np.array([x[0], x[1]*np.cos(theta)-x[2]*np.sin(theta), x[1]*np.sin(theta)+x[2]*np.cos(theta)])
rot_y = lambda x, theta: np.array([x[0]*np.cos(theta)-x[2]*np.sin(theta), x[1], x[0]*np.sin(theta)+x[2]*np.cos(theta)])
rot_z = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

rot_op = [rot_y, rot_z, rot_x]
orig = np.array([0, 0, 0])

orth_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

angs = np.linspace(0, 3*np.pi/2, 20)

# fig = plt.figure(figsize=(8,8))
# ax = plt.axes(projection='3d')
# ax.set_proj_type('ortho')

# d_col = ['r', 'g', 'b']
# for i, d in enumerate(orth_basis):
#     for ang in angs:
#         d_rotated = rot_op[i](d, ang)
#         ax.plot3D([orig[0], d_rotated[0]],
#                   [orig[1], d_rotated[1]],
#                   [orig[2], d_rotated[2]],  marker='o', color=d_col[i])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

def rotate_axis(v, alpha, ax):
    r = np.sqrt(np.sum(ax**2))
    theta = np.arccos( ax[2] / r )
    phi = np.arctan2(ax[1], ax[0])
    # print(f"phi={phi}, theta={theta}")
    v = rot_z(v, -phi)
    v = rot_y(v, -(np.pi/2 - theta))
    
    v = rot_x(v, alpha)
    
    v = rot_y(v, (np.pi/2 - theta))
    v = rot_z(v, phi)
    
    return v

rot_axis = np.array([0, 0, -1])
point = np.array([0.1, 0.75, 0.5])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')

ax.plot3D([orig[0], rot_axis[0]],
          [orig[1], rot_axis[1]],
          [orig[2], rot_axis[2]], linestyle=':', marker='^', color='c')

for i, ang in enumerate(angs):
    p_rotated = rotate_axis(point, ang, rot_axis)
    color='g'
    if i == 0:
        color='y'
    ax.plot3D([orig[0], p_rotated[0]],
              [orig[1], p_rotated[1]],
              [orig[2], p_rotated[2]],  marker='o', color=color)
    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# %% Cylindric detector test - rays casting and detection - cone vector geometry

rot = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

det_size=8*16    # i.e., number or pixels per projection
ray_origin = np.array([-10, 0, 0])

ray_end = np.array([
[-2.77391243, 10.78016567, -4.27391243],
[-2.77391243, 10.78016567, -3.27391267],
[-2.77391243, 10.78016567, -2.27391267],
[-2.77391243, 10.78016567, -1.27391267],
[-2.35141158, 12.22161674, -3.85141182],
[-2.35141158, 12.22161674, -2.85141182],
[-2.35141158, 12.22161674, -1.85141182],
[-2.35141158, 12.22161674, -0.85141182],
[-1.57108486, 13.32478523, -3.07108498],
[-1.57108486, 13.32478523, -2.07108498],
[-1.57108486, 13.32478523, -1.07108498],
[-1.57108486, 13.32478523, -0.07108498],
[-0.55167848, 13.92179394, -2.05167866],
[-0.55167848, 13.92179394, -1.05167878],
[-0.55167848, 13.92179394, -0.05167878],
[-0.55167848, 13.92179394, 0.94832134],
[0.55167913, 13.92179394, -0.94832134],
[0.55167913, 13.92179394, 0.05167866],
[0.55167913, 13.92179394, 1.05167866],
[0.55167913, 13.92179394, 2.05167866],
[1.57108557, 13.32478428, 0.07108521],
[1.57108557, 13.32478428, 1.07108521],
[1.57108557, 13.32478428, 2.07108498],
[1.57108557, 13.32478428, 3.07108498],
[2.35141182, 12.22161674, 0.85141110],
[2.35141182, 12.22161674, 1.85141110],
[2.35141182, 12.22161674, 2.85141087],
[2.35141182, 12.22161674, 3.85141087],
[2.77391291, 10.78016472, 1.27391219],
[2.77391291, 10.78016472, 2.27391243],
[2.77391291, 10.78016472, 3.27391243],
[2.77391291, 10.78016472, 4.27391243],
[10.78036118, -4.98380136, -1.06066012],
[10.78036118, -4.27669477, -0.35355341],
[10.78036118, -3.56958771, 0.35355341],
[10.78036118, -2.86248112, 1.06066024],
[12.22228050, -4.38653851, -1.06066012],
[12.22228050, -3.67943144, -0.35355341],
[12.22228050, -2.97232485, 0.35355341],
[12.22228050, -2.26521802, 1.06066024],
[13.32587814, -3.28294086, -1.06066012],
[13.32587814, -2.57583427, -0.35355341],
[13.32587814, -1.86872733, 0.35355341],
[13.32587814, -1.16162062, 1.06066024],
[13.92314148, -1.84102130, -1.06066012],
[13.92314148, -1.13391459, -0.35355341],
[13.92314148, -0.42680776, 0.35355341],
[13.92314148, 0.28029895, 1.06066024],
[13.92314148, -0.28029895, -1.06066012],
[13.92314148, 0.42680776, -0.35355341],
[13.92314148, 1.13391459, 0.35355341],
[13.92314148, 1.84102142, 1.06066024],
[13.32587814, 1.16162026, -1.06066012],
[13.32587814, 1.86872709, -0.35355341],
[13.32587814, 2.57583380, 0.35355341],
[13.32587814, 3.28294039, 1.06066024],
[12.22228050, 2.26521778, -1.06066012],
[12.22228050, 2.97232437, -0.35355341],
[12.22228050, 3.67943096, 0.35355341],
[12.22228050, 4.38653803, 1.06066024],
[10.78036118, 2.86248064, -1.06066012],
[10.78036118, 3.56958771, -0.35355341],
[10.78036118, 4.27669430, 0.35355341],
[10.78036118, 4.98380089, 1.06066024],


])

at_detector = np.array([
[-2.77391267, 10.78016663, -4.27391291],
[-2.77391267, 10.78016663, -3.27391291],
[-2.35141158, 12.22161865, -3.85141206],
[2.35141253, 12.22161674, 3.85141182],
[2.77391315, 10.78016376, 3.27391243],
[2.77391315, 10.78016376, 4.27391243],
[13.32587910, -1.86872721, 0.35355353],
[13.32587814, -1.16162062, 1.06066048],
[13.92314148, -1.84102154, -1.06066012],
[13.92314148, -1.13391447, -0.35355341],
[13.92314148, -0.42680740, 0.35355353],
[13.92314148, 0.28029919, 1.06066024],
[13.92314148, -0.28029871, -1.06066012],
[13.92314148, 0.42680788, -0.35355341],
[13.92314148, 1.13391495, 0.35355353],
[13.92314148, 1.84102201, 1.06066024],
[13.32587910, 1.16162014, -1.06066012],
[13.32587814, 1.86872721, -0.35355341],
[13.32587910, 2.57583380, 0.35355353],
[13.32587910, 3.28294039, 1.06066048],
[12.22228146, 2.26521826, -1.06066024],
[12.22228241, 2.97232389, -0.35355341],
[10.78036118, 2.86248064, -1.06066012],
[10.78036118, -4.98380136, -1.06066012],
[10.78035927, -4.27669477, -0.35355318],
[10.78036118, -3.56958771, 0.35355353],
[10.78036022, -2.86248064, 1.06066072],
[12.22228146, -4.38653851, -1.06066012],
[12.22228146, -3.67943144, -0.35355341],
[12.22228146, -2.97232485, 0.35355353],
[12.22228146, -2.26521826, 1.06066024],
[13.32587910, -3.28294086, -1.06066012],
[13.32587814, -2.57583427, -0.35355318],
[12.22228241, 3.67943144, 0.35355353],
[12.22228241, 4.38653803, 1.06066072],
[10.78036022, 3.56958818, -0.35355341],
[10.78036118, 4.27669477, 0.35355353],
[10.78036022, 4.98380136, 1.06066024],
[-2.77391267, 10.78016663, -2.27391291],
[-2.77391267, 10.78016663, -1.27391267],
[-2.35141158, 12.22161865, -2.85141206],
[-2.35141182, 12.22161865, -1.85141182],
[-2.35141182, 12.22161865, -0.85141182],
[-1.57108498, 13.32478619, -3.07108521],
[-1.57108498, 13.32478619, -2.07108498],
[-1.57108498, 13.32478619, -1.07108474],
[-1.57108498, 13.32478619, -0.07108498],
[-0.55167818, 13.92179394, -2.05167818],
[-0.55167866, 13.92179394, -1.05167842],
[0.55167890, 13.92179394, 1.05167866],
[0.55167890, 13.92179394, 2.05167818],
[1.57108569, 13.32478523, 0.07108545],
[1.57108569, 13.32478523, 1.07108545],
[1.57108569, 13.32478523, 2.07108498],
[1.57108569, 13.32478523, 3.07108545],
[2.35141253, 12.22161674, 0.85141182],
[2.35141253, 12.22161674, 1.85141182],
[2.35141253, 12.22161674, 2.85141134],
[2.77391315, 10.78016376, 1.27391243],
[2.77391315, 10.78016376, 2.27391243],
[-0.55167866, 13.92179394, -0.05167818],
[-0.55167818, 13.92179394, 0.94832182],
[0.55167890, 13.92179394, -0.94832158],
[0.55167890, 13.92179394, 0.05167866],


])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')

colors = ['g', 'r', 'c']
for i, e in enumerate(ray_end):
    ax.plot3D([e[0]],
              [ e[1]],
              [ e[2]],  marker='o', color='g')

for i, d in enumerate(at_detector):
    ax.plot3D(d [0], d[ 1], d[ 2],  marker='x', markersize=8)


ax.set_xlim(np.min(ray_end), np.max(ray_end))
ax.set_ylim(np.min(ray_end), np.max(ray_end))
ax.set_zlim(np.min(ray_end), np.max(ray_end))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# %%
rot = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

det_size=2*16   # i.e., number or pixels per projection
ray_origin = np.array([[0, -10, 0], [-10, 0, 0], [0, 0, -10]])

ray_end = np.array([
[-3.92314124, -1.50000000, 10.78036118],
[-3.92314124, -0.50000000, 10.78036118],
[-3.92314124, 0.50000000, 10.78036118],
[-3.92314124, 1.50000000, 10.78036118],
[-3.32587814, -1.50000000, 12.22228050],
[-3.32587814, -0.50000000, 12.22228050],
[-3.32587814, 0.50000000, 12.22228050],
[-3.32587814, 1.50000000, 12.22228050],
[-2.22228074, -1.50000000, 13.32587814],
[-2.22228074, -0.50000000, 13.32587814],
[-2.22228074, 0.50000000, 13.32587814],
[-2.22228074, 1.50000000, 13.32587814],
[-0.78036118, -1.50000000, 13.92314148],
[-0.78036118, -0.50000000, 13.92314148],
[-0.78036118, 0.50000000, 13.92314148],
[-0.78036118, 1.50000000, 13.92314148],
[0.78036118, -1.50000000, 13.92314148],
[0.78036118, -0.50000000, 13.92314148],
[0.78036118, 0.50000000, 13.92314148],
[0.78036118, 1.50000000, 13.92314148],
[2.22228050, -1.50000000, 13.32587814],
[2.22228050, -0.50000000, 13.32587814],
[2.22228050, 0.50000000, 13.32587814],
[2.22228050, 1.50000000, 13.32587814],
[3.32587790, -1.50000000, 12.22228050],
[3.32587790, -0.50000000, 12.22228050],
[3.32587790, 0.50000000, 12.22228050],
[3.32587790, 1.50000000, 12.22228050],
[3.92314100, -1.50000000, 10.78036118],
[3.92314100, -0.50000000, 10.78036118],
[3.92314100, 0.50000000, 10.78036118],
[3.92314100, 1.50000000, 10.78036118],
[-3.92314124, 10.78036118, -1.50000000],
[-3.92314124, 10.78036118, -0.50000000],
[-3.92314124, 10.78036118, 0.50000000],
[-3.92314124, 10.78036118, 1.50000000],
[-3.32587814, 12.22228050, -1.50000000],
[-3.32587814, 12.22228050, -0.50000000],
[-3.32587814, 12.22228050, 0.50000000],
[-3.32587814, 12.22228050, 1.50000000],
[-2.22228074, 13.32587814, -1.50000000],
[-2.22228074, 13.32587814, -0.50000000],
[-2.22228074, 13.32587814, 0.50000000],
[-2.22228074, 13.32587814, 1.50000000],
[-0.78036118, 13.92314148, -1.50000000],
[-0.78036118, 13.92314148, -0.50000000],
[-0.78036118, 13.92314148, 0.50000000],
[-0.78036118, 13.92314148, 1.50000000],
[0.78036118, 13.92314148, -1.50000000],
[0.78036118, 13.92314148, -0.50000000],
[0.78036118, 13.92314148, 0.50000000],
[0.78036118, 13.92314148, 1.50000000],
[2.22228050, 13.32587814, -1.50000000],
[2.22228050, 13.32587814, -0.50000000],
[2.22228050, 13.32587814, 0.50000000],
[2.22228050, 13.32587814, 1.50000000],
[3.32587790, 12.22228050, -1.50000000],
[3.32587790, 12.22228050, -0.50000000],
[3.32587790, 12.22228050, 0.50000000],
[3.32587790, 12.22228050, 1.50000000],
[3.92314100, 10.78036118, -1.50000000],
[3.92314100, 10.78036118, -0.50000000],
[3.92314100, 10.78036118, 0.50000000],
[3.92314100, 10.78036118, 1.50000000],
[10.78036118, -3.92314124, -1.50000000],
[10.78036118, -3.92314124, -0.50000000],
[10.78036118, -3.92314124, 0.50000000],
[10.78036118, -3.92314124, 1.50000000],
[12.22228050, -3.32587814, -1.50000000],
[12.22228050, -3.32587814, -0.50000000],
[12.22228050, -3.32587814, 0.50000000],
[12.22228050, -3.32587814, 1.50000000],
[13.32587814, -2.22228074, -1.50000000],
[13.32587814, -2.22228074, -0.50000000],
[13.32587814, -2.22228074, 0.50000000],
[13.32587814, -2.22228074, 1.50000000],
[13.92314148, -0.78036118, -1.50000000],
[13.92314148, -0.78036118, -0.50000000],
[13.92314148, -0.78036118, 0.50000000],
[13.92314148, -0.78036118, 1.50000000],
[13.92314148, 0.78036118, -1.50000000],
[13.92314148, 0.78036118, -0.50000000],
[13.92314148, 0.78036118, 0.50000000],
[13.92314148, 0.78036118, 1.50000000],
[13.32587814, 2.22228050, -1.50000000],
[13.32587814, 2.22228050, -0.50000000],
[13.32587814, 2.22228050, 0.50000000],
[13.32587814, 2.22228050, 1.50000000],
[12.22228050, 3.32587790, -1.50000000],
[12.22228050, 3.32587790, -0.50000000],
[12.22228050, 3.32587790, 0.50000000],
[12.22228050, 3.32587790, 1.50000000],
[10.78036118, 3.92314100, -1.50000000],
[10.78036118, 3.92314100, -0.50000000],
[10.78036118, 3.92314100, 0.50000000],
[10.78036118, 3.92314100, 1.50000000],


])

at_detector = np.array([
[-3.92314124, -1.50000000, 10.78036022],
[-3.92314124, -0.50000000, 10.78036022],
[-3.92314124, 0.50000000, 10.78036022],
[-3.92314124, 1.50000000, 10.78036022],
[-3.32587838, -1.50000000, 12.22228146],
[-3.32587838, -0.50000000, 12.22228146],
[-3.32587838, 0.50000000, 12.22228146],
[-3.32587838, 1.50000024, 12.22228146],
[-2.22228098, -1.50000012, 13.32587814],
[-2.22228098, -0.50000000, 13.32587814],
[-2.22228098, 0.50000024, 13.32587814],
[-2.22228098, 1.50000000, 13.32587814],
[-0.78036118, -1.50000000, 13.92314148],
[-0.78036118, -0.50000000, 13.92314148],
[-0.78036118, 0.50000000, 13.92314148],
[-0.78036118, 1.50000000, 13.92314148],
[0.78036165, -1.50000000, 13.92314148],
[0.78036165, -0.50000000, 13.92314148],
[0.78036165, 0.50000000, 13.92314148],
[0.78036165, 1.50000000, 13.92314148],
[2.22228050, -1.50000012, 13.32587814],
[2.22228050, -0.50000000, 13.32587814],
[2.22228050, 0.50000024, 13.32587814],
[2.22228050, 1.50000000, 13.32587814],
[3.32587814, -1.50000000, 12.22228146],
[3.32587814, -0.50000000, 12.22228146],
[3.32587814, 0.50000000, 12.22228146],
[3.32587814, 1.50000024, 12.22228146],
[3.92314100, -1.50000012, 10.78036118],
[3.92314100, -0.50000000, 10.78036118],
[3.92314100, 0.50000000, 10.78036118],
[3.92314100, 1.50000000, 10.78036118],
[-3.92314124, 10.78036022, -1.50000000],
[-3.92314124, 10.78036022, -0.50000000],
[-3.92314124, 10.78036022, 0.50000000],
[-3.92314124, 10.78036022, 1.50000000],
[-3.32587838, 12.22228146, -1.50000000],
[-3.32587838, 12.22228146, -0.50000000],
[-3.32587838, 12.22228146, 0.50000000],
[-3.32587838, 12.22228146, 1.50000024],
[3.32587814, 12.22228146, -1.50000000],
[3.32587814, 12.22228146, -0.50000000],
[3.32587814, 12.22228146, 0.50000000],
[3.32587814, 12.22228146, 1.50000024],
[3.92314100, 10.78036118, -1.50000012],
[3.92314100, 10.78036118, -0.50000000],
[3.92314100, 10.78036118, 0.50000000],
[3.92314100, 10.78036118, 1.50000000],
[13.92314148, -0.78036118, -1.50000000],
[13.92314148, -0.78036118, -0.50000000],
[13.92314148, -0.78036118, 0.50000000],
[13.92314148, -0.78036118, 1.50000000],
[13.92314148, 0.78036165, -1.50000000],
[13.92314148, 0.78036165, -0.50000000],
[13.92314148, 0.78036165, 0.50000000],
[13.92314148, 0.78036165, 1.50000000],
[13.32587814, 2.22228050, -1.50000012],
[13.32587814, 2.22228050, -0.50000000],
[13.32587814, 2.22228050, 0.50000024],
[13.32587814, 2.22228050, 1.50000000],
[12.22228146, 3.32587814, -1.50000000],
[12.22228146, 3.32587814, 1.50000024],
[10.78036118, 3.92314100, -1.50000012],
[10.78036118, 3.92314100, 1.50000000],
[10.78036118, -3.92314124, -1.50000012],
[10.78036022, -3.92314124, -0.50000000],
[10.78036022, -3.92314124, 0.50000000],
[10.78036022, -3.92314124, 1.50000000],
[12.22228146, -3.32587814, -1.50000000],
[12.22228146, -3.32587814, -0.50000000],
[12.22228146, -3.32587838, 0.50000000],
[12.22228146, -3.32587838, 1.50000024],
[13.32587814, -2.22228074, -1.50000000],
[13.32587814, -2.22228074, -0.50000000],
[13.32587814, -2.22228074, 0.50000024],
[13.32587814, -2.22228074, 1.50000000],
[12.22228146, 3.32587814, -0.50000000],
[12.22228146, 3.32587814, 0.50000000],
[10.78036022, 3.92314148, -0.50000000],
[10.78036022, 3.92314148, 0.50000000],
[-2.22228074, 13.32587814, -1.50000000],
[-2.22228074, 13.32587814, -0.50000012],
[-2.22228074, 13.32587814, 0.50000000],
[-2.22228074, 13.32587814, 1.50000000],
[-0.78036118, 13.92314148, -1.50000000],
[-0.78036118, 13.92314148, 1.50000000],
[0.78036165, 13.92314148, -1.50000000],
[0.78036165, 13.92314148, 1.50000024],
[2.22228050, 13.32587814, -1.50000000],
[2.22228050, 13.32587814, -0.50000000],
[2.22228050, 13.32587814, 0.50000024],
[2.22228050, 13.32587814, 1.50000000],
[-0.78036141, 13.92314148, -0.50000000],
[-0.78036118, 13.92314148, 0.50000024],
[0.78036165, 13.92314148, -0.49999988],
[0.78036165, 13.92314148, 0.50000024],

])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')

colors = ['g', 'r', 'c']
for i, e in enumerate(ray_end):
    ax.plot3D(e[0],
              e[1],
              e[2],  marker='o', color='g')

for i, d in enumerate(at_detector):
    ax.plot3D(d [0], d[ 1], d[ 2],  marker='x', markersize=8)


ax.set_xlim(np.min(ray_end), np.max(ray_end))
ax.set_ylim(np.min(ray_end), np.max(ray_end))
ax.set_zlim(np.min(ray_end), np.max(ray_end))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# %%
rot = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

det_size=2*16   # i.e., number or pixels per projection
ray_origin = np.array([[0, -10, 0], [-10, 0, 0], [0, 0, -10]])

ray_end = np.array([
[-7.50971603, 10.12247086, -0.50000000],
[-7.50971603, 10.12247086, 0.50000000],
[-6.52363873, 10.34411049, -0.50000000],
[-6.52363873, 10.34411049, 0.50000000],
[-5.53104544, 10.53450012, -0.50000000],
[-5.53104544, 10.53450012, 0.50000000],
[-4.53295279, 10.69345093, -0.50000000],
[-4.53295279, 10.69345093, 0.50000000],
[-3.53032899, 10.82079697, -0.50000000],
[-3.53032899, 10.82079697, 0.50000000],
[-2.52418351, 10.91641998, -0.50000000],
[-2.52418351, 10.91641998, 0.50000000],
[-1.51551926, 10.98022461, -0.50000000],
[-1.51551926, 10.98022461, 0.50000000],
[-0.50534374, 11.01214218, -0.50000000],
[-0.50534374, 11.01214218, 0.50000000],
[0.50533390, 11.01214409, -0.50000000],
[0.50533390, 11.01214409, 0.50000000],
[1.51551068, 10.98022461, -0.50000000],
[1.51551068, 10.98022461, 0.50000000],
[2.52416873, 10.91642380, -0.50000000],
[2.52416873, 10.91642380, 0.50000000],
[3.53031683, 10.82079887, -0.50000000],
[3.53031683, 10.82079887, 0.50000000],
[4.53294468, 10.69344711, -0.50000000],
[4.53294468, 10.69344711, 0.50000000],
[5.53104591, 10.53450012, -0.50000000],
[5.53104591, 10.53450012, 0.50000000],
[6.52362204, 10.34411049, -0.50000000],
[6.52362204, 10.34411049, 0.50000000],
[7.50970221, 10.12246895, -0.50000000],
[7.50970221, 10.12246895, 0.50000000],

[10.12247086, -7.50971603, -0.50000000],
[10.12247086, -7.50971603, 0.50000000],
[10.34411049, -6.52363873, -0.50000000],
[10.34411049, -6.52363873, 0.50000000],
[10.53450012, -5.53104544, -0.50000000],
[10.53450012, -5.53104544, 0.50000000],
[10.69345093, -4.53295231, -0.50000000],
[10.69345093, -4.53295231, 0.50000000],
[10.82079697, -3.53032875, -0.50000000],
[10.82079697, -3.53032875, 0.50000000],
[10.91641998, -2.52418327, -0.50000000],
[10.91641998, -2.52418327, 0.50000000],
[10.98022461, -1.51551914, -0.50000000],
[10.98022461, -1.51551914, 0.50000000],
[11.01214218, -0.50534391, -0.50000000],
[11.01214218, -0.50534391, 0.50000000],
[11.01214409, 0.50533438, -0.50000000],
[11.01214409, 0.50533438, 0.50000000],
[10.98022461, 1.51551104, -0.50000000],
[10.98022461, 1.51551104, 0.50000000],
[10.91642380, 2.52416849, -0.50000000],
[10.91642380, 2.52416849, 0.50000000],
[10.82079887, 3.53031731, -0.50000000],
[10.82079887, 3.53031731, 0.50000000],
[10.69344711, 4.53294420, -0.50000000],
[10.69344711, 4.53294420, 0.50000000],
[10.53450012, 5.53104544, -0.50000000],
[10.53450012, 5.53104544, 0.50000000],
[10.34411049, 6.52362204, -0.50000000],
[10.34411049, 6.52362204, 0.50000000],
[10.12246895, 7.50970173, -0.50000000],
[10.12246895, 7.50970173, 0.50000000],
    
[-7.50971603, -0.50000000, 10.12247086],
[-7.50971603, 0.50000000, 10.12247086],
[-6.52363873, -0.50000000, 10.34411049],
[-6.52363873, 0.50000000, 10.34411049],
[-5.53104544, -0.50000000, 10.53450012],
[-5.53104544, 0.50000000, 10.53450012],
[-4.53295231, -0.50000000, 10.69345093],
[-4.53295231, 0.50000000, 10.69345093],
[-3.53032875, -0.50000000, 10.82079697],
[-3.53032875, 0.50000000, 10.82079697],
[-2.52418327, -0.50000000, 10.91641998],
[-2.52418327, 0.50000000, 10.91641998],
[-1.51551914, -0.50000000, 10.98022461],
[-1.51551914, 0.50000000, 10.98022461],
[-0.50534391, -0.50000000, 11.01214218],
[-0.50534391, 0.50000000, 11.01214218],
[0.50533438, -0.50000000, 11.01214409],
[0.50533438, 0.50000000, 11.01214409],
[1.51551104, -0.50000000, 10.98022461],
[1.51551104, 0.50000000, 10.98022461],
[2.52416849, -0.50000000, 10.91642380],
[2.52416849, 0.50000000, 10.91642380],
[3.53031731, -0.50000000, 10.82079887],
[3.53031731, 0.50000000, 10.82079887],
[4.53294420, -0.50000000, 10.69344711],
[4.53294420, 0.50000000, 10.69344711],
[5.53104544, -0.50000000, 10.53450012],
[5.53104544, 0.50000000, 10.53450012],
[6.52362204, -0.50000000, 10.34411049],
[6.52362204, 0.50000000, 10.34411049],
[7.50970173, -0.50000000, 10.12246895],
[7.50970173, 0.50000000, 10.12246895],


])

at_detector = np.array([
    
[-7.50833893, 10.11878109, -0.49990827],
[-7.49734020, 10.08930969, 0.49917603],
[-6.52243853, 10.34036636, -0.49990797],
[-6.51284981, 10.31046486, 0.49917316],
[-5.53002548, 10.53071213, -0.49990779],
[-5.52187061, 10.50043678, 0.49917066],
[-4.53211403, 10.68962097, -0.49990743],
[-4.52541351, 10.65903282, 0.49916840],
[-3.52967453, 10.81693935, -0.49990731],
[-3.52444553, 10.78609848, 0.49916673],
[3.52966309, 10.81694031, -0.49990737],
[3.52443409, 10.78609943, 0.49916673],
[4.53210735, 10.68962193, -0.49990755],
[4.52540684, 10.65903378, 0.49916852],
[5.53002548, 10.53071213, -0.49990779],
[5.52187061, 10.50043678, 0.49917066],
[6.52242279, 10.34037018, -0.49990809],
[6.51283455, 10.31046867, 0.49917328],
[7.50832748, 10.11878395, -0.49990839],
[7.49732780, 10.08931255, 0.49917603],
[-2.52371502, 10.91253853, -0.49990726],
[-2.51997042, 10.88150883, 0.49916553],
[-1.51523781, 10.97632504, -0.49990702],
[-1.51298714, 10.94516850, 0.49916446],
[1.51522923, 10.97632599, -0.49990702],
[1.51297855, 10.94516945, 0.49916446],
[2.52370071, 10.91253948, -0.49990720],
[2.51995564, 10.88150978, 0.49916542],
[-0.50524950, 11.00823498, -0.49990696],
[-0.50449848, 10.97701359, 0.49916422],
[0.50524044, 11.00823593, -0.49990720],
[0.50448895, 10.97701454, 0.49916410],

[10.11878109, -7.50833893, -0.49990827],
[10.08930969, -7.49734020, 0.49917603],
[10.97632504, -1.51523733, -0.49990702],
[10.94516850, -1.51298666, 0.49916446],
[11.00823498, -0.50524998, -0.49990702],
[10.97701359, -0.50449896, 0.49916410],
[11.00823593, 0.50524044, -0.49990702],
[10.97701454, 0.50448895, 0.49916399],
[10.97632599, 1.51522923, -0.49990702],
[10.94516945, 1.51297855, 0.49916446],
[10.91253948, 2.52370071, -0.49990714],
[10.88150978, 2.51995564, 0.49916530],
[10.53071213, 5.53002453, -0.49990779],
[10.50043678, 5.52186966, 0.49917066],
[10.34037018, 6.52242279, -0.49990809],
[10.31046867, 6.51283455, 0.49917328],
[10.11878395, 7.50832653, -0.49990839],
[10.08931255, 7.49732685, 0.49917603],
[10.34036636, -6.52243853, -0.49990797],
[10.31046486, -6.51284981, 0.49917316],
[10.53071213, -5.53002548, -0.49990779],
[10.50043678, -5.52187061, 0.49917066],
[10.68962097, -4.53211308, -0.49990749],
[10.65903282, -4.52541256, 0.49916840],
[10.81693935, -3.52967453, -0.49990731],
[10.78609848, -3.52444553, 0.49916685],
[10.91253853, -2.52371502, -0.49990720],
[10.88150883, -2.51996994, 0.49916553],
[10.81694031, 3.52966309, -0.49990737],
[10.78609943, 3.52443409, 0.49916673],
[10.68962193, 4.53210545, -0.49990755],
[10.65903378, 4.52540493, 0.49916863],    

[-7.50833893, -0.49990827, 10.11878109],
[-7.49734020, 0.49917603, 10.08930969],
[-6.52243853, -0.49990797, 10.34036636],
[-6.51284981, 0.49917316, 10.31046486],
[-5.53002548, -0.49990779, 10.53071213],
[-5.52187061, 0.49917066, 10.50043678],
[-4.53211308, -0.49990743, 10.68962097],
[-4.52541256, 0.49916840, 10.65903282],
[-3.52967453, -0.49990731, 10.81693935],
[-3.52444553, 0.49916673, 10.78609848],
[-2.52371502, -0.49990720, 10.91253853],
[-2.51997042, 0.49916542, 10.88150883],
[-1.51523733, -0.49990702, 10.97632504],
[-1.51298666, 0.49916446, 10.94516850],
[-0.50524998, -0.49990702, 11.00823498],
[-0.50449896, 0.49916410, 10.97701359],
[0.50524044, -0.49990702, 11.00823593],
[0.50448895, 0.49916399, 10.97701454],
[1.51522923, -0.49990702, 10.97632599],
[1.51297855, 0.49916446, 10.94516945],
[2.52370071, -0.49990714, 10.91253948],
[2.51995564, 0.49916530, 10.88150978],
[3.52966309, -0.49990737, 10.81694031],
[3.52443409, 0.49916673, 10.78609943],
[4.53210640, -0.49990755, 10.68962193],
[4.52540588, 0.49916852, 10.65903378],
[5.53002453, -0.49990779, 10.53071213],
[5.52186966, 0.49917066, 10.50043678],
[6.52242279, -0.49990809, 10.34037018],
[6.51283455, 0.49917328, 10.31046867],
[7.50832653, -0.49990839, 10.11878395],
[7.49732685, 0.49917603, 10.08931255],
# next projection angle

])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')

colors = ['g', 'r', 'c']
for i, d in enumerate(ray_end):
    ax.plot3D([ray_origin[int(i/det_size),0], d[0]],
              [ray_origin[int(i/det_size),1], d[1]],
              [ray_origin[int(i/det_size),2], d[2]],  marker='o', color=colors[int(i/det_size)])
    ax.plot3D(at_detector [i, 0], at_detector [i, 1], at_detector[i, 2],  marker='x', markersize=10)

ax.set_xlim(np.min(ray_end), np.max(ray_end))
ax.set_ylim(np.min(ray_end), np.max(ray_end))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# %% Detector test - cone vec geometry
rot = lambda x, theta: np.array([x[0]*np.cos(theta)-x[1]*np.sin(theta), x[0]*np.sin(theta)+x[1]*np.cos(theta), x[2]])

det_size=2*32   # i.e., number or pixels per projection
ray_origin = np.array([0, 9, 0])

ray_end = np.array([
[1.01066887, 12.02428436, -3.50000000],
[1.01066887, 12.02428436, -2.50000000],
[1.01066887, 12.02428436, -1.50000000],
[1.01066887, 12.02428436, -0.50000000],
[3.03102255, 11.96044540, -3.50000000],
[3.03102255, 11.96044540, -2.50000000],
[3.03102255, 11.96044540, -1.50000000],
[3.03102255, 11.96044540, -0.50000000],
[5.04833841, 11.83284760, -3.50000000],
[5.04833841, 11.83284760, -2.50000000],
[5.04833841, 11.83284760, -1.50000000],
[5.04833841, 11.83284760, -0.50000000],
[7.06063509, 11.64159775, -3.50000000],
[7.06063509, 11.64159775, -2.50000000],
[7.06063509, 11.64159775, -1.50000000],
[7.06063509, 11.64159775, -0.50000000],
[9.06589031, 11.38689423, -3.50000000],
[9.06589031, 11.38689423, -2.50000000],
[9.06589031, 11.38689423, -1.50000000],
[9.06589031, 11.38689423, -0.50000000],
[11.06209278, 11.06899643, -3.50000000],
[11.06209278, 11.06899643, -2.50000000],
[11.06209278, 11.06899643, -1.50000000],
[11.06209278, 11.06899643, -0.50000000],
[13.04724598, 10.68822098, -3.50000000],
[13.04724598, 10.68822098, -2.50000000],
[13.04724598, 10.68822098, -1.50000000],
[13.04724598, 10.68822098, -0.50000000],
[15.01940632, 10.24493790, -3.50000000],
[15.01940632, 10.24493790, -2.50000000],
[15.01940632, 10.24493790, -1.50000000],
[15.01940632, 10.24493790, -0.50000000],
[1.01066887, 12.02428436, 0.50000000],
[1.01066887, 12.02428436, 1.50000000],
[1.01066887, 12.02428436, 2.50000000],
[1.01066887, 12.02428436, 3.50000000],
[3.03102255, 11.96044540, 0.50000000],
[3.03102255, 11.96044540, 1.50000000],
[3.03102255, 11.96044540, 2.50000000],
[3.03102255, 11.96044540, 3.50000000],
[5.04833841, 11.83284760, 0.50000000],
[5.04833841, 11.83284760, 1.50000000],
[5.04833841, 11.83284760, 2.50000000],
[5.04833841, 11.83284760, 3.50000000],
[7.06063509, 11.64159775, 0.50000000],
[7.06063509, 11.64159775, 1.50000000],
[7.06063509, 11.64159775, 2.50000000],
[7.06063509, 11.64159775, 3.50000000],
[9.06589031, 11.38689423, 0.50000000],
[9.06589031, 11.38689423, 1.50000000],
[9.06589031, 11.38689423, 2.50000000],
[9.06589031, 11.38689423, 3.50000000],
[11.06209278, 11.06899643, 0.50000000],
[11.06209278, 11.06899643, 1.50000000],
[11.06209278, 11.06899643, 2.50000000],
[11.06209278, 11.06899643, 3.50000000],
[13.04724598, 10.68822098, 0.50000000],
[13.04724598, 10.68822098, 1.50000000],
[13.04724598, 10.68822098, 2.50000000],
[13.04724598, 10.68822098, 3.50000000],
[15.01940632, 10.24493790, 0.50000000],
[15.01940632, 10.24493790, 1.50000000],
[15.01940632, 10.24493790, 2.50000000],
[15.01940632, 10.24493790, 3.50000000],
[-15.01943111, 10.24494171, -3.50000000],
[-15.01943111, 10.24494171, -2.50000000],
[-15.01943111, 10.24494171, -1.50000000],
[-15.01943111, 10.24494171, -0.50000000],
[-13.04727650, 10.68822098, -3.50000000],
[-13.04727650, 10.68822098, -2.50000000],
[-13.04727650, 10.68822098, -1.50000000],
[-13.04727650, 10.68822098, -0.50000000],
[-11.06209087, 11.06900024, -3.50000000],
[-11.06209087, 11.06900024, -2.50000000],
[-11.06209087, 11.06900024, -1.50000000],
[-11.06209087, 11.06900024, -0.50000000],
[-9.06590462, 11.38690186, -3.50000000],
[-9.06590462, 11.38690186, -2.50000000],
[-9.06590462, 11.38690186, -1.50000000],
[-9.06590462, 11.38690186, -0.50000000],
[-7.06065750, 11.64159012, -3.50000000],
[-7.06065750, 11.64159012, -2.50000000],
[-7.06065750, 11.64159012, -1.50000000],
[-7.06065750, 11.64159012, -0.50000000],
[-5.04836655, 11.83283997, -3.50000000],
[-5.04836655, 11.83283997, -2.50000000],
[-5.04836655, 11.83283997, -1.50000000],
[-5.04836655, 11.83283997, -0.50000000],
[-3.03103781, 11.96044922, -3.50000000],
[-3.03103781, 11.96044922, -2.50000000],
[-3.03103781, 11.96044922, -1.50000000],
[-3.03103781, 11.96044922, -0.50000000],
[-1.01068652, 12.02428436, -3.50000000],
[-1.01068652, 12.02428436, -2.50000000],
[-1.01068652, 12.02428436, -1.50000000],
[-1.01068652, 12.02428436, -0.50000000],
[-15.01943111, 10.24494171, 0.50000000],
[-15.01943111, 10.24494171, 1.50000000],
[-15.01943111, 10.24494171, 2.50000000],
[-15.01943111, 10.24494171, 3.50000000],
[-13.04727650, 10.68822098, 0.50000000],
[-13.04727650, 10.68822098, 1.50000000],
[-13.04727650, 10.68822098, 2.50000000],
[-13.04727650, 10.68822098, 3.50000000],
[-11.06209087, 11.06900024, 0.50000000],
[-11.06209087, 11.06900024, 1.50000000],
[-11.06209087, 11.06900024, 2.50000000],
[-11.06209087, 11.06900024, 3.50000000],
[-9.06590462, 11.38690186, 0.50000000],
[-9.06590462, 11.38690186, 1.50000000],
[-9.06590462, 11.38690186, 2.50000000],
[-9.06590462, 11.38690186, 3.50000000],
[-7.06065750, 11.64159012, 0.50000000],
[-7.06065750, 11.64159012, 1.50000000],
[-7.06065750, 11.64159012, 2.50000000],
[-7.06065750, 11.64159012, 3.50000000],
[-5.04836655, 11.83283997, 0.50000000],
[-5.04836655, 11.83283997, 1.50000000],
[-5.04836655, 11.83283997, 2.50000000],
[-5.04836655, 11.83283997, 3.50000000],
[-3.03103781, 11.96044922, 0.50000000],
[-3.03103781, 11.96044922, 1.50000000],
[-3.03103781, 11.96044922, 2.50000000],
[-3.03103781, 11.96044922, 3.50000000],
[-1.01068652, 12.02428436, 0.50000000],
[-1.01068652, 12.02428436, 1.50000000],
[-1.01068652, 12.02428436, 2.50000000],
[-1.01068652, 12.02428436, 3.50000000],

])

at_detector = np.array([

])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')

colors = ['g', 'r', 'c']
for i, d in enumerate(ray_end):
    ax.plot3D([ray_origin[0], d[0]],
               [ray_origin[1], d[1]],
               [ray_origin[2], d[2]],  marker='o', color=colors[0])
    # ax.plot3D(at_detector [i, 0], at_detector [i, 1], at_detector[i, 2],  marker='x', markersize=10)

ax.set_xlim(np.min(ray_end), np.max(ray_end))
ax.set_ylim(np.min(ray_end), np.max(ray_end))
ax.set_zlim(np.min(ray_end), np.max(ray_end))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# %%

det_indices = np.array([
[1.38106799, 0.50000000],
[1.95312512, 0.50000000],
[1.95312512, 0.50000000],
[3.38291192, 0.50000000],
[4.58048439, 0.50000000],

    ])

plt.plot(det_indices[:, 0], det_indices[:,1], 'o')

# %%

det_size=2*16   # i.e., number or pixels per projection
ray_origin = np.array([[-50, 0, 0]])

ray_end = np.array([
[550.00006104, 0.30000001, 0.00000000],
[550.00006104, 0.14999999, 0.00000000],
[550.00012207, -0.00000003, 0.00000000],
[550.00006104, -0.14999998, 0.00000000],
[550.00006104, -0.30000007, 0.00000000],
])

at_detector = np.array([
[550.00006104, 0.30000001, 0.00000000],
[550.00012207, 0.15000001, 0.00000000],
[550.00018311, -0.00000006, 0.00000000],
[550.00012207, -0.15000004, 0.00000000],
[550.00006104, -0.30000019, 0.00000000],
])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')

colors = ['g', 'r', 'c']
for i, e in enumerate(ray_end):
    ax.plot3D(e[0],
              e[1],
              e[2],  marker='o', color='g')

for i, d in enumerate(at_detector):
    ax.plot3D(d [0], d[ 1], d[ 2],  marker='x', markersize=8)

ax.set_xlim(550, 550.0005)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# %%
hit_point     = np.array([-1.81119192, 2.54060435, 2.50292969]) # is front face: true
hit_points    = hit_point.copy()
ray_direction = np.array([19.45257568, 7.88531637, 5.96718740])
triangle      = np.array([[-1.80213487, 2.56417131, 2.48542476], [-1.83520162, 2.54618669, 2.47971129], [-1.81366372, 2.53415203, 2.50772786]])
poligons      = triangle.copy()
# barycentrics  = np.array([0.00023483, 0.78492135, 0.21484381])
# t_hit         = 0.41944882
# hit_point-origin = np.array([8.15935993, 3.30748653, 2.50292969]
# <hit_point-origin, ray_direction> = 199.73660278

source = np.array([-10, 0, 0])

mesh_dir = '/home/pparamonov/Projects/mesh-fp-prototype/assets/stl/'

mesh = trimesh.load(mesh_dir + "icosphere_327680.stl")
mesh_vertices = mesh.vertices*4

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')

ax.plot3D(hit_points[0], hit_points[1], hit_points[2], linestyle=':', marker='x')

face_colors = ['b', 'g']

for p in range(int(poligons.shape[0]/3)):
# for p in [int(poligons.shape[0]/3)-2, int(poligons.shape[0]/3)-1]:
    triang = np.vstack((poligons[p*3 : (p+1)*3], poligons[p*3]))
    ax.plot3D(triang[:, 0], triang[:, 1], triang[:, 2])

d = hit_point - source
# d = d / np.linalg.norm(d)

ax.plot3D([hit_points[0]-0.005*d[0], hit_points[0]],
          [hit_points[1]-0.005*d[1], hit_points[1]],
          [hit_points[2]-0.005*d[2], hit_points[2]], linestyle=':', marker='x')

# ax.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
#                 triangles=mesh.faces, alpha=0.5)
# ax.set_xlim([-2, -1])
# ax.set_ylim([ 2,  3])
# ax.set_zlim([ 2,  3])