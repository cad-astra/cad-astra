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

class Transformation:
    def __init__(self, begin_time=0, end_time=1):
        self.__begin_time = begin_time
        self.__end_time   = end_time

    def optix_srt_data(self):
        pass

    @property
    def begin_time(self):
        return self.__begin_time

    @property
    def end_time(self):
        return self.__end_time

class RotationTransformation(Transformation):
    def __init__(self, begin_time=0, end_time=1, angles=np.zeros((2,)), axis='x'):
        super().__init__(begin_time, end_time)
        self.__angles = angles
        self.__axis   = axis
    
    def optix_srt_data(self):
        srt_vec = np.empty(shape=(len(self.__angles), 16), dtype=np.float32)
        srt_rotation_ax = {'x': 9, 'y': 10, 'z': 11}
        for i, ang in enumerate(self.__angles):
            srt_vec[i] = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, np.cos(ang*0.5), 0, 0, 0])
            srt_vec[i, srt_rotation_ax[self.__axis]] = np.sin(ang*0.5)
        return srt_vec

class TranslationTransformation(Transformation):
    def __init__(self, begin_time=0, end_time=1, points=np.zeros((2,3))):
        super().__init__(begin_time, end_time)
        self.__points = points

    def optix_srt_data(self):
        srt_vec = np.empty(shape=(self.__points.shape[0], 16), dtype=np.float32)
        for i, point in enumerate(self.__points[:]):
            srt_vec[i] = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, point[0], point[1], point[2]])
        return srt_vec

def create_transformation(trans_type: str, begin_time: float=0, end_time: float=1, *args, **kwargs):
    if trans_type == 'rotation':
        nargin = len(args)
        angles = args[0] if nargin > 0 else kwargs.get('angles', np.zeros((2,)))
        axis = args[1] if nargin > 1 else kwargs.get('axis', 'x')
        return RotationTransformation(begin_time, end_time, angles, axis)
    if trans_type == 'translation':
        nargin = len(args)
        points = args[0] if nargin > 0 else kwargs.get('points', np.zeros((2,3)))
        return TranslationTransformation(begin_time, end_time, points)
