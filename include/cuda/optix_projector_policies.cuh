/*
    This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
    Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.

    This program is free software: you can redistribute it and/or modify  
    it under the terms of the GNU General Public License as published by  
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

enum OptiXGeometryPolicyIndex
{
    OptiXRotationPolicy = 0,                                      // "__direct_callable__rotation_geom_policy"
    OptiXVectorPolicy,                                            // "__direct_callable__vector_geom_policy"
    OptiXMeshTransformPolicy,                                     // "__direct_callable__vector_geom_policy"
    OptiXGeometryPolicySize
};

enum OptiXDetectorPolicyIndex
{
    OptiXMaxAttenuationPolicy    = OptiXGeometryPolicySize,       // "__direct_callable__max_attenuation_detector_pixel_policy"
    OptiXSumIntencityPolicy      = OptiXMaxAttenuationPolicy+1,   // "__direct_callable__sum_intensity_detector_pixel_policy"
    OptiXSumShareIntencityPolicy = OptiXSumIntencityPolicy+1,
    OptiXDetPolicySize
};

enum OptiXDetectorGeometryPolicyIndex
{
    OptiXPlanePolicy    = OptiXDetPolicySize,
    OptiXCylinderPolicy = OptiXPlanePolicy + 1,
    OptiXDetectorGeometryPolicySize
};

enum OptiXTracerPolicyIndex
{
    OptiXRecursivePolicy    = OptiXDetectorGeometryPolicySize,
    OptiXNonRecursivePolicy = OptiXRecursivePolicy+1
};