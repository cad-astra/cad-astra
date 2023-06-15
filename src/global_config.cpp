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

#include "../include/global_config.h"

std::string MeshFP::CGlobalConfig::ptx_dir = "";
int MeshFP::CGlobalConfig::optix_log_level = 1;

std::string MeshFP::CGlobalConfig::get_ptx_dir()
{
    return ptx_dir;
}

void MeshFP::CGlobalConfig::set_ptx_dir(const std::string &dir)
{
    ptx_dir = dir;
}

int MeshFP::CGlobalConfig::get_optix_log_level()
{
    return optix_log_level;
}
void MeshFP::CGlobalConfig::set_optix_log_level(int level)
{
    optix_log_level = level;
}