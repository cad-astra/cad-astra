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

#ifndef CCLASSINFO_HPP
#define CCLASSINFO_HPP

#ifdef __GNUC__
    #include <cxxabi.h>
#endif

template <typename TClass>
class CClassInfo
{
public:
    // Consider making it static:
    const char *class_name()
    {
        #ifdef __GNUC__
        return abi::__cxa_demangle(typeid(TClass).name(), nullptr, nullptr, nullptr);
        #else   // MSVC
        return typeid(TClass).name();
        #endif
    }
};

#define MESH_FP_CLASS_NAME this->class_name()

#endif