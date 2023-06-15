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

#include "../include/object_manager.h"
#include "../include/CObjectFactory.h"
#include "../include/CBaseCudaArray.h"
#include "../include/CProjector.h"

#include "../include/cuda/common.cuh"

#include <chrono>
#include <sstream>

typedef std::chrono::duration<int, std::ratio<1,1>> int_sec;

CObjectManager* CObjectManager::m_pInstance = nullptr;
CObjectManager *pObjectManager = CObjectManager::getSingletonPtr();

CObjectManager::~CObjectManager()
{
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME,"Destroying CObjectManager");
    clear_cuda_arrays();
    clear_projectors ();
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME,"Cleared maps in CObjectManager");
}

int CObjectManager::create_cuda_array(CObjectFactory<CBaseCudaArray> *pFactory)
{
    int id = create_object(m_cuda_arrays, pFactory);
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME,"adding cuda array with id=%d, total arrays: %lu", id, m_cuda_arrays.size());
    return id;
}

int CObjectManager::create_projector(CObjectFactory<CProjector>     *pFactory)
{
    int id = create_object(m_projectors, pFactory);
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME,"adding projector with id=%d, total projectors: %lu", id, m_projectors.size());
    return id;
}

CBaseCudaArray *CObjectManager::get_cuda_array(int id)
{
    return get_object(m_cuda_arrays, id);
}

CProjector     *CObjectManager::get_projector (int id)
{
    return get_object(m_projectors, id);
}

void CObjectManager::delete_cuda_array(int id)
{
    delete_object(m_cuda_arrays, id);
}

void CObjectManager::delete_projector (int id)
{
    delete_object(m_projectors, id);
}

template<typename BaseObjectType>
int CObjectManager::generate_id(const std::map<int, BaseObjectType *> &objects) const
{
    auto now = std::chrono::system_clock::now();
    int id = static_cast<int>(objects.size()) +
             std::chrono::duration_cast<int_sec>(now.time_since_epoch()).count();
    return id;
}

template<typename BaseObjectType>
int CObjectManager::create_object(  std::map<int, BaseObjectType *> &objects,
                                    CObjectFactory<BaseObjectType> *pFactory)
{
    int id = generate_id(objects);
    BaseObjectType * pObject = pFactory->create_object();
    if(pObject == nullptr)
        throw std::runtime_error("CObjectManager: Could not create an object");
    objects[id] = pObject;
    return id;
}

template<typename BaseObjectType>
BaseObjectType *CObjectManager::get_object(const std::map<int, BaseObjectType *> &objects, int id) const
{
    if(objects.count(id) < 1)
    {
        std::stringstream ss;
        ss << "Error: no object with id " << id;
        throw std::invalid_argument(ss.str());
    }
    return objects.at(id);
}

template<typename BaseObjectType>
void CObjectManager::delete_object(std::map<int, BaseObjectType *> &objects, int id)
{
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Deleting object with id %d", id);
    if(objects.count(id) < 1)
    {
        std::stringstream ss;
        ss << "Error: no object with id " << id;
        throw std::invalid_argument(ss.str());
    }
    delete objects[id];
    objects.erase(id);
}

template<typename BaseObjectType>
void CObjectManager::clear(std::map<int, BaseObjectType *> &objects)
{
    MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Starting clear in CObjectManager");

    for(auto &object_record: objects)
    {
        MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Deleting object with id %d", object_record.first);
        // DEBUG_PRINTLN("cuda array object size: %lu", object_record.second->size());
        delete object_record.second;
    }
    objects.clear();
}