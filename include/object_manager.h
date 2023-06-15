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

#include <map>

#include "cuda/common.cuh"
#include "host_logging.h"
#include "CClassInfo.hpp"

template<class> class CObjectFactory;
class CBaseCudaArray;
class CProjector;

typedef CObjectFactory<CBaseCudaArray> CudaArrayObjectFactory;
typedef CObjectFactory<CProjector>     GPUProjectorObjectFactory;

class CObjectManager : public CClassInfo<CObjectManager>
{
public:
    int create_cuda_array(CObjectFactory<CBaseCudaArray> *pFactory);
    int create_projector (CObjectFactory<CProjector>     *pFactory);
    CBaseCudaArray *get_cuda_array(int id);
    CProjector     *get_projector (int id);
    void delete_cuda_array(int id);
    void delete_projector (int id);
    size_t size_cuda_arrays() const {return size(m_cuda_arrays);}
    size_t size_projector  () const {return size(m_projectors); }
    void clear_cuda_arrays() {clear(m_cuda_arrays);};
    void clear_projectors () {clear(m_projectors );};
private:
    template<typename BaseObjectType>
    int generate_id(const std::map<int, BaseObjectType *> &objects) const;
    std::map<int, CBaseCudaArray *> m_cuda_arrays;
    std::map<int, CProjector*     > m_projectors;
    
    /* ---- Service members ---- */
    template<typename BaseObjectType>
    int create_object(  std::map<int, BaseObjectType *> &objects,
                        CObjectFactory<BaseObjectType> *pFactory);

    template<typename BaseObjectType>
    BaseObjectType *get_object(const std::map<int, BaseObjectType *> &objects, int id) const;
    
    template<typename BaseObjectType>
    void delete_object(std::map<int, BaseObjectType *> &objects, int id);

    template<typename BaseObjectType>
    size_t size(const std::map<int, BaseObjectType *> &objects) const {return objects.size();}

    template<typename BaseObjectType>
    void clear(std::map<int, BaseObjectType *> &objects);

// Singleton definition:
// public:
//     static CObjectManager & instance()
//     {
//         static CObjectManager one;
//         return one;
//     }
 
//     private:
//         CObjectManager()
//         {}
//         ~CObjectManager();
 
//         // CObjectManager(const CObjectManager &);
//         // CObjectManager &operator=(const CObjectManager& );
// };

// extern CObjectManager *global_manager;

protected:
    CObjectManager() {MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Constructing CObjectManager");}
    ~CObjectManager();
    static CObjectManager *m_pInstance;
public:
    static CObjectManager *getSingletonPtr()
    {
        if(m_pInstance == nullptr)
            m_pInstance = new CObjectManager;
        return m_pInstance;
    }
    // static void resetInstancePtr()
    // {
    //     MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Calling CObjectManager::resetInstancePtr()");
    //     if(m_pInstance != nullptr)
    //     { 
    //         MeshFP::Logging::debug(MESH_FP_CLASS_NAME, "Deleting CObjectManager::m_pInstance");            
    //         delete m_pInstance;
    //         m_pInstance = nullptr;
    //     }
    // }
};

extern CObjectManager *pObjectManager;