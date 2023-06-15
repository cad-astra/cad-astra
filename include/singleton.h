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

#ifndef SINGLETON_H
#define SINGLETON_H
// #include <iostream>

template<typename T>
class CSingleton
{
public:
    virtual ~CSingleton() {}
protected:
    CSingleton() { /* std::cerr << "Created a Singleton...\n"; */ }
    static T *m_pInstance;
public:
    static T *getSingletonPtr()
    {
        if(m_pInstance == nullptr)
            m_pInstance = new T;
        return m_pInstance;
    }

    static void resetInstancePtr()
    {
        // std::cerr << "Calling CSingleton::resetInstancePtr()" << std::endl;
        if(m_pInstance != nullptr)
        { 
            // std::cerr << "Deleting CSingleton::m_pInstance" << std::endl;
            delete m_pInstance;
            m_pInstance = nullptr;
        }
    }
};

template<typename T> T* CSingleton<T>::m_pInstance = nullptr;

#endif