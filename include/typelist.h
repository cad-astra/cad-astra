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

#include <string>

// Type lists definitions
template<class T, class V>
struct TypeList
{
    typedef T Head;
    typedef V Tail;
};

class NullType
{};

#define  TYPELIST_1(T1) TypeList<T1, NullType>
#define  TYPELIST_2(T1, T2) TypeList<T1, TYPELIST_1(T2)>
#define  TYPELIST_3(T1, T2, T3) TypeList<T1, TYPELIST_2(T2, T3)>
#define  TYPELIST_4(T1, T2, T3, T4) TypeList<T1, TYPELIST_3(T2, T3, T4)>
#define  TYPELIST_5(T1, T2, T3, T4, T5) TypeList<T1, TYPELIST_4(T2, T3, T4, T5)>
#define  TYPELIST_6(T1, T2, T3, T4, T5, T6) TypeList<T1, TYPELIST_5(T2, T3, T4, T5, T6)>
#define  TYPELIST_7(T1, T2, T3, T4, T5, T6, T7) TypeList<T1, TYPELIST_6(T2, T3, T4, T5, T6, T7)>
#define  TYPELIST_8(T1, T2, T3, T4, T5, T6, T7, T8) TypeList<T1, TYPELIST_7(T2, T3, T4, T5, T6, T7, T8)>
#define  TYPELIST_9(T1, T2, T3, T4, T5, T6, T7, T8, T9) TypeList<T1, TYPELIST_8(T2, T3, T4, T5, T6, T7, T8, T9)>
#define TYPELIST_10(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) TypeList<T1, TYPELIST_9(T2, T3, T4, T5, T6, T7, T8, T9, T10)>
#define TYPELIST_11(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) TypeList<T1, TYPELIST_10(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)>
#define TYPELIST_12(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12) TypeList<T1, TYPELIST_11(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)>
#define TYPELIST_13(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13) TypeList<T1, TYPELIST_12(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)>
#define TYPELIST_14(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14) TypeList<T1, TYPELIST_13(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)>
#define TYPELIST_15(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15) TypeList<T1, TYPELIST_14(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15)>
#define TYPELIST_16(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16) TypeList<T1, TYPELIST_15(T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16)>

// namespace TL
// {
//     template<class TList> struct Length;
//     template<> struct Length<NullType>
//     {
//         enum {value = 0};
//     };
//     template <class T, class U>
//     struct Length<TypeList<T, U>>
//     {
//         enum {value = 1 + Length<TypeList<T, U>::Tail>::value};
//     };
//     //
//     template<class TList, unsigned int index> struct TypeAt;
//     template<class Head, class Tail>
//     struct TypeAt<TypeList<Head, Tail>, 0>
//     {
//         typedef Head Result;
//     };
//     template<class Head, class Tail, unsigned int i>
//     struct TypeAt<TypeList<Head, Tail>, i>
//     {
//         typedef typename TypeAt<Tail, i-1>::Result Result;
//     };
// }

template<class TList, class T> struct Append;
template<>
struct Append<NullType, NullType>
{
    typedef NullType Result;
};
template<class T>
struct Append<NullType, T>
{
    typedef TYPELIST_1(T) Result;
};
template<class Head, class Tail>
struct Append<NullType, TypeList<Head, Tail>>
{
    typedef TypeList<Head, Tail> Result;
};
template<class Head, class Tail, class T>
struct Append<TypeList<Head, Tail>, T>
{
    typedef TypeList<Head, typename Append<Tail, T>::Result > Result;
};

template<class Base>
struct functor_find
{
    functor_find() {res = nullptr;}
    bool operator()(std::string name)
    {
        return name == to_find;
    }
    std::string to_find;
    Base *res;
};

template<class TList>
struct CreateObject
{
    template<class U>
    static void find(U &functor)
    {
        if( functor(TList::Head::type) )
        {
            functor.res = new typename TList::Head();
        }
        CreateObject<typename TList::Tail>::find(functor);
    }
};

template<> struct CreateObject<NullType>
{
    template<class U>
    static void find(U &functor)
    {}
}; 

template<typename BaseType, typename TList>
BaseType *create(std::string _sType)
{
    functor_find<BaseType> finder;
    finder.to_find = _sType;
    CreateObject<TList>::find(finder);
    return finder.res;
}
// End of TypeList definitions