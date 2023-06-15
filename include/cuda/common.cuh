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

#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

#define NUM_BLOCKS(N_ELEMS, BLOCK_SIZE) (N_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE

#define UNPACK_3D_POINT(ARRAY, INDEXER, N0, N1) \
    {ARRAY[INDEXER(N0, N1)], ARRAY[INDEXER(N0, N1+1)], ARRAY[INDEXER(N0, N1+2)]}

void check(cudaError_t result, char const *const func, const char *const file, int const line);

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define CUDA_CALL(val) check((val), #val, __FILE__, __LINE__)

// -------------------------------------------------- //
//                      DEBUG PRINT
// -------------------------------------------------- //
// #define __DEBUG_PRINTOUT__ 0     // This is set by build script, see build_python.sh
#if __DEBUG_PRINTOUT__
// DEBUG_PRINTLN automatically adds a newline
#define DEBUG_PRINTLN(msg, ...) \
    printf(msg, ## __VA_ARGS__);\
    printf("\n");
#else
#define DEBUG_PRINTLN(...)  // the macro compiles to nothing
#endif

#define PRINT_FLOAT3(COMMENT, POINT) \
        printf("%s (%5.3f, %5.3f, %5.3f)\n", COMMENT, POINT.x, POINT.y, POINT.z)

#endif
