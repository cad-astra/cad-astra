#!/bin/bash

# Env. variables defaults 
DEFAULT_NVCC="nvcc"
DEFAULT_SM_ARCH="sm_75"
DEFAULT_CPP="g++-9"
DEFAULT_DEBUG_PRINTOUT="FALSE"
DEFAULT_USE_FAST_MATH="FALSE" # turn on/off nvcc -use_fast_math
DEFAULT_OPTIX_MATERIAL_STACK_SIZE="8"

wrong_argument_message="Unknown argument: "
help_message="Usage: WITH_OPTIX=<TRUE|FALSE> OPTIX_DIR=<OptiX 7.x root directory> $0 <install | clean | help | -h | --help>"
env_vars_list="Other environment variables that control build:\n\
\tCUDA_INCLUDE_DIR : directory with cuda header files. If not set, some default paths are tried, e.g., /usr/local/cuda/include.\n
\tNVCC: path to the CUDA compiler. If not specified, the script will try to find it in one of the standard locations. If that does'n work, the ${DEFAULT_NVCC} will be used.\n
\tSM_ARCH: CUDA compute capability in the following format: sm_<compute capability>. If not specified, ${DEFAULT_SM_ARCH} will be used.\n
\tCPP: host c++ compiler for nvcc. If not specified, ${DEFAULT_CPP} will be used.\n
\tDEBUG_PRINTOUT: allow debug printout. Possible values: TRUE or FALSE. May impact performance. If not specified, ${DEFAULT_DEBUG_PRINTOUT} will be used.\n
\tUSE_FAST_MATH: turn on/off nvcc -use_fast_math. Possible values: TRUE or FALSE. Impacts performance and precision. If not specified, ${DEFAULT_USE_FAST_MATH} will be used\n
\tOPTIX_MATERIAL_STACK_SIZE: maximal allowed size of the material stack in OptiX recursive tracing. Impacts performance. If not specified, ${DEFAULT_OPTIX_MATERIAL_STACK_SIZE} will be used."

export PROJ_DIR=$(pwd)
BUILD_DIR=$PROJ_DIR/build

DISTUTILS_INSTALL=""
if [[ $# -ge 1 ]]; then
    if [[ $# -gt 1 ]]; then
        echo "Only one command line argument is allowed"
        echo ${help_message}
        echo -e ${env_vars_list}
        exit
    else
        if [[ "$1" == "install" ]]; then
            DISTUTILS_INSTALL=install
        elif [[ "$1" == "help" || "$1" == "-h" || "$1" == "--help" ]]; then
            echo ${help_message}
            echo -e ${env_vars_list}
            exit
        elif [[ "$1" == "clean" ]]; then
            if [[ -d ${BUILD_DIR} ]]; then
                echo "Cleaning up ${BUILD_DIR} directory..."
                if [ ! -z "$(ls -A ${BUILD_DIR})" ]; then
                    rm -r ${BUILD_DIR}/*
                    echo "Done..."
                else
                    echo "${BUILD_DIR} is empty, nothing to clean..."
                fi
            else
                echo "${BUILD_DIR} does not exist, nothing to clean..."
            fi
            exit
        else
            echo "${wrong_argument_message} $1"
            echo ${help_message}
            echo -e ${env_vars_list}
            exit
        fi
    fi
fi

# Find cuda include directory
FOUND_NVCC=DEFAULT_NVCC
if [[ "${CUDA_INCLUDE_DIR}" == "" ]]; then
    if [[ -f "/usr/local/cuda/lib64/libcudart.so" ]]; then
        CUDA_INCLUDE_DIR="/usr/local/cuda/include"
        FOUND_NVCC="/usr/local/cuda/bin/nvcc"
    fi
    if [[ -f "/usr/local/cuda/lib/x64/libcudart.so" ]]; then
        CUDA_INCLUDE_DIR="/usr/local/cuda/include"
        FOUND_NVCC="/usr/local/cuda/bin/nvcc"
    fi
    if [[ -f "/usr/lib/x86_64-linux-gnu/libcudart.so" ]]; then
        CUDA_INCLUDE_DIR="/usr/include"
        FOUND_NVCC="/usr/bin/nvcc"
    fi
fi

if [[ "${NVCC}" == "" ]]; then
    NVCC=${FOUND_NVCC}
fi
echo "Picked $NVCC as the NVCC compiler"

if [[ "${SM_ARCH}" == "" ]]; then
    SM_ARCH=${DEFAULT_SM_ARCH}
fi

if [[ "${CPP}" == "" ]]; then
    CPP=$DEFAULT_CPP
fi
echo "Picked $CPP as the host compiler for NVCC"

if [[ "${USE_FAST_MATH}" == "" ]]; then
    USE_FAST_MATH=${DEFAULT_USE_FAST_MATH}
fi

fast_math=""
if [[ "${USE_FAST_MATH}" == "TRUE" ]]; then
    fast_math="--use_fast_math"
fi

if [[ "${DEBUG_PRINTOUT}" == "" ]]; then
    DEBUG_PRINTOUT=${DEFAULT_DEBUG_PRINTOUT}
fi

debug_macro="0"
if [[ "${DEBUG_PRINTOUT}" == "TRUE" ]]; then
    debug_macro="1"
fi

if [[ "${OPTIX_MATERIAL_STACK_SIZE}" == "" ]]; then
    OPTIX_MATERIAL_STACK_SIZE=$DEFAULT_OPTIX_MATERIAL_STACK_SIZE
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory $BUILD_DIR"
    mkdir $BUILD_DIR
fi

if [ ! -d "$BUILD_DIR/python" ]; then
    echo "Creating build directory $BUILD_DIR/python"
    mkdir $BUILD_DIR/python
fi
if [ ! -d "$BUILD_DIR/obj" ]; then
    echo "Creating build directory $BUILD_DIR/obj"
    mkdir $BUILD_DIR/obj
fi
if [ ! -d "$BUILD_DIR/lib" ]; then
    echo "Creating build directory $BUILD_DIR/lib"
    mkdir $BUILD_DIR/lib
fi

if [[ "${WITH_OPTIX}" == "TRUE" ]]; then
    # TODO: find include/optix.h instead of just include/
    if [[ "${OPTIX_DIR}" == "" ]]; then
        echo "Please set the OPTIX_DIR environment variable to OptiX 7 root directory"
        exit
    elif [ ! -d "${OPTIX_DIR}/include" ]; then
        echo "Failed to detect OptiX 7 in ${OPTIX_DIR}: could not find the 'include' subdirectory"
        exit
    fi
fi

set -e

echo "Compiling object files..."
${NVCC} $PROJ_DIR/src/common.cu -D__DEBUG_PRINTOUT__=${debug_macro} ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -c -o ${BUILD_DIR}/obj/common.o -arch=${SM_ARCH} -O3 #-g -G -O0
${NVCC} $PROJ_DIR/src/cuManagedArray.cu -D__DEBUG_PRINTOUT__=${debug_macro} ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -c -o ${BUILD_DIR}/obj/cuManagedArray.o -arch=${SM_ARCH} -O3 #-g -G -O0
${NVCC} $PROJ_DIR/src/CRasterProjector.cu -D__DEBUG_PRINTOUT__=${debug_macro} ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -c -o ${BUILD_DIR}/obj/CRasterProjector.o -arch=${SM_ARCH} -O3
${NVCC} $PROJ_DIR/src/array_index.cu -D__DEBUG_PRINTOUT__=${debug_macro} ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -c -o ${BUILD_DIR}/obj/array_index.o -arch=${SM_ARCH} -O3
${NVCC} $PROJ_DIR/src/gpu_utils.cu   -D__DEBUG_PRINTOUT__=${debug_macro} ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -c -o ${BUILD_DIR}/obj/gpu_utils.o -I${CUDA_INCLUDE_DIR} -arch=${SM_ARCH} -O3
${CPP} -D__DEBUG_PRINTOUT__=${debug_macro} $PROJ_DIR/src/object_manager.cpp -c -o ${BUILD_DIR}/obj/object_manager.o -fPIC -I${CUDA_INCLUDE_DIR} #-g -O0
${CPP} -D__DEBUG_PRINTOUT__=${debug_macro} $PROJ_DIR/src/global_config.cpp  -c -o ${BUILD_DIR}/obj/global_config.o  -fPIC -I${CUDA_INCLUDE_DIR} #-g -O0
${CPP} -D__DEBUG_PRINTOUT__=${debug_macro} $PROJ_DIR/src/CHostLogger.cpp   -c -o ${BUILD_DIR}/obj/CHostLogger.o  -fPIC -I${CUDA_INCLUDE_DIR} #-g -O0

if [[ "${WITH_OPTIX}" == "TRUE" ]]; then
    echo "Generating PTX code from OptiX programs..."
    # generate PTX
    ${NVCC} ${PROJ_DIR}/src/optix_projector.cu -D__DEBUG_PRINTOUT__=${debug_macro} -DMAX_OPTIX_MATERIAL_STACK_SIZE=${OPTIX_MATERIAL_STACK_SIZE} -ptx --relocatable-device-code=true -o ${BUILD_DIR}/optix_projector_generated.cu.ptx -ccbin ${CPP} -m64 --std c++14 -arch=${SM_ARCH} ${fast_math} -I${OPTIX_DIR}/include # --generate-line-info -Wno-deprecated-gpu-targets
    ${NVCC} ${PROJ_DIR}/src/optix_projector_non_recursive.cu -D__DEBUG_PRINTOUT__=${debug_macro} -DMAX_OPTIX_MATERIAL_STACK_SIZE="1" -ptx --relocatable-device-code=true -o ${BUILD_DIR}/optix_projector_non_recursive_generated.cu.ptx -ccbin ${CPP} -m64 --std c++14 -arch=${SM_ARCH} ${fast_math} -I${OPTIX_DIR}/include # --generate-line-info -Wno-deprecated-gpu-targets
    ${NVCC} ${PROJ_DIR}/src/optix_projector_policies.cu -D__DEBUG_PRINTOUT__=${debug_macro} -ptx --relocatable-device-code=true -o ${BUILD_DIR}/optix_projector_policies_generated.cu.ptx -ccbin ${CPP} -m64 --std c++14 -arch=${SM_ARCH} ${fast_math} -I${OPTIX_DIR}/include # --generate-line-info -Wno-deprecated-gpu-targets
    cp ${BUILD_DIR}/*.ptx ${PROJ_DIR}/python/mesh_projector
    echo "Compiling OptiX host helper functions..."
    ${NVCC} ${PROJ_DIR}/src/optix_host_functions.cu -D__DEBUG_PRINTOUT__=${debug_macro} -DWITH_OPTIX_PROJECTOR ${fast_math} --compiler-options -fPIC -c -o ${BUILD_DIR}/obj/optix_host_functions.o -I${CUDA_INCLUDE_DIR} -I${OPTIX_DIR}/include -I${OPTIX_DIR}/SDK -I${OPTIX_DIR}/SDK/support -I/${BUILD_DIR} -O3 #-g -G -O0
    echo "Compiling OptiX projector class..."
    # OptiX projector class
    ${NVCC} ${PROJ_DIR}/src/COptiXProjector.cu -D__DEBUG_PRINTOUT__=${debug_macro} -DWITH_OPTIX_PROJECTOR ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -c -o ${BUILD_DIR}/obj/COptiXProjector.o -I${CUDA_INCLUDE_DIR} -I${OPTIX_DIR}/include -I${OPTIX_DIR}/SDK -I${OPTIX_DIR}/SDK/support -I/${BUILD_DIR} -O3 #-g -G -O0
    echo "Linking libproject_mesh.so..."
    ${NVCC} ${BUILD_DIR}/obj/CHostLogger.o ${BUILD_DIR}/obj/array_index.o ${BUILD_DIR}/obj/optix_host_functions.o ${BUILD_DIR}/obj/CRasterProjector.o ${BUILD_DIR}/obj/COptiXProjector.o $BUILD_DIR/obj/cuManagedArray.o ${BUILD_DIR}/obj/gpu_utils.o $BUILD_DIR/obj/common.o ${BUILD_DIR}/obj/object_manager.o ${BUILD_DIR}/obj/global_config.o ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -shared -o $BUILD_DIR/lib/libproject_mesh.so -L$BUILD_DIR/lib -arch=${SM_ARCH} -O3 #-g -G -O0
else
    ${NVCC} ${BUILD_DIR}/obj/CHostLogger.o ${BUILD_DIR}/obj/array_index.o ${BUILD_DIR}/obj/CRasterProjector.o $BUILD_DIR/obj/cuManagedArray.o ${BUILD_DIR}/obj/gpu_utils.o $BUILD_DIR/obj/common.o ${BUILD_DIR}/obj/object_manager.o ${BUILD_DIR}/obj/global_config.o ${fast_math} -ccbin ${CPP} --compiler-options -fPIC -shared -o $BUILD_DIR/lib/libproject_mesh.so -L$BUILD_DIR/lib -arch=${SM_ARCH} -O3 #-g -G -O0
fi

# build python extension
cd $PROJ_DIR/python
# LDFLAGS="-L$BUILD_DIR -lproject_mesh -Wl,-rpath=$BUILD_DIR" python setup.py build --build-base=$BUILD_DIR/python
echo "Starting setuptools..."
python setup.py build --build-base=$BUILD_DIR/python ${DISTUTILS_INSTALL}
# TODO: install with pip:
# PROJ_DIR=$(pwd)/../ pip install .

# Clean up
if [[ "${WITH_OPTIX}" == "TRUE" ]]; then
    rm ${PROJ_DIR}/python/mesh_projector/*.ptx  # clean up ptx files after install
fi
rm ${PROJ_DIR}/python/mesh_projector/*.cpp # clean cpp files created by cython
