#!/bin/bash -x

PROJ_DIR=$(pwd)
BUILD_DIR=$PROJ_DIR/build
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory $BUILD_DIR"
    mkdir $BUILD_DIR
fi

set -e
# compile cuda libs
nvcc $PROJ_DIR/src/common.cu --compiler-options -fPIC -c -o $BUILD_DIR/common.o -g -G -O0 -arch=sm_60
nvcc $PROJ_DIR/src/project_mesh.cu -dc -o $BUILD_DIR/project_mesh.o -g -G -O0 -arch=sm_60
nvcc $PROJ_DIR/src/tests.cu -dc -o $BUILD_DIR/tests.o -g -G -O0 -arch=sm_60
nvcc $PROJ_DIR/src/cuManagedArray.cu -dc -o $BUILD_DIR/cuManagedArray.o -g -G -O0 -arch=sm_60

g++ $PROJ_DIR/src/object_manager.cpp -O0 -g -c -o $BUILD_DIR/object_manager.o


OUTFILE="run_tests.out"  # run_tests.exe
# compile tests
nvcc $BUILD_DIR/tests.o $BUILD_DIR/common.o $BUILD_DIR/project_mesh.o $BUILD_DIR/object_manager.o $BUILD_DIR/cuManagedArray.o --compiler-options -fPIC -o $BUILD_DIR/$OUTFILE -g -G -O0 -arch=sm_60

${BUILD_DIR}/$OUTFILE
