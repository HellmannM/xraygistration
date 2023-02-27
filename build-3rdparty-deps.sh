#!/bin/bash

BUILD_VSNRAY=1
BUILD_DESKVOX=1

NUM_CORES=24

# 3rdparty/visionaray
if [ $BUILD_VSNRAY == "1" ]; then
    pushd 3rdparty/visionaray
    mkdir build
    cd build
    cmake .. \
        -DCMAKE_POLICY_DEFAULT_CMP0072=NEW \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_CXX_FLAGS="-march=native" \
        -DVSNRAY_ENABLE_PBRT_PARSER=OFF \
        -DVSNRA_ENABLE_EXAMPLES=OFF \
        -DVSNRAY_ENABLE_PEDANTIC=OFF \
        -DVSNRAY_ENABLE_PTEX=OFF
    make -j$NUM_CORES
    popd 
fi

# 3rdparty/deskvox
if [ $BUILD_DESKVOX == "1" ]; then
    pushd 3rdparty/deskvox
    mkdir build
    cd build
    cmake .. \
        -DCMAKE_POLICY_DEFAULT_CMP0072=NEW \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-march=native" \
        -DVISIONARAY_INCLUDE_DIR=./3rdparty/visionaray/include \
        -DVISIONARAY_LIBRARY=./3rdparty/visionaray/build/src/visionaray/libvisionaray.so
    make -j$NUM_CORES
    popd 
fi
