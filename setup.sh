#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd "${SCRIPT_DIR}"

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
        -DVSNRAY_ENABLE_EXAMPLES=OFF \
        -DVSNRAY_ENABLE_VIEWER=OFF \
        -DVSNRAY_ENABLE_PEDANTIC=OFF \
        -DVSNRAY_ENABLE_PTEX=OFF
    make -j$NUM_CORES
    popd 
fi

# 3rdparty/deskvox
if [ $BUILD_DESKVOX == "1" ]; then
    VISIONARAY_DIR="$PWD/3rdparty/visionaray"
    pushd 3rdparty/deskvox
    mkdir build
    cd build
    cmake .. \
        -DCMAKE_POLICY_DEFAULT_CMP0072=NEW \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-march=native -I$VISIONARAY_DIR/build/config" \
        -DCUDA_NVCC_FLAGS="-I$VISIONARAY_DIR/build/config" \
        -DVISIONARAY_INCLUDE_DIR="$VISIONARAY_DIR/include" \
        -DVISIONARAY_LIBRARY="$VISIONARAY_DIR/build/src/visionaray/libvisionaray.so"
    make -j$NUM_CORES
    popd 
fi

popd
