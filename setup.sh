#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd "${SCRIPT_DIR}"

git submodule update --init --recursive
mkdir 3rdparty/visionaray/build
pushd 3rdparty/visionaray/build
cmake .. -DCMAKE_CXX_FLAGS="-std=c++17" -DCMAKE_BUILD_TYPE="Release" -DBUILD_SHARED_LIBS="ON" -DVSNRAY_ENABLE_PBRT_PARSER="OFF" -DVSNRAY_ENABLE_EXAMPLES="OFF" -DVSNRAY_ENABLE_VIEWER="OFF" -DVSNRAY_ENABLE_CUDA="OFF"
make -j8
popd

mkdir build
pushd build
cmake ..
make -j8
popd

popd
