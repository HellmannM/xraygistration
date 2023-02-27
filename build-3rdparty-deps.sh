#!/bin/bash

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
make -j24
popd 
