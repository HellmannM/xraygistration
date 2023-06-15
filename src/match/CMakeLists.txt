# This file is distributed under the MIT license.
# See the LICENSE file for details.

#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Boost COMPONENTS filesystem iostreams system thread REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(Nifti REQUIRED)
find_package(OpenCV REQUIRED)

visionaray_use_package(Boost)
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(OpenGL)
visionaray_use_package(Threads)
visionaray_use_package(Nifti)
visionaray_use_package(OpenCV)

# CUDA

if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    visionaray_use_package(CUDA)
endif()

# TBB

if (VSNRAY_ENABLE_TBB)
    find_package(TBB)
    visionaray_use_package(TBB)
endif()

#--------------------------------------------------------------------------------------------------
#
#

visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)
visionaray_link_libraries(virvo_fileio)
visionaray_link_libraries(virvo_transfunc)
visionaray_link_libraries(virvo)
visionaray_link_libraries(${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${VISIONARAY_INCLUDE_DIR})
include_directories(${VISIONARAY_COMMON_DIR})
include_directories(${VISIONARAY_SRC_DIR})
include_directories(${__VSNRAY_CONFIG_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})
include_directories(${DESKVOX_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

#--------------

set(MATCH_HEADERS
    host_device_rt.h
    match_result.h
    render.h
)

set(MATCH_SOURCES
    host_device_rt.cpp
    render.cpp
)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--extended-lambda --expt-extended-lambda")
    visionaray_cuda_compile(MATCH_CUDA_SOURCES
        render.cu
        match.cu
    )
else()
    set(MATCH_SOURCES ${MATCH_SOURCES}
        match.cpp
    )
endif()

#--------------------------------------------------------------------------------------------------
# Add match target
#

link_directories(
    ${VISIONARAY_BUILD_DIR}/src/visionaray
    ${VISIONARAY_BUILD_DIR}/src/common
    ${DESKVOX_BIN_DIR}
)

visionaray_add_executable(match
    ${MATCH_HEADERS}
    ${MATCH_SOURCES}
    ${MATCH_CUDA_SOURCES}
)
