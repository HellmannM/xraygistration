cmake_minimum_required(VERSION 3.25.2)

project(match)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})


#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Boost COMPONENTS filesystem iostreams system thread REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ITK CONFIG REQUIRED)

include(${ITK_USE_FILE})

visionaray_use_package(Boost)
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(OpenGL)
visionaray_use_package(Threads)
visionaray_use_package(OpenCV)

# CUDA
if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    set(CUDA_ARCHITECTURES "native")
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
visionaray_link_libraries(${ITK_LIBRARIES})
visionaray_link_libraries(${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${VISIONARAY_INCLUDE_DIR})
include_directories(${VISIONARAY_COMMON_DIR})
include_directories(${VISIONARAY_SRC_DIR})
include_directories(${__VSNRAY_CONFIG_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})
include_directories(${ITK_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})


#--------------

set(MATCH_HEADERS
    host_device_rt.h
    match_result.h
    feature_matcher.h
    render.h
    timer.h
    volume_reader.h
    attenuation.h
)

set(MATCH_SOURCES
    host_device_rt.cpp
    render.cpp
    volume_reader.cpp
)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -fPIC; -O3; "--extended-lambda --expt-extended-lambda")
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
)

visionaray_add_executable(match
    ${MATCH_HEADERS}
    ${MATCH_SOURCES}
    ${MATCH_CUDA_SOURCES}
)

visionaray_add_library(render SHARED
    ${MATCH_SOURCES}
    ${MATCH_CUDA_SOURCES}
)
set_property(TARGET render PROPERTY POSITION_INDEPENDENT_CODE ON)

