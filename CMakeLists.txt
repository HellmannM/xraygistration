cmake_minimum_required(VERSION 3.25.2)

project(xraygistration)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

#---------------------------------------------------------------------------------------------------
# Directories containing dependencies
#

set(3RDPARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty" CACHE PATH "Path to 3rd-party libs.")
set(visionaray_DIR "${3RDPARTY_DIR}/visionaray/install/lib/cmake/visionaray" CACHE PATH "Path to visionaray cmake module.")
set(CMD_LINE_DIR "${3RDPARTY_DIR}/CmdLine" CACHE PATH "Path to CmdLine.")
set(CMD_LINE_INCLUDE_DIR "${CMD_LINE_DIR}/include" CACHE PATH "Path to CmdLine include dir.")
set(IMGUI_DIR "${VISIONARAY_DIR}/src/3rdparty/imgui" CACHE PATH "Path to ImGUI.")
set(ITK_DIR "${3RDPARTY_DIR}/ITK/install/lib/cmake/ITK-6.0/" CACHE PATH "Path to ITK cmake module.")
set(NLOHMANN_JSON_DIR "${3RDPARTY_DIR}/nlohmann_json" CACHE PATH "Path to nlohmann json dir.")
set(NLOHMANN_JSON_INCLUDE_DIR "${NLOHMANN_JSON_DIR}/include" CACHE PATH "Path to nlohmann json include dir.")
set(IMAGE_TRANSFORM_ESTIMATOR_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/image-transform-estimator" CACHE PATH "Path to Image Transform Estimator include files.")

#--------------------------------------------------------------------------------------------------
# CMake policies to control backward compatible behavior
#

# Old behavior: COMPILE_DEFINITIONS_<CONFIG>
# New behavior: COMPILE_DEFINITIONS + generator expression
if(POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
endif()

if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()

add_subdirectory(anari-drr-viewer)
add_subdirectory(anari-visionaray)
add_subdirectory(image-transform-estimator)
add_subdirectory(chess)

