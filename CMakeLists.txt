project(ct-rg-matcher)

cmake_minimum_required(VERSION 3.1.3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#---------------------------------------------------------------------------------------------------
# Directories containing dependencies
#

set(3RDPARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(VISIONARAY_DIR "${3RDPARTY_DIR}/visionaray")
set(VISIONARAY_INCLUDE_DIR "${VISIONARAY_DIR}/include")
set(VISIONARAY_COMMON_DIR "${VISIONARAY_DIR}/src/common")
set(VISIONARAY_SRC_DIR "${VISIONARAY_DIR}/src")
set(VISIONARAY_BUILD_DIR "${VISIONARAY_DIR}/build")
#set(__VSNRAY_LINK_LIBRARIES ${__VSNRAY_LINK_LIBRARIES} ${VISIONARAY_BUILD_DIR}/src/visionaray ${VISIONARAY_BUILD_DIR}/src/common)
set(CMD_LINE_DIR "${3RDPARTY_DIR}/CmdLine")
set(CMD_LINE_INCLUDE_DIR "${CMD_LINE_DIR}/include")
set(IMGUI_DIR "${VISIONARAY_DIR}/src/3rdparty/imgui")
set(DESKVOX_DIR "${3RDPARTY_DIR}/deskvox")
set(DESKVOX_INCLUDE_DIR "${DESKVOX_DIR}/virvo")
set(DESKVOX_BIN_DIR "${DESKVOX_DIR}/build/Release/bin")
set(NLOHMANN_JSON_DIR "${3RDPARTY_DIR}/nlohmann_json")
set(NLOHMANN_JSON_INCLUDE_DIR "${NLOHMANN_JSON_DIR}/include")

#--------------------------------------------------------------------------------------------------
# CMake modules
#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${VISIONARAY_DIR}/cmake
    ${VISIONARAY_DIR}/cmake/modules
    ${DESKVOX_DIR}/cmake
    ${DESKVOX_DIR}/cmake/modules
)

include(VisionarayAddTarget)
include(VisionarayUseFramework)
include(VisionarayUsePackage)

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

#---------------------------------------------------------------------------------------------------
# Directories containing config headers
#

# Config file location (#include <visionaray/config.h>,#include <common/config.h>,...)
set(__VSNRAY_CONFIG_DIR "${PROJECT_SOURCE_DIR}/3rdparty/visionaray/build/config")

# Config file location for the public visionaray library
set(__VSNRAY_VISIONARAY_CONFIG_DIR "${__VSNRAY_CONFIG_DIR}/visionaray")

# Config file location for the private visionaray_common library
set(__VSNRAY_COMMON_CONFIG_DIR "${__VSNRAY_CONFIG_DIR}/common")


#---------------------------------------------------------------------------------------------------
# Options
#
option(VSNRAY_ENABLE_CUDA "Use CUDA, if available" ON)
option(VSNRAY_ENABLE_TBB "Use TBB, if available" ON)

add_subdirectory(src)

