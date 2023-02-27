project(ct-rg-matcher)

cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0072)
    cmake_policy(SET CMP0072 OLD)
endif()

find_package(Boost COMPONENTS filesystem iostreams system thread REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(JPEG)
find_package(TIFF)
find_package(PNG)
find_package(ZLIB)

set(INCLUDE_DIR "./include")
set(SRC_DIR "./src")
set(3RDPARTY_DIR "./3rdparty")
get_filename_component(INCLUDE_DIR "./include" ABSOLUTE)
get_filename_component(SRC_DIR "./src" ABSOLUTE)
get_filename_component(3RDPARTY_DIR "./3rdparty" ABSOLUTE)
set(VISIONARAY_DIR "${3RDPARTY_DIR}/visionaray")
set(VISIONARAY_INCLUDE_DIR "${VISIONARAY_DIR}/include")
set(VISIONARAY_COMMON_DIR "${VISIONARAY_DIR}/src/common")
set(VISIONARAY_BUILD_DIR "${VISIONARAY_DIR}/build")
set(CMD_LINE_DIR "${VISIONARAY_DIR}/src/3rdparty/CmdLine")
set(CMD_LINE_INCLUDE_DIR "${CMD_LINE_DIR}/include")
set(IMGUI_DIR "${VISIONARAY_DIR}/src/3rdparty/imgui")

# Hostprogram
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${TIFF_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIRS})

include_directories(${VISIONARAY_INCLUDE_DIR})
include_directories(${VISIONARAY_COMMON_DIR})
include_directories("${VISIONARAY_COMMON_DIR}/..")
include_directories(${CMD_LINE_INCLUDE_DIR})
include_directories(${IMGUI_DIR})
include_directories(${INCLUDE_DIR})
include_directories("${VISIONARAY_DIR}/src/3rdparty/tinyply/source")


add_definitions(${PNG_DEFINITIONS})

set(MATCH_HEADERS)

set(MATCH_SOURCES
    ${SRC_DIR}/match.cpp
)

add_executable(match ${MATCH_HEADERS} ${MATCH_SOURCES})

target_link_libraries(
    match
    "${VISIONARAY_BUILD_DIR}/src/common/libvisionaray_common.so"
    "${VISIONARAY_BUILD_DIR}/src/visionaray/libvisionaray.so"
    ${ZLIB_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${GLEW_LIBRARY}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${TIFF_LIBRARIES}
    ${PNG_LIBRARIES}
)

