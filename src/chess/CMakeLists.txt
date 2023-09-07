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
find_package(OpenCV REQUIRED)

visionaray_use_package(Boost)
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(OpenGL)
visionaray_use_package(Threads)
visionaray_use_package(OpenCV)

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
visionaray_link_libraries(${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${VISIONARAY_INCLUDE_DIR})
include_directories(${VISIONARAY_COMMON_DIR})
include_directories(${VISIONARAY_SRC_DIR})
include_directories(${__VSNRAY_CONFIG_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

#--------------

set(CHESS_HEADERS
    render.h
)

set(CHESS_SOURCES
    render.cpp
)

set(CHESS_SOURCES ${CHESS_SOURCES}
    chess.cpp
)

#--------------------------------------------------------------------------------------------------
# Add chess target
#

link_directories(
    ${VISIONARAY_BUILD_DIR}/src/visionaray
    ${VISIONARAY_BUILD_DIR}/src/common
)

visionaray_add_executable(chess
    ${CHESS_HEADERS}
    ${CHESS_SOURCES}
)
