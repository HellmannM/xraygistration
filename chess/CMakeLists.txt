find_package(OpenCV REQUIRED)
find_package(visionaray 0.4.2 REQUIRED)

set(SUBPROJECT_NAME chessCalibration)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(${SUBPROJECT_NAME}
    ${CMD_LINE_DIR}/src/CmdLine.cpp
    ${CMD_LINE_DIR}/src/CmdLineUtil.cpp
    render.h
    render.cpp
    chess.cpp
)
target_link_libraries(${SUBPROJECT_NAME} visionaray::visionaray_common)
target_link_libraries(${SUBPROJECT_NAME} ${OpenCV_LIBS})

target_include_directories(${SUBPROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(${SUBPROJECT_NAME} PUBLIC ${CMD_LINE_INCLUDE_DIR})
