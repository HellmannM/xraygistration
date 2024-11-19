set(glm_DIR ${CMAKE_CURRENT_LIST_DIR}/external/glm/lib/cmake/glm)
find_package(glm REQUIRED)
mark_as_advanced(glm_DIR)

find_package(anari 0.8.0 REQUIRED COMPONENTS viewer)

set(SUBPROJECT_NAME anariDRRViewer)

add_executable(${SUBPROJECT_NAME}
    SettingsEditor.cpp Viewport.cpp viewer.cpp)
target_link_libraries(${SUBPROJECT_NAME} glm::glm anari::anari_viewer)

option(USE_ITK "Support loading Nifti files from ITK" ON)
if (USE_ITK)
  find_package(ITK CONFIG REQUIRED)
  include(${ITK_USE_FILE})
  include_directories(${ITK_INCLUDE_DIRS})
  target_sources(${SUBPROJECT_NAME} PRIVATE readNifti.cpp)
  target_compile_definitions(${SUBPROJECT_NAME} PRIVATE -DHAVE_ITK)
  target_link_libraries(${SUBPROJECT_NAME} ${ITK_LIBRARIES})
endif()
