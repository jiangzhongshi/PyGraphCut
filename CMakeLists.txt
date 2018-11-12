cmake_minimum_required(VERSION 3.1)
project(pygraphcut)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(PythonLibs 3.6 REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
set(PYBIND11_INCLUDE_DIR "$ENV{PYBIND11_INCLUDE_DIR}")
set(PYBIND11_PYTHON_VERSION 3.6)


# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_CGAL            "Use OpenGL viewer"  ON)

find_package(LIBIGL REQUIRED QUIET)
set(PYBIND11_DIR ${LIBIGL_INCLUDE_DIR}/../external/pybind11)
add_subdirectory(${PYBIND11_DIR}/ pybind11)
#
# Add your project files

pybind11_add_module(pygraphcut MODULE main.cpp)
target_link_libraries(pygraphcut PRIVATE pybind11::module igl::core igl::opengl_glfw igl::cgal)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")
#target_include_directories(${PROJECT_NAME}_bin PUBLIC external/cnpy)
