cmake_minimum_required(VERSION 3.13)
project(dune-io CXX)

if(NOT (dune-common_DIR OR dune-common_ROOT OR
      "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*"))
    string(REPLACE  ${PROJECT_NAME} dune-common dune-common_DIR
      ${PROJECT_BINARY_DIR})
endif()

#find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules"
  ${dune-common_MODULE_PATH})

#include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()

dune_enable_all_packages()

add_subdirectory(gridformat)
add_subdirectory(dune)
add_subdirectory(doc)
add_subdirectory(cmake/modules)

set(CMAKE_BUILD_TYPE Release)

find_package(MPI)  # TODO: double-check why HAVE_MPI is not set automatically

dune_add_test(NAME test_io SOURCES test.cpp)
target_link_libraries(test_io PRIVATE gridformat::gridformat)
target_compile_options(test_io PRIVATE "-Wno-abi")
target_compile_definitions(test_io PRIVATE HAVE_MPI=${MPI_FOUND})

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
