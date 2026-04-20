cmake_minimum_required(VERSION 3.18)

include(FetchContent)
set(PYBIND11_FINDPYTHON ON)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

pybind11_add_module(pytamm MODULE NO_EXTRAS tamm_pybind.cpp)
target_compile_features(pytamm PRIVATE cxx_std_17)
target_link_libraries(pytamm PRIVATE tamm Python3::Module MPI::MPI_CXX)

if(TARGET GlobalArrays::ga)
  target_link_libraries(pytamm PRIVATE GlobalArrays::ga)
endif()
if(TARGET hptt::hptt)
  target_link_libraries(pytamm PRIVATE hptt::hptt)
endif()
if(TARGET BLAS::BLAS)
  target_link_libraries(pytamm PRIVATE BLAS::BLAS)
endif()
if(TARGET LAPACK::LAPACK)
  target_link_libraries(pytamm PRIVATE LAPACK::LAPACK)
endif()
if(TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(pytamm PRIVATE OpenMP::OpenMP_CXX)
endif()

# CMakeBuild installs bundled dependencies under:
#   ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_PREFIX}/include
set(PYBIND11_STAGE_INCLUDE "${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_PREFIX}/include")
target_include_directories(pytamm PRIVATE ${PYBIND11_STAGE_INCLUDE})
target_include_directories(pytamm PRIVATE
  ${CMAKE_INSTALL_PREFIX}/include
  ${CMAKE_INSTALL_PREFIX}/include/eigen3
  ${PYBIND11_STAGE_INCLUDE}/eigen3
)
target_link_directories(pytamm PRIVATE
  ${CMAKE_INSTALL_PREFIX}/lib
  ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_PREFIX}/lib
)

set_target_properties(pytamm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

install(TARGETS pytamm LIBRARY DESTINATION python)
install(FILES
  README.md
  tensor_contraction.py
  DESTINATION python
)
