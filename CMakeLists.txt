cmake_minimum_required(VERSION 3.20)
project(mpi_cuda_coursework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(mpi_cuda_coursework
  src/main.cpp
  src/mpi_distribution.cpp
  src/cpu_reference.cpp
  src/benchmarks.cpp
  src/cuda_kernels.cu
)

target_include_directories(mpi_cuda_coursework PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(mpi_cuda_coursework PRIVATE MPI::MPI_CXX)

set_target_properties(mpi_cuda_coursework PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "native"
)

target_compile_options(mpi_cuda_coursework PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(mpi_cuda_tests
  tests/test_main.cpp
  src/mpi_distribution.cpp
  src/cpu_reference.cpp
  src/cuda_kernels.cu
)

target_include_directories(mpi_cuda_tests PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(mpi_cuda_tests PRIVATE MPI::MPI_CXX)
set_target_properties(mpi_cuda_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "native"
)
