cmake_minimum_required(VERSION 3.8)
project(prime_finder LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -pthread -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNUM_THREADS=8")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")

set(CXX_SOURCE_FILES Main.cpp Prime.cpp PrimePosix.cpp PrimeCpp11.cpp PrimeOmp.cpp PrimeCuda.cpp)
set(CUDA_SOURCE_FILES PrimeCudaWrapper.cu)

add_executable(prime_finder ${CXX_SOURCE_FILES} ${CUDA_SOURCE_FILES})
add_executable(prime_finder_mpi PrimeMpi.cpp Prime.cpp)

target_link_libraries(prime_finder m)
target_link_libraries(prime_finder_mpi m ${MPI_LIBRARIES})

set_target_properties(prime_finder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
