cmake_minimum_required(VERSION 3.5)
project(prime_finder)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -pthread -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNUM_THREADS=4")
set(MPIEXEC_MAX_NUMPROCS "4")
set(SOURCE_FILES main.cpp Prime.cpp PrimePosix.cpp PrimeCpp11.cpp PrimeOmp.cpp PrimeMpi.cpp)

add_executable(prime_finder ${SOURCE_FILES})

target_link_libraries(prime_finder m ${MPI_LIBRARIES})
