cmake_minimum_required(VERSION 3.5)
project(prime_finder)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE_FILES main.cpp Prime.cpp PrimePosix.cpp PrimeCpp11.cpp)

add_executable(prime_finder ${SOURCE_FILES})

target_link_libraries(prime_finder m)
