cmake_minimum_required(VERSION 3.8)
project(prime_finder)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE_FILES main.cpp single_thread.cpp posix_threads.cpp)
add_executable(prime_finder ${SOURCE_FILES})