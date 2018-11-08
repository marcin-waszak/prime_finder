cmake_minimum_required(VERSION 3.8)
project(prime_finder)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp single_thread.cpp)
add_executable(prime_finder ${SOURCE_FILES})