cmake_minimum_required(VERSION 3.1)

project(cuda_nbody)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_executable(cuda_nbody nbody.cu)

