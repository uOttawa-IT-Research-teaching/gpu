cmake_minimum_required(VERSION 3.1)

project(cuda_add)

find_package(CUDA REQUIRED)

cuda_add_executable(cuda_add add.cu)
