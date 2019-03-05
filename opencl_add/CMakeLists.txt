cmake_minimum_required(VERSION 3.1)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(opencl_add add.cpp)
set_property(TARGET opencl_add PROPERTY CXX_STANDARD 11)
target_link_libraries(opencl_add ${OpenCL_LIBRARIES})
