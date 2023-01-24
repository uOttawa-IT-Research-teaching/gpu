# Code for the GPU programming seminar

These are the three example programs used during the seminar.

* cuda_add
  * Very basic program to show what a minimal CUDA-enabled program looks like
* opencl_add
  * Very basic program to show what a minimal OpenCL-enabled program looks like
* cuda_nbody
  * Demonstration on how to use the various memories on the GPU, including shared memory

All example programs come with a CMakeLists.txt for easy compilation with cmake. To compile, simply run:

    mkdir -p build
    cd build
    cmake ..
    make

For the CUDA programs you need a NVIDIA GPU, OpenCL runs on anything as long as you have the drivers installled.
