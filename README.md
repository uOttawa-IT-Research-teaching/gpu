# Code pour le séminaire de programmation GPU
Ce sont les trois exemples de programmes utilisés pendant le séminaire.

* cuda_add
  * Programme très basique pour montrer à quoi ressemble un programme minimal utilisant CUDA.
* opencl_add
  * Programme très basique pour montrer à quoi ressemble un programme minimal utilisant OpenCL.
* cuda_nbody
  * Démonstration de l'utilisation des différentes mémoires du GPU, et notamment de la mémoire partagée.

Tous les programmes d'exemple sont accompagnés d'un CMakeLists.txt pour une compilation facile avec cmake. Pour compiler, il faut simplement exécuter :

    mkdir -p build
    cd build
    cmake ..
    make

Pour les programmes CUDA, vous avez besoin d'un GPU NVIDIA. OpenCL fonctionne sur n'importe quoi, si vous avez installé les pilotes.

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

For the CUDA programs you need a NVIDIA GPU. OpenCL runs on anything as long as you have the drivers installed.
