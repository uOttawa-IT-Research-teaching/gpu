#include <iostream>
#include <math.h>
#include <cuda.h>
#include <random>

using namespace std;

// Returns the square of the distance between two particles
__device__ float dist2(const float3 pos1, const float3 pos2)
{
  float3 r;
  r.x = pos2.x - pos1.x;
  r.y = pos2.x - pos1.y;
  r.z = pos2.x - pos1.z;
  
  return r.x*r.x + r.z*r.z + r.z*r.z;
}

// Perform an iteration of the simulation
__global__ void iterate(int n, float3 *particles)
{
  // Use shared memory
  extern __shared__ float3 localparticles[];

  // Get the index for the data point this thread works on
  int index = blockIdx.x * blockDim.x + threadIdx.x;;
  int stride = blockDim.x * gridDim.x;

  // Copy the global memory to the shared memory of the block
  for (int i = index; i < n; i += stride)
  {
    localparticles[i].x = particles[i].x;
    localparticles[i].y = particles[i].y;
    localparticles[i].z = particles[i].z;
  }

  // Wait until all threads have copied the data
  __syncthreads();

  // Do some physics
  for (int i = index; i < n; i += stride)
  {
    localparticles[i].z -= 0.1;
  }

  // Wait until all threads have finished the simulation
  __syncthreads();

  // Copy the shared memory back to the block
  for (int i = index; i < n; i += stride)
  {
    particles[i].x = localparticles[i].x;
    particles[i].y = localparticles[i].y;
    particles[i].z = localparticles[i].z;
  }
}

int main(void)
{
  const int num_particles = 512;
  float3 *particles;
  
  // Allocate Unified Memory  accessible from CPU or GPU
  cudaError_t err = cudaMallocManaged(&particles, num_particles*sizeof(float3));
  if(err != cudaSuccess) {
    cerr << "Failed to allocated memory on GPU: " << cudaGetErrorString(err) << endl;
    return 1;
  }

  std::default_random_engine generator;
  std::uniform_real_distribution<double> dist(-10, 10);

  // Initialize the arrays
  for(int i = 0; i < num_particles; i++) {
    particles[i].x = dist(generator);
    particles[i].y = dist(generator);
    particles[i].z = dist(generator);
  }

  // Run kernel on the GPU
  int blockSize = 256;
  int numBlocks = (num_particles + blockSize - 1) / blockSize;
  int sharedMemSize = (num_particles / numBlocks) * sizeof(float3);
  iterate<<<numBlocks, blockSize, sharedMemSize>>>(num_particles, particles);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Show the result
  cout << "Result:\n";
  for(int i = 0; i < num_particles; i++)
    cout << particles[i].x << "\t" << particles[i].y << "\t" << particles[i].z << "\n";
  cout << endl;

  // Free memory
  cudaFree(particles);
  
  return 0;
} 

