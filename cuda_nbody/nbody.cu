#include <iostream>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <random>

using namespace std;

// Returns the square of the distance between two particles
__device__ float distance2(const float3 p1, const float3 p2)
{
  float3 r;
  r.x = p2.x - p1.x;
  r.y = p2.y - p1.y;
  r.z = p2.z - p1.z;

  return r.x*r.x + r.y*r.y + r.z*r.z;
}

// Calculate the force between two particles. It's repulsive when too close and
// attractive when far
__device__ float3 force(const float3 from, const float3 on, float coefficient)
{
  float d2 = distance2(from, on);
  float3 f;

  if(d2 < 100)
  {
    // Repel
    f.x = -coefficient * (from.x - on.x) / d2;
    f.y = -coefficient * (from.y - on.y) / d2;
    f.z = -coefficient * (from.z - on.z) / d2;
  }
  else if(d2 > 101)
  {
    // Attract
    f.x = coefficient * (from.x - on.x) / d2;
    f.y = coefficient * (from.y - on.y) / d2;
    f.z = coefficient * (from.z - on.z) / d2;
  }
  else
  {
    // No force
    f.x = 0;
    f.y = 0;
    f.z = 0;
  }

  return f;
}

// Perform an iteration of the simulation
__global__ void iterate(const int num_particles, const int num_iterations, const float coefficient, const float damping, float3 *particles, float3 *velocities)
{
  // Use shared memory. There is only one "extern".
  // You need to divide it up yourself
  extern __shared__ float3 localparticles[];

  float3 my_position;
  float3 my_acceleration;

  // Get the index in the shared data
  int sid = threadIdx.x;

  // Get the index in the global data
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Do a few iterations before returning to the host
  for(int j = 0; j < num_iterations; j++)
  {
    my_position = particles[gid];
    my_acceleration = {0, 0, 0};

    // Copy the global memory to the shared memory of the block
    int num_blocks = (num_particles + blockDim.x - 1) / blockDim.x;
    for(int i = 0; i < num_blocks; i++)
    {
      // Load the i-th tile into shared memory for all the blocks
      localparticles[sid] = particles[sid + i*blockDim.x];
      
      // Wait until all threads have copied the data
      __syncthreads();

      // Calculate the force on my particle from all the other particles in the i-th block
      for(int j = 0; j < blockDim.x; j++)
      {
        // Skip my particle
        if(i*blockDim.x + j == gid)
          continue;

        float3 a = force(localparticles[j], my_position, coefficient);
        my_acceleration.x += a.x;
        my_acceleration.y += a.y;
        my_acceleration.z += a.z;
      }

      // Wait until all threads have calculated my acceleration
      __syncthreads();

    }
    
    // Now all blocks have been accounted for and my acceleration due to all of the
    // other particles is now known.
      __syncthreads();
    velocities[gid].x = (1 - damping) * (velocities[gid].x + my_acceleration.x);
    velocities[gid].y = (1 - damping) * (velocities[gid].y + my_acceleration.y);
    velocities[gid].z = (1 - damping) * (velocities[gid].z + my_acceleration.z);
    particles[gid].x += velocities[gid].x;
    particles[gid].y += velocities[gid].y;
    particles[gid].z += velocities[gid].z;
  }
}

void writeParticles(float3 *particles, float3 *velocities, int num_particles, int iteration_number)
{
  ofstream out("particles_" + to_string(iteration_number) + ".csv");
  out << "x,y,z,vx,vy,vz\n";
  for(int i = 0; i < num_particles; i++)
  {
    out << particles[i].x << "," << particles[i].y << "," << particles[i].z << ",";
    out << velocities[i].x << "," << velocities[i].y << "," << velocities[i].z << "\n";
  }
  out.close();
}

int main(void)
{
  const int num_particles = 2048;
  const float coefficient = 0.001;
  const float damping = 0.01;
  const int num_iterations = 16000;
  const int iterations_per_write = 20;
  float3 *particles;
  float3 *velocities;
  
  // Allocate Unified Memory
  // This is automatically synced between host and GPU
  cudaError_t err = cudaMallocManaged(&particles, num_particles*sizeof(float3));
  if(err != cudaSuccess) {
    cerr << "Failed to allocated memory on GPU: " << cudaGetErrorString(err) << endl;
    return 1;
  }
  err = cudaMallocManaged(&velocities, num_particles*sizeof(float3));
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
  for(int i = 0; i < num_particles; i++) {
    velocities[i].x = 0.;
    velocities[i].y = 0.;
    velocities[i].z = 0.;
  }
  
  cudaDeviceSynchronize();
  cout << "Data written" << endl;

  int blockSize;
  err = cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxBlockDimX, 0);
  if(err != cudaSuccess) {
    cerr << "Failed to query GPU: " << cudaGetErrorString(err) << endl;
    return 1;
  }
  
  // Run kernel on the GPU
//  int blockSize = 512;
  int numBlocks = (num_particles + blockSize - 1) / blockSize;
  int sharedMemSize = (num_particles / numBlocks) * sizeof(float3);
  cout << "Requesting " << sharedMemSize << " bytes" << endl;
  cout << "blockSize: " << blockSize << endl;
  cout << "numBlocks: " << numBlocks << endl;
  
  writeParticles(particles, velocities, num_particles, 0);
  for(int i = 1; i <= num_iterations / iterations_per_write; i++)
  {
    cout << "Iteration " << (i * iterations_per_write) << endl;
    iterate<<<numBlocks, blockSize, sharedMemSize>>>(num_particles, iterations_per_write, coefficient, damping, particles, velocities);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    writeParticles(particles, velocities, num_particles, i);
  }
  cout << "Done" << endl;

  // Free memory
  cudaFree(particles);
  cudaFree(velocities);

  return 0;
} 

