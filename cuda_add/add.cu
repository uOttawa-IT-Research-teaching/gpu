#include <iostream>
#include <math.h>

using namespace std;

// Kernel function to add the elements of two arrays
// The __global__ means kernel function
__global__ void add(int n, const float *A, const float *B, float *C)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
    C[i] = A[i] + B[i];
}

int main(void)
{
  const int num_numbers = 40000;
  float *A, *B, *C;

  // Allocate Unified Memory  accessible from CPU or GPU
  cudaError_t err = cudaMallocManaged(&A, num_numbers*sizeof(float));
  if(err != cudaSuccess) {
    cerr << "Failed to allocated memory on GPU: " << cudaGetErrorString(err) << endl;
    return 1;
  }
  err = cudaMallocManaged(&B, num_numbers*sizeof(float));
  if(err != cudaSuccess) {
    cerr << "Failed to allocated memory on GPU: " << cudaGetErrorString(err) << endl;
    return 1;
  }
  err = cudaMallocManaged(&C, num_numbers*sizeof(float));
  if(err != cudaSuccess) {
    cerr << "Failed to allocated memory on GPU: " << cudaGetErrorString(err) << endl;
    return 1;
  }

  // Initialize the arrays
  for(int i = 0; i < num_numbers; i++) {
    A[i] = 0.25*i;
    B[i] = 0.75*i;
  }

  // Run kernel on the GPU
  add<<<1, 512>>>(num_numbers, A, B, C);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Show the result
  cout << "Result: ";
  for(int i = 0; i < num_numbers; i++)
    cout << C[i] << " ";
  cout << endl;

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return 0;
} 
