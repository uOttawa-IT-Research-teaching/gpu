#include <iostream>
#include <vector>
#include <CL/cl.hpp>

using namespace std;

static const char kernel_add[] =
  "void kernel add(global const float *A, global const float *B, global float *C)\n"
  "{\n"
  "  C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];\n"
  "}\n";

int main() {
  // Open the context
  cl_int err;
  cl::Context context(CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL, &err);
  if(err != CL_SUCCESS) {
    cerr << "Failed to open context" << endl;
    switch(err) {
      case CL_DEVICE_NOT_AVAILABLE:
        cerr << "  Device not available" << endl;
        break;
      case CL_DEVICE_NOT_FOUND:
        cerr << "  Device not found" << endl;
        break;
      case -1001:
        cerr << "  No platform available" << endl;
        break;
      default:
        cerr << "  " << err << endl;
        break;
    }
    return 1;
  }

  // Load and compile kernel
  cl::Program::Sources sources;
  sources.push_back({kernel_add, strlen(kernel_add)});
  cl::Program program(context, sources);
  if(program.build() != CL_SUCCESS) {
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    for(auto device : devices)
      cerr << "Failed to build:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
    return 1;
  }

  // Prepare the data
  const int num_numbers = 40000;

  cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float)*num_numbers);
  cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, sizeof(float)*num_numbers);
  cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, sizeof(float)*num_numbers);

  float *A = new float[num_numbers];
  float *B = new float[num_numbers];

  for(int i = 0; i < num_numbers; i++) {
    A[i] = 0.25*i;
    B[i] = 0.75*i;
  }

  // Create a queue for commands on the GPU
  cl::CommandQueue queue(context);

  // Copy the input data to the GPU
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float)*num_numbers, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float)*num_numbers, B);

  // It's safe to delete the local memory because it's on the GPU now and write buffer function is blocking
  delete[] A;
  delete[] B;

  // Run the kernel
  cl::Kernel kernel_add(program, "add");
  kernel_add.setArg(0, buffer_A);
  kernel_add.setArg(1, buffer_B);
  kernel_add.setArg(2, buffer_C);
  queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(num_numbers), cl::NullRange);

  // Copy the output data from the GPU
  float *C = new float[num_numbers];
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float)*num_numbers, C);

  // Show the result
  cout << "Result: ";
  for(int i = 0; i < num_numbers; i++)
    cout << C[i] << " ";
  cout << endl;
  
  delete[] C;

  return 0;
}

