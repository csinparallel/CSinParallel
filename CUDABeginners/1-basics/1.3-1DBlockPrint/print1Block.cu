// System includes
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>

// Given a 1 dimensional grid of blocks of threads, 
// determine my thread number.
// This is run on the GPU on each thread.
__device__ int find1DThreadNumber() {
  // illustrative variable names
  int threadsPerBlock_horizontal = blockDim.x;
  int gridBlockNumber = blockIdx.x;

  int threadNumber = (gridBlockNumber * threadsPerBlock_horizontal) + threadIdx.x;
  return threadNumber;
}

// Print information about a thread running this function.
// This is run on the GPU on each thread.
__global__ void hellofromDevice1D(int val) {

  int threadNumber = find1DThreadNumber();   // device function call
  printf("[b%d of %d, t%d]:\tValue sent to kernel function is:%d\n",     
             blockIdx.x, gridDim.x, 
             threadNumber, val);   
}

int main(int argc, char **argv) {

  //////////////////////////////////////////////////////////////
  //    Each block that you specify maps to an SM.
  //////////////////////////////////////////////////////////////

  printf("1D grid of blocks\n");
  // 1 block of 16 threads goes to 1 SM 
  dim3 gridDim1(1), blockDim1(16);        // 1 block, 16 threads
  
  hellofromDevice1D<<<gridDim1, blockDim1>>>(1);

  cudaDeviceSynchronize();         // comment out and re-make and run

  // TODO: uncomment 2 code lines below and try again.
  // For simple cases like this, some developers bypass the use of dim3
  // variable and make calls like this:

  // hellofromDevice1D<<<1, 16>>>(2);  
  // cudaDeviceSynchronize(); 

  return 0;

}
