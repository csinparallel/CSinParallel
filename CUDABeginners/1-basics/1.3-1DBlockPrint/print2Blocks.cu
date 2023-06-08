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
  // 2 blocks of 8 threads each goes to 2 SMs 
  dim3 gridDim1(2), blockDim1(8);   
  // TODO: try 8 blocks of 8 threads each. What do you observe?    
  
  hellofromDevice1D<<<gridDim1, blockDim1>>>(1);

  cudaDeviceSynchronize();         // comment out and re-make and run
}
