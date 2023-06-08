/*
 * Showing how to do one version of vector addition using 
 * CUDA timing functions.

 * Array of floats x is added to array of floats y and the 
 * result is placed back in y using unified memory.
 *
 */

#include <math.h>
#include <iostream> // alternative cout print for illustration

#include <cuda.h>

void initialize(float *x, float *y, int N);
void verifyCorrect(float *y, int N);
void getArguments(int argc, char **argv, int *blockSize, int *numElements);

///////
// error checking macro taken from Oakridge Nat'l lab training code:
// https://github.com/olcf/cuda-training-series
////////
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// Kernel function based on 1D grid of 1D blocks of threads
// In this version, thread number is:
//  its block number in the grid (blockIdx.x) times 
// the threads per block plus which thread it is in that block.
//
// This thread id is then the index into the 1D array of floats.
// This represents the simplest type of mapping:
// Each thread takes care of one element of the result
//
// For this to work, the number of blocks specified 
// times the specified threads per block must
// be the same or greater than the size of the array.
__global__ 
void vecAdd(float *x, float *y, int n)
{
    // Get our global thread ID
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        y[id] = x[id] + y[id];
}


int main(int argc, char **argv)
{
  printf("This program lets us experiment with \n");
  printf("number of threads per block to see its effect on running time.\n");
  printf("\nUsage:\n");
  printf("%s [num threads per block] [array_size]\n\n", argv[0]);
  printf("\nwhere you can specify the number of threads per block \n");
  printf("and the number of blocks will be calculated based on the size\n");
  printf("of the array.\n\n");

  // for timing using CUDA functions
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0.0;

  // Set up size of arrays
  // multiple of 1024 to match largest threads per block 
  // allowed in many NVIDIA GPUs
  //
  int N = 32*1048576; 
  int blockSize = 256;     // threads per block
  float *x, *y;

  // determine largest number of threads per block allowed
  int devId;            // the number assigned to the GPU
  int maxThreadsPerBlock;  // maximum threads available per block
  cudaGetDevice(&devId);
  cudaDeviceGetAttribute(&maxThreadsPerBlock, 
    cudaDevAttrMaxThreadsPerBlock, devId);

  // get optional arguments 
  getArguments(argc, argv, &blockSize, &N); 

  //change if requested block size is too large
  if (blockSize > maxThreadsPerBlock) {
    blockSize = maxThreadsPerBlock;
    printf("WARNING: using %d threads per block, which is the maximum.", blockSize);
  }

  printf("size (N) of 1D array is: %d\n\n", N);
 // Size, in bytes, of each vector; just use below
  size_t bytes = N*sizeof(float);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, bytes);
  cudaMallocManaged(&y, bytes);
  cudaCheckErrors("allocate managed memory");

  // initialize x and y arrays on the host
  initialize(x, y, N);

  int numBlocks = 0;   // signifies we should calculate numBlocks
  
  // Knowing the length of our array, determine the number of 
  // blocks threads we need to map each array element to a
  // thread in a block.
  if (numBlocks == 0) {   //signifies we should calculate numBlocks
    //numBlocks = (N + blockSize - 1) / blockSize;
    numBlocks = (int)ceil((float)N/blockSize);
  }

  
  printf("\n----------- number of %d-thread blocks: %d\n", blockSize, numBlocks);

  cudaEventRecord(start);
     // the kernel call
  vecAdd<<<numBlocks, blockSize>>>(x, y, N);
  cudaCheckErrors("add kernel call");

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");

  cudaEventRecord(stop);
  milliseconds = 0.0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaCheckErrors("timer stop");

  printf("no stride Parallel time on %d blocks of %d threads: %f milliseconds\n", numBlocks, blockSize, milliseconds);

  verifyCorrect(y, N);

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaCheckErrors("free unified memory");
  
  return 0;
}

// To reset the arrays for each trial
void initialize(float *x, float *y, int N) {
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

// check whether the kernel functions worked as expected
void verifyCorrect(float *y, int N) {
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
}

// simple argument gather for this simple 1D example program
//
// Design is the arguments will be optional in this order:
//  number of threads per block
//  number of  data elements in 1D vector arrays
void getArguments(int argc, char **argv, int *blockSize, int *numElements) {

  if (argc == 3) {   // both given
    *numElements = atoi(argv[2]);
    *blockSize = atoi(argv[1]);
  } else if (argc == 2) {   // just threads per block given
    *blockSize = atoi(argv[1]);
  }

}

