//
// Demonstration using a single 1D grid and 1D block size
//
/*
 * Example of vector addition :
 * Array of floats x is added to array of floats y and the 
 * result is placed back in y
 *
 */
#include <math.h>   // ceil function
#include <stdio.h>  // printf
#include <iostream> // alternative cout print for illustration

#include <cuda.h>

void initialize(float *x, float *y, int N);
void verifyCorrect(float *y, int N);
void getArguments(int argc, char **argv, int *blockSize);

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
__global__ void vecAdd(float *x, float *y, int n)
{
    // Get our global thread ID
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        y[id] = x[id] + y[id];
}

////////////////////                   main
int main(int argc, char **argv)
{
  printf("Vector addition using managed memory.\n");
  // Set up size of arrays for vectors 
  int N = 32*1048576;
  // TODO: try changng the size of the arrays by doubling or
  //       halving (32 becomes 64 or 16). Note how the grid size changes.
  printf("size (N) of 1D arrays are: %d\n\n", N);
  
  // host vectors, which are arrays of length N
  float *x, *y;

  // Size, in bytes, of each vector; just use below
  size_t bytes = N*sizeof(float);

  // 1.1 Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, bytes);
  cudaMallocManaged(&y, bytes);
  cudaCheckErrors("allocate managed memory");

  // 1.2 initialize x and y arrays on the host
  initialize(x, y, N);  // set values in each vector

  // Number of threads in each thread block
  int blockSize = 256;
  getArguments(argc, argv, &blockSize); //update blocksize from cmd line
 
  // Number of thread blocks in grid needs to be based on array size
  int gridSize = (int)ceil((float)N/blockSize);
 
  printf("add vectors on device using grid with ");
  printf("%d blocks of %d threads each.\n", gridSize, blockSize);
  // 2. Execute the kernel
  vecAdd<<<gridSize, blockSize>>>(x, y, N);
  cudaCheckErrors("vecAdd kernel call");

  // 3. Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");

  // 4. Check that the computation ran correctly
  verifyCorrect(y, N);

  printf("execution complete\n");

  // 5. free unified memory
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
    maxError = fmax(maxError, fabsf(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
}

// simple argument gather for this simple example program
void getArguments(int argc, char **argv, int *blockSize) {

  if (argc == 2) {
    *blockSize = atoi(argv[1]);
  }

}
