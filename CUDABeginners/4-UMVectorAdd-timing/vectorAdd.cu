//
// Demonstration using a single 1D grid and 
// 0, 1, or more 1D blocks of optional size
//
/*
 * Example of vector addition :
 * Array of floats x is added to array of floats y and the 
 * result is placed back in y
 *
 * Timng added for analysis of block size differences.
 */

#include <math.h>
#include <iostream> // alternative cout print for illustration
#include <time.h>
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


// To run code on host for comparison
void HostAdd(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

// Kernel function to add the elements of two arrays
// This one is sequential on one GPU core.
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

// Parallel version that uses threads in the block.
//
//  If block size is 8, e.g.
//    thread 0 works on index 0, 8, 16, 24, etc. of each array
//    thread 1 works on index 1, 9, 17, 25, etc.
//    thread 2 works on index 2, 10, 18, 26, etc.
//
// This is mapping a 1D block of threads onto these 1D arrays.
__global__
void add_parallel_1block(int n, float *x, float *y)
{
  int index = threadIdx.x;    // which thread am I in the block?
  int stride = blockDim.x;    // threads per block
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

// In this version, thread number is its block number 
// in the grid (blockIdx.x) times 
// the threads per block plus which thread it is in that block.
//
// Then the 'stride' to the next element in the array goes forward
// by multiplying threads per block (blockDim.x) times 
// the number of blocks in the grid (gridDim.x).

__global__
void add_parallel_nblocks(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

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
/// temp test


int main(int argc, char **argv)
{
  printf("This program lets us experiment with the number of blocks and the \n");
  printf("number of threads per block to see its effect on running time.\n");
  printf("\nUsage:\n");
  printf("%s [num threads per block] [array_size]\n\n", argv[0]);
  printf("\nwhere you can specify only the number of threads per block \n");
  printf("and the number of blocks will be calculated based on the size\n");
  printf("of the array.\n\n");

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

  clock_t t_start, t_end;              // for timing
  double tot_time_secs;
  double tot_time_milliseconds;
  
  ///////////////////////////////////////////////////////////////////
  // case 1: run on the host on one core
  t_start = clock();
  // sequentially on the host
  HostAdd(N, x, y);
  t_end = clock();
  tot_time_secs = ((double)(t_end-t_start)) / CLOCKS_PER_SEC;
  tot_time_milliseconds = tot_time_secs*1000;
  printf("\nSequential time on host: %f seconds (%f milliseconds)\n", tot_time_secs, tot_time_milliseconds);
 
  verifyCorrect(y, N);
  
  ///////////////////////////////////////////////////////////////////
  // case 2:
  // Purely illustration of something you do not ordinarilly do:
  // Run kernel on all elements on the GPU sequentially on one thread
  
  // re-initialize
   initialize(x, y, N);

  t_start = clock();

  add<<<1, 1>>>(N, x, y);   // the kernel call
  cudaCheckErrors("add kernel call");

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");
  
  t_end = clock();
  tot_time_secs = ((double)(t_end-t_start)) / CLOCKS_PER_SEC;
  tot_time_milliseconds = tot_time_secs*1000;
  printf("\nSequential time on one device thread: %f seconds (%f milliseconds)\n", tot_time_secs, tot_time_milliseconds);
 
  verifyCorrect(y, N);

  ///////////////////////////////////////////////////////////////////
  // case 3: using a single block of 256 threads
  // re-initialize x and y arrays on the host
  initialize(x, y, N);

  // Use the GPU in parallel with one block of threads.
  // Essentially using one SM for the block.

  t_start = clock();

  add_parallel_1block<<<1, 256>>>(N, x, y);   // the kernel call
  cudaCheckErrors("add_parallel_1block kernel call");

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");

  t_end = clock();
  tot_time_secs = ((double)(t_end-t_start)) / CLOCKS_PER_SEC;
  tot_time_milliseconds = tot_time_secs*1000;

  printf("\nParallel time on 1 block of 256 threads: %f milliseconds\n", tot_time_milliseconds);
  
  verifyCorrect(y, N);

  ///////////////////////////////////////////////////////////////////
  // Case 4:
  // Now use multiple blocks so that we use more than one SM.
  // Also use a slightly different 'stride' pattern for which 
  // thread works on which element.
    
  // re-initialize x and y arrays on the host
  initialize(x, y, N);

  // Number of thread blocks in grid could be fixed 
  // and smaller than maximum needed.
  int gridSize = 16;

  printf("\n----------- number of %d-thread blocks: %d\n", blockSize, gridSize);

  t_start = clock();
  // the kernel call assuming a fixed grid size and using a stride
  add_parallel_nblocks<<<gridSize, blockSize>>>(N, x, y);
  cudaCheckErrors("add_parallel_nblocks kernel call");
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");

  t_end = clock();
  tot_time_secs = ((double)(t_end-t_start)) / CLOCKS_PER_SEC;
  tot_time_milliseconds = tot_time_secs*1000;
  printf("Stride loop pattern: \n");
  printf("Parallel time on %d blocks of %d threads = %f milliseconds\n", gridSize, blockSize, tot_time_milliseconds);

  verifyCorrect(y, N);

  //////////////////////////////////////////////////////////////////
  // case 5: withot using stride
  //
  // re-initialize x and y arrays on the host
  initialize(x, y, N);

  // set grid size based on array size and block size
  gridSize = ((int)ceil((float)N/blockSize));

  printf("\n----------- number of %d-thread blocks: %d\n", blockSize, gridSize);

  t_start = clock();
  // the kernel call
  vecAdd<<<gridSize, blockSize>>>(x, y, N);
  cudaCheckErrors("vecAdd kernel call");

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");

  t_end = clock();
  tot_time_secs = ((double)(t_end-t_start)) / CLOCKS_PER_SEC;
  tot_time_milliseconds = tot_time_secs*1000;
  printf("No stride loop pattern: \n");
  printf("Parallel time on %d blocks of %d threads = %f milliseconds\n", gridSize, blockSize, tot_time_milliseconds);

  verifyCorrect(y, N);

  ///////////////////////////// end of tests

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaCheckErrors("free cuda memory");
  
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
    maxError = fmax(maxError, fabs(y[i]- 3.0f));
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
