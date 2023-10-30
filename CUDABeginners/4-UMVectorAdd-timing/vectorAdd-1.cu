/*
 * Example of vector addition :
 * Array of floats x is added to array of floats y and the 
 * result is placed back in y
 *
 * Timng added for analysis of CPU and GPU differences.
 * 
 * This is a simple example that is for demonstration only:
 * THIS IS NOT HOW WE NORMALLY WRITE AND RUN VECTOR ADDITION CODE.
 */

#include <math.h>
#include <iostream> // alternative cout print for illustration
#include <time.h>
#include <cuda.h>

void initialize(float *x, float *y, int N);
void verifyCorrect(float *y, int N);
void getArguments(int argc, char **argv, int *numElements);

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

/////////////////////////////////////  main  //////////////////////////////
int main(int argc, char **argv)
{
  printf("This program lets us experimentally see the difference in running\n");
  printf("time between a single host CPU core and a single device GPU core.\n");
  
  // Set up size of arrays
  // multiple of 1024 to match largest threads per block 
  // allowed in many NVIDIA GPUs
  //
  int N = 32*1048576;
  float *x, *y;

  // get optional argument: change array size
  getArguments(argc, argv, &N); 

  printf("size (N) of 1D array is: %d\n\n", N);

  // Size, in bytes, of each vector; use just below
  size_t bytes = N*sizeof(float);

  // Allocate Unified Memory - accessible from CPU or GPU
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
//  number of  data elements in 1D vector arrays
void getArguments(int argc, char **argv, int *numElements) {

  if (argc == 2) {  
    *numElements = atoi(argv[1]);
  }
}
