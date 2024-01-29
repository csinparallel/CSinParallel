#include <math.h>   // fmaxf function
#include <stdio.h>  // printf
#include <stdlib.h> // malloc
#include <omp.h>    // openMP (just for timing)

// helper functions and utilities 
#include "../utils/helper_add.h"
// command line arguments
#include "../utils/getCommandLine.h"

#define PRINT_LIMIT 40

// CPU version of add with acc pragma
// Note the use of num_gangs() to indicate the number of threads
// can slow down the code.
void CPUadd(int n, float *x, float *y, int numThreads)
{
  #pragma acc parallel loop num_gangs(numThreads)      // A. comment
  // #pragma acc parallel loop                    // B. uncomment, numThreads setting ignored
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
    
  }
}

int main(int argc, char **argv)
{
  printf("Vector addition using acc pragma pgcc compiler.\n");
  // Set up size of arrays for vectors
  // large N
  int N = 1024*1024*1024; 

  int numThreads =1;
  
  // get command line args to change size of array
  getArguments(argc, argv, &N, &numThreads);

  printf("size (N) of 1D arrays are: %d\n\n", N);

  // host vectors
  float *x, *y;

   // Size, in bytes, of each vector
  size_t bytes = N*sizeof(float);

  // Allocate memory for each vector on host
  x = (float*)malloc(bytes);
  y = (float*)malloc(bytes);

  // initialize x and y arrays on the host
  initialize(x, y, N);  // set values in each vector

  if (N <= PRINT_LIMIT) {   // debug
    printf("x:\n");
    showVec(x, N);
    printf("y:\n");
    showVec(y, N);
  }

  printf("add vectors on host using %d threads\n", numThreads);
  
  double start, end;
  start = omp_get_wtime();
  CPUadd(N, x, y, numThreads);
  end = omp_get_wtime();
  printf("Time: %lf secs\n", end - start);

  if (N <= PRINT_LIMIT) {   // debug
    printf("y result:\n");
    showVec(y, N);
  }

  checkForErrors(y, N);

  printf("execution complete\n");

  // Release host memory
  free(x);
  free(y);

  return 0;
}
