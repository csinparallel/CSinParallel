#include <math.h>   // ceil function
#include <stdio.h>  // printf
#include <stdlib.h> // malloc
#include <omp.h>    // openMP

// helper functions and utilities 
#include "../utils/helper_add.h"
// command line arguments
#include "../utils/getCommandLine.h"

// CPU version of add sequentially
void CPUadd(int n, float *x, float *y)
{
  #pragma omp parallel for shared(n, x, y)
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
    // debug
    if (n < 40) {
      printf("thread %d loop iteration %d\n", omp_get_thread_num(), i);
    }
  }
}

int main(int argc, char **argv)
{
  printf("Vector addition using openMP pragma with gcc or pgcc compiler.\n");
  // Set up size of arrays for vectors
  // large N
  int N = 1024*1024*1024; 

  int numThreads =1;
  
  // get command line args to change size of array
  getArguments(argc, argv, &N, &numThreads);

  //set the number of threads to use
  omp_set_num_threads(numThreads);

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

  if (N < 40) {   // debug
    printf("x:\n");
    showVec(x, N);
    printf("y:\n");
    showVec(y, N);
  }

  printf("add vectors on host using %d threads\n", numThreads);
  
  double start, end;
  start = omp_get_wtime();
  CPUadd(N, x, y);
  end = omp_get_wtime();
  printf("Time: %lf secs\n", end - start);

  if (N < 40) {   // debug
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
