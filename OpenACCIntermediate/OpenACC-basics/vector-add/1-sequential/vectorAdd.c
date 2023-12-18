#include <math.h>   // ceil function
#include <stdio.h>  // printf
#include <stdlib.h> // malloc

// helper functions and utilities 
#include "../utils/helper_add.h"
// command line arguments
#include "../utils/getCommandLine.h"

// CPU version of add sequentially
void CPUadd(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

int main(int argc, char **argv)
{
  printf("Vector addition using several compilers.\n");
  // Set up size of arrays for vectors
  // int N = 1<<20;
  // same value, shown as multiple of 1024
  int N = 1024*1024; 

  int numThreads =1;
  
  // get command line args to change size of array
  getArguments(argc, argv, &N, &numThreads);

  // ignore numThreads for sequential case
  if (numThreads != 1) {
      numThreads = 1;     // not used below
      printf("Warning: this is a squential version and the number of threads is always 1, even though you used -t\n");
  }

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

  printf("add vectors on host\n");
  
  CPUadd(N, x, y);

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
