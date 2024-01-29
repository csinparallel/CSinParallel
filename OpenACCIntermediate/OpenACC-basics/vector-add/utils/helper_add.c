/*
 * Functions for setting up arrays and error checking
 */
 #include <stdio.h>
 #include <math.h>
 #include "helper_add.h"

// To reset the arrays for each trial
void initialize(float *x, float *y, int N) {
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

// showVec prints the vector for debugging
void showVec(float *vector, int n) {
  for (int i = 0; i < n; i++) {
    printf("%f  ", vector[i]);
    if ((i+1)%10 == 0) printf("\n");
  }
  printf("\n");
}

// check whether the kernel functions worked as expected
void checkForErrors(float *y, int N) {
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  float diff = 0.0f;
  for (int i = 0; i < N; i++) {
    // maxError = fmaxf(maxError, fabs(y[i]-3.0f));
    diff = fabs(y[i]-3.0f);
    maxError = diff > maxError ? diff : maxError;
  }
  
  printf("Max error: %f\n", maxError);
}

