/*
 * Functions for setting up arrays and error checking
 */

// To reset the arrays for each trial
void initialize(float *x, float *y, int N);

// showVec prints the vector for debugging
void showVec(float *vector, int n);

// check whether the kernel functions worked as expected
void checkForErrors(float *y, int N);
