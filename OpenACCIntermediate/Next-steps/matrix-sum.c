/*
 * OpenACC GPU version of matrix summation operation.
 * Demonstrates collapse and reduction clauses.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // just for timing


// function declarations
void fillMatrix(int size, float * A);
void getArguments(int argc, char **argv, int *size, int *verbose, int *check);
void debugPrintMatrix(int verbose, int size, float *matrix, const char *msg);
void showMatrix(int size, float * matrix);
void checkForErrors(float *y, int N,  float sum);


// device function:
// Update matrix A values by doing some math
//
float MatrixSum(int size, float * __restrict__ A) {

   float sum = 0.0;
   
   #pragma acc kernels
   #pragma acc loop collapse(2) independent reduction(+:sum)
   for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
         // do some contrived calculations that add up to 1.0 in each cell
         // when each data element is equal to Pi.
         A[i*size + j] = hypot(cos(A[i*size + j]), sin(A[i*size + j]));
         sum += A[i*size + j];
      }
   }
   return sum;
}

////////////////////////////////////////////////////////// main
int main (int argc, char **argv) {
 
   // default values
   int size = 256;          // num rows, cols of square matrix
   int verbose = 0;         // default to not printing matrices
   int check = 0;           // check for errors if >0
   getArguments(argc, argv, &size, &verbose, &check); //change defaults

   float * A;  // matrix to fill and perform calculations on

   // Use a 'flattened' 1D array of contiguous memory for the matrix
   // size = number of rows = number of columns in the square matrix
   size_t num_elements = size * size * sizeof(float);
   A = (float *)malloc(num_elements);

   fillMatrix(size, A);
   
   char msgA[32] = "matrix A after filling:";
   debugPrintMatrix(verbose, size, A, msgA);
   
   double startTime = omp_get_wtime();
   
   float total = MatrixSum(size, A);

   char msgC[32] = "matrix A after Matrixupdate(): ";
   debugPrintMatrix(verbose, size, A, msgC);
   printf("Sum of all values = %f\n", total);
   
   double endTime = omp_get_wtime();

   printf("\nTotal runtime %f seconds (%f milliseconds)\n", 
   (endTime-startTime), (endTime-startTime)*1000);

   if (check) {
      checkForErrors(A, size, total);
   }

   free(A); 
   return 0;
}
////////////////////////////////////// end main

// fill a given square matrix with rows of float values 
// equal to Pi
void fillMatrix(int size, float * A) {
   for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        A[i*size + j] = ((float)M_PI);
      }
   }
}

void getArguments(int argc, char **argv, int *size, int *verbose, int *check) {
   // 3 arguments optional: 
   //   size of one side of square matrix
   //   verbose printing for debugging
   //   whether to check for correct result
   if (argc > 4) {
      fprintf(stderr,"Use: %s [size] [verbose] [check] \n", argv[0]);
      exit(EXIT_FAILURE);
   }

   if (argc >= 2) {
      *size = atoi(argv[1]);
      if (argc >= 3) {
         *verbose = atoi(argv[2]);
      }
      if (argc == 4) {
         *check = atoi(argv[3]);
      }
   }
   
   if (*verbose) {
      printf("size of matrix side: %d\n", *size);
   }
}

void debugPrintMatrix(int verbose, int size, float *matrix, const char *msg) {
   if (verbose){
      printf("%s \n", msg);
      showMatrix(size, matrix);
   }
}

// display a given square matrix for debugging purposes
void showMatrix(int size, float * matrix) {
   int i, j;
   for (i=0; i<size; i++){
      for (j=0; j<size; j++) {
         printf("element [%d][%d] = %f \n",i,j, matrix[i*size + j]);
      }
   }
}

// check whether the kernel functions worked as expected
void checkForErrors(float *y, int size, float sum) {
  // Check for errors (all values should be 1.0f)
  float maxError = 0.0f;
  for (int i=0; i<size; i++){
      for (int j=0; j<size; j++) {
         maxError = fmaxf(maxError, fabs(y[i]-1.0f));
      }
  }
  printf("Max error in any data element: %f\n", maxError);
  float estSum = (float)(size * size);
  maxError = estSum - sum;
  printf("Sum is off by: %f\n", maxError);

}

