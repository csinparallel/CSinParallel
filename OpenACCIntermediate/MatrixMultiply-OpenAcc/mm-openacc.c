/*
 * OpenACC GPU version of matrix multiplication.
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
void verfiyCorrect(int size, float *matrix);

// mutiply matrix A times matrix B, placing result in matrix C
void MatrixMult(int size, float * __restrict__ A, 
               float * __restrict__ B, float * __restrict__ C) {

#pragma acc data copyin(A,B) copy(C)
#pragma acc kernels
#pragma acc loop collapse(2) independent 
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float tmp = 0.;
      #pragma acc loop seq 
      for (int k = 0; k < size; ++k) {
        tmp += A[i*size + k] * B[k*size + i];
      }
      C[i*size + j] = tmp;    // update cell of C once
    }
  }
}

int main (int argc, char **argv) {
 
   // default values
   int size = 256;          // num rows, cols of square matrix
   int verbose = 0;         // default to not printing matrices
    int check = 0;           // check for errors if >0
   getArguments(argc, argv, &size, &verbose, &check); //change defaults

   printf("matrix rows, cols = %d\n", size);

   float * A;  // input matrix
   float * B;  // input matrix
   float * C;  // output matrix

// Use a 'flattened' 1D array of contiguous memory for the matrices
// size = number of rows = number of columns in the square matrices
   size_t num_elements = size * size * sizeof(float);
   A = (float *)malloc(num_elements);
   B = (float *)malloc(num_elements);
   C = (float *)malloc(num_elements);

   fillMatrix(size, A);
   fillMatrix(size, B);
   char msgA[32] = "matrix A after filling:";
   debugPrintMatrix(verbose, size, A, msgA);
   
   double startTime = omp_get_wtime();
   
   MatrixMult(size, A, B, C);

   char msgC[32] = "matrix C after MatrixMult(): ";
   debugPrintMatrix(verbose, size, C, msgC);
   
   double endTime = omp_get_wtime();

   printf("\nTotal omp runtime %f seconds (%f milliseconds)\n", 
   (endTime-startTime), (endTime-startTime)*1000);

   if (check) {
      verfiyCorrect(size, C);
   }

   free(A); free(B); free(C); 
   return 0;
}
////////////////////////////////////// end main

// fill a given square matrix with rows of float values 
// equal to each row number
void fillMatrix(int size, float * A) {
   for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        A[i*size + j] = ((float)i);
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

// Check whether last row of result matrix is what we expect
void verfiyCorrect(int size, float *matrix) {
   // determine what the last row should contain
   float lastRowValue = 0.0;
   float maxError = 0.0;
   float nextVal = 0.0;

   for (int i=0; i<size; i++) 
      lastRowValue += i * (size-1);
   
   for (int j=0; j<size; j++) {
      nextVal = matrix[(size-1)*size + j];
      maxError = fmaxf(maxError, fabs(nextVal - lastRowValue));
   }
   printf("max error of last row matrix C values: %f\n", maxError);
}
