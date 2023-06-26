#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

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
////////////////////////////////// end cudaCheckErrors

//threads per block is BLOCK_ROW_SIZE * BLOCK_ROW_SIZE
#define BLOCK_SIZE 256  
// number of threads in a row of a square block     
#define BLOCK_ROW_SIZE 16   

//////////////////////    function declarations  ///////////
void fillMatrix(int size, float * A);
__global__ void MatrixMult(int size, float * __restrict__ A, 
                   float * __restrict__ B, float * __restrict__ C);
void showMatrix(int size, float * matrix);
void getArguments(int argc, char **argv, int *size, int *verbose);
void debugPrintMatrix(int verbose, int size, float *matrix, const char *msg);
void verfiyCorrect(int size, float *matrix);

///////////////////////////////////////////////////// begin main
int main (int argc, char **argv) {
   
   // default values
   int size = 256;          // num rows, cols of square matrix
   int verbose = 0;         // default to not printing matrices
   getArguments(argc, argv, &size, &verbose); //change defaults

   printf("matrix rows, cols = %d\n", size);
 
   float * A;  // input matrix
   float * B;  // input matrix
   float * C;  // output matrix

// Use a 'flattened' 1D array of contiguous memory for the matrices
// size = number of rows = number of columns in the square matrices
   size_t num_elements = size * size * sizeof(float);
   cudaMallocManaged(&A, num_elements);
   cudaMallocManaged(&B, num_elements);
   cudaMallocManaged(&C, num_elements);
   cudaCheckErrors("allocate arrays in unified memory");

   fillMatrix(size, A);
   fillMatrix(size, B);
   char msgA[32] = "matrix A after filling:";
   debugPrintMatrix(verbose, size, A, msgA);
   
   // for timing using CUDA functions
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaCheckErrors("create events");
   float milliseconds = 0.0;
   
   // set up 2D grid of 2D thread blocks based on matrix size

   // threads per block
   int blockSide = BLOCK_ROW_SIZE;
   // small cases of size of 4, 8 for debugging
   if (size < 16) blockSide = size; 
   dim3 dimBlock2D(blockSide, blockSide);  

   // number of blocks in a row, col of a square grid 
   int gridSide = 1;
   if (size > BLOCK_ROW_SIZE) {
      gridSide = size/BLOCK_ROW_SIZE; 
   }
   dim3 dimGrid2D(gridSide, gridSide);
   //////////////////////////////////// end grid setup
   
   cudaEventRecord(start);
   cudaCheckErrors("start timing");

   MatrixMult<<<dimGrid2D, dimBlock2D>>>(size, A, B, C);
   cudaCheckErrors("kernal matrix multiply");

   cudaDeviceSynchronize();    // ensure all threads finish
   
   cudaEventRecord(stop);      // end timing
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   cudaCheckErrors("synchronize and finish timing");

   char msgC[32] = "matrix C after MatrixMult(): ";
   debugPrintMatrix(verbose, size, C, msgC);

   printf("\nTotal CUDA time : %f milliseconds\n", milliseconds);

   verfiyCorrect(size, C);

   cudaFree(A); cudaFree(B); cudaFree(C); 
   cudaCheckErrors("free cuda memory");

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

// On the GPU device:
// mutiply matrix A times matrix B, placing result in matrix C
//
__global__ void MatrixMult(int size, float * __restrict__ A, 
                           float * __restrict__ B, float * __restrict__ C) {

   // From 2DArraMapping example
   int row = (blockIdx.y * blockDim.y) + threadIdx.y;
   int col = (blockIdx.x * blockDim.x) + threadIdx.x;

   float tmp = 0.;
   // each thread computes one element of the output matrix C
   for (int k = 0; k < size; ++k) {
      tmp += A[row*size + k] * B[k*size + col];
   }
   C[row*size + col] = tmp;  // one thread updates cell of C once
}

void getArguments(int argc, char **argv, int *size, int *verbose) {
   // 2 arguments optional: 
   //   size of one side of square matrix
   //   verbose printing for debugging
   if (argc > 3) {
      fprintf(stderr,"Usage: %s [size] [verbose]\n", argv[0]);
      fprintf(stderr, "where size is a multiple of %d.\n", BLOCK_ROW_SIZE);
      exit(EXIT_FAILURE);
   }

   if (argc >= 2) {
      *size = atoi(argv[1]);
   
      if (argc == 3) {
         *verbose = atoi(argv[2]);
      }
   }
   
   if (*verbose) {
      printf("size of matrix side: %d\n", *size);
   }

   // cases for debugging, where we will need a smaller thread block
   if (verbose && (*size == 4 || *size == 8) ) return;

   if ( ((*size % BLOCK_ROW_SIZE) != 0) || 
        ((*size % BLOCK_SIZE ) != 0) 
      ) {
      fprintf(stderr, "Usage: %s [size] [verbose]\n", argv[0]);
      fprintf(stderr, "where size is a multiple of %d ", BLOCK_ROW_SIZE);
      fprintf(stderr, "and is a multiple of %d.\n", BLOCK_SIZE);
      exit(EXIT_FAILURE);
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
   for (i=0; i<size; i++) {
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
