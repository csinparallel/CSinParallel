//
// Demonstrate one way of mapping threads in a 2D grid of 2D blocks
// to indexes in a 2D array of data values.
//
// Libby Shoop    March 2023
//

#include <stdio.h>
#include <cuda_runtime.h>

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

// print information about the mapping of a thread in a 2D block
// where the blocks are in a 2D grid
__global__ void find2DIndex() {
  int row = (blockIdx.y * blockDim.y) + threadIdx.y;
  int column = (blockIdx.x * blockDim.x) + threadIdx.x;

  printf("block (%d, %d) thread (%d, %d) maps to (%d, %d) of array\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, column);
}

int main(int argc, char **argv) {

  printf("Host calls: 2x2 grid of 2x2 blocks of 4 threads each\n");
  printf("2D grid of blocks\n");

  // Kernel configuration, where a two-dimensional grid and
  // two-dimensional blocks are configured.
  dim3 dimGrid2D(2, 2);                      // 2x2 = 4 blocks
  dim3 dimBlock2D(2, 2);                  // 2x2 = 4 threads per block
  // TODO: try replacing the above line with the following 
  // by commenting and uncommenting and re-compiling:
//   dim3 dimBlock2D(512, 2);
  // TODO: then try this one next:
//   dim3 dimBlock2D(1024, 2);
  
  find2DIndex<<<dimGrid2D, dimBlock2D>>>();
  cudaCheckErrors("kernel launch failure");

  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");

  return 0;
}

