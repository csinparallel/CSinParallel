#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

__global__ void hello() {
    // special dim3 variables available to each thread in a kernel
    // or device function:
    // blockIdx    the x, y, z coordinate of the block in the grid
    // threadIdX   the x, y, z coordinate of the thread in the block
    printf("I am thread (%d, %d, %d) of block (%d, %d, %d) in the grid\n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z );

}

// Note that this is called from the host, not the GPU device.
// We create dim3 structs there and can print their components
// with this function
void printDims(dim3 gridDim, dim3 blockDim) {
    printf("Grid Dimensions : {%d, %d, %d} blocks. \n",
    gridDim.x, gridDim.y, gridDim.z);

    printf("Block Dimensions : {%d, %d, %d} threads.\n",
    blockDim.x, blockDim.y, blockDim.z);
}

int main(int argc, char **argv) {

    // dim3 is a special data type: a vector of 3 integers.
    // each integer is accessed using .x, .y and .z (see printDims() above)

    // 1 dimensionsional case is the following: 
    // 1D grid of 2 1D blocks
    dim3 gridDim(2);      // 2 blocks in x direction, y, z default to 1
    dim3 blockDim(8);     // 8 threads per block in x direction
    
    printDims(gridDim, blockDim);
    
    printf("From each thread:\n");
    hello<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();     // need for printfs in kernel to flush

    return 0;
}
