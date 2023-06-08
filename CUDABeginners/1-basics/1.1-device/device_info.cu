/*
 *  Use cuda functions to print device information.
 */
// System includes
#include <stdio.h>

// helper functions and utilities to work with CUDA
int _ConvertSMVer2Cores(int major, int minor);
void getDeviceInformation();


int main(int argc, char **argv) {
  
  // shows how many SMs on our device, among other things
  getDeviceInformation();   

  return 0;
}

////////// Details below here.  /////////////////////////////
//
// If you are interested in some details about the CUDA library
// functions that help us find out about the device we are running 
// code on, you can look at the detail below.

/*
 *  Functions for checking info about a GPU device.
 */
 
 // taken from help_cuda.h from the NVIDIA samples.
 // Used to determine how many cores we have for the
 // GPU's partucular architecture.
 //
 inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  
  return nGpuArchCoresPerSM[index - 1].Cores;

 }

// Find out info about a GPU.
// See this page for list of all the values we can "query" for:
// https://rdrr.io/github/duncantl/RCUDA/man/cudaDeviceGetAttribute.html
//
void getDeviceInformation() {
  int devId;            // the number assigned to the GPU
  int memSize;          // shared mem in each streaming multiprocessor (SM)
  int numProcs;         // number of SMs
  
  struct cudaDeviceProp props;

  cudaGetDevice(&devId);
  
  // can get one by one like this
  cudaDeviceGetAttribute(&memSize, 
    cudaDevAttrMaxSharedMemoryPerBlock, devId);
  cudaDeviceGetAttribute(&numProcs,
    cudaDevAttrMultiProcessorCount, devId);
  
  // or we can get all the properties
  cudaGetDeviceProperties(&props, devId);

  // Then print those we are interested in
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devId, props.name,
         props.major, props.minor);

  char msg[256];
  snprintf(msg, sizeof(msg),
             "Total amount of global memory:      %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(props.totalGlobalMem / 1048576.0f),
             (unsigned long long)props.totalGlobalMem);

  printf("%s", msg);

  printf("GPU device shared memory per block of threads on an SM: %d bytes\n", memSize);
  printf("GPU device total number of streaming multiprocessors: %d\n", numProcs);

  printf("With %3d Multiprocessors (MPs), this device has %03d CUDA Cores/MP,\n for total of  %d CUDA Cores on this device.\n",
           props.multiProcessorCount,
           _ConvertSMVer2Cores(props.major, props.minor),
           _ConvertSMVer2Cores(props.major, props.minor) *
               props.multiProcessorCount);

  printf("\n");
  printf("Max dimension sizes of a grid (x,y,z): (%d, %d, %d)\n",
  props.maxGridSize[0], props.maxGridSize[1],
           props.maxGridSize[2]);
  
  printf("Max dimension sizes of a thread block (x,y,z): (%d, %d, %d)\n",
           props.maxThreadsDim[0], props.maxThreadsDim[1],
           props.maxThreadsDim[2]);

}
