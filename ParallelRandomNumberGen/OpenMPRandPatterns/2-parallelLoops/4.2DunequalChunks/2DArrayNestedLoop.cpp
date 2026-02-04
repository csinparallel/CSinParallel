
#include <omp.h>
#include <stdio.h>


#include <trng/uniform01_dist.hpp>
#include <trng/yarn2.hpp>

#include <stdio.h>  // printf()
#include <time.h>   // time()
#include <string.h> // C++ string comparison

#include <omp.h>

// separate file for handling command line arguments
#include "../utils/getCommandLine.h"
// separate file for handling unequal chunk calculation
#include "../utils/chunks.h"
#include "grid.h"

// trng YARN (yet another random number) generator class
#include <trng/yarn2.hpp>
#include <trng/uniform_dist.hpp> // we'll use a uniform distribution
                                 // of the random numbers

void createNewGrid(unsigned long int seed, double *grid, int w, int l, int doleOut);
void populateRows(double *grid, int w, int l, trng::yarn2 RNengine, 
                  trng::uniform01_dist<> uni, int startRow, int endRow, int tid);
void populateColumns(double *grid, int w, int l, int numThreads,
                     trng::yarn2 RNengine, trng::uniform01_dist<> uni, int tid);
void seqSet(int repetitions, long unsigned int seedValue);

#define LEAPFROG 0
#define BLOCKSPLIT 1

////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{

  // set up what conditions we will use:
  //  - defult size of repetitions in the loop
  //  - whether to use a constant seed so we can repeat and
  //    get same random values in stream

  int width = 8;
  int length = 8;
  int useConstantSeed = 0; // for same stream, set this to 1 on cammand line with -c
 
  // rnage for uniform distribution of random numbers
  int min = 1;
  int max = 99;

  // method the random generator will use to dole out the numbers
  int doleOut = LEAPFROG;

  // for openMP
  int numThreads = 1;

  // gather command line arguments
  getArguments(argc, argv, &numThreads, &width, &length,
               &useConstantSeed, &doleOut);

  //check ifvalid number of threads vs repetitions
  if (numThreads > length) {
    printf("\n*** Number of threads (%u) exceeds rows in grid (%u)\n", numThreads, length);
    printf("*** Please run with -t value less than or equal to %u\n\n", length);
    
    return 0;
  }

  // create an array to hold the random numbers on the 'heap':
  // a flattened 2D grid of doubles of size width x length
  int arraySize = (width) * (length);
  size_t bytes = arraySize * sizeof(double);
  double *grid = (double *)malloc(bytes);

  
  int tid = 0;
  omp_set_num_threads(numThreads);
  // Print out info
  printf("trng random number stream will be split ");
  if (doleOut == LEAPFROG) {
    printf("using leapfrog and populate the grid by columns.\n");
  }
  else {
    printf("into blocks and populate the grid by rows.\n");
  }
  printf("The nested loop is partitioned into possibly slightly unequal chunks per thread.\n");
  
   
  // ///////////////  random generator setup /////////////////////
  // random numbers start from a seed value
  long unsigned int seedValue; // note for trng this is long unsigned

  // same constant seed will generate the same sequence of rndom numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed){
    seedValue = 888777666;
  } else {      // variable seed based on computer clock time; use for simulations
    seedValue = (long unsigned int)time(NULL); 
  }

  // debugging info
  if (width <= 8 && length <= 8) {
    int sampleSize = (width * length) + 16;
    printf("the stream of random numbers is:\n");
    seqSet(sampleSize, seedValue); // print a sample of random numbers for debugging
    printf("The per-thread output is printed like this:\n");
    printf("randNumber flattenedIndex threadID row col |\n");
  }

  // random numbers into new grid
  createNewGrid(seedValue, grid, width, length, doleOut);

  // print final grid for debugging
  if (width <= 8 && length <= 8) {
    printf("\nFinal grid of random numbers:\n");
    printGrid(grid, width, length);
  }
  // free the grid memory
  free(grid);

  return 0;
}

// Create a new grid of random numbers using trng and OpenMP.
//
void createNewGrid(unsigned long int seed, double *grid, int w, int l, int doleOut) {

  trng::yarn2 RNengine;       // generator
  trng::uniform01_dist<> uni;  // uniform distribution for random numbers
                               // in the range [0.0, 1.0)

  #pragma omp parallel default(none) \
  shared(grid, w, l, seed, doleOut) private(RNengine, uni)
  {
    int i, j;

    RNengine.seed((long unsigned int)seed); // seed the generator

    int tid = omp_get_thread_num();
    int numThreads = omp_get_num_threads();

    // for blocking mode case
    int startRow =0;
    int endRow = 0;
    if (doleOut == LEAPFROG) {
      if (numThreads > 1) {
          // Use leapfrogging to partition random numbers among threads
          RNengine.split((unsigned)numThreads, tid);
      }
    } else {
      // Use block splitting to partition random numbers among threads
      getStartStopRow(tid, numThreads, l, &startRow, &endRow);   // enables unequal blocks per thread
      long unsigned int numsToSkip = (long unsigned int)startRow * (long unsigned int)w;
      RNengine.jump(numsToSkip);
    }
    
   
    // iterate over the grid by either rows or columns depending on doleOut method
    if (doleOut == LEAPFROG) {
        populateColumns(grid, w, l, numThreads, RNengine, uni, tid);

    } else {
        populateRows(grid, w, l, RNengine, uni, startRow, endRow, tid);
    }
  }  // end of parallel region
}

// Traverse row by row, knowing the start and end row for this thread.
// PREREQUISITE:  block splitting of the random numbers between threads is being used.
//
void populateRows(double *grid, int w, int l, trng::yarn2 RNengine, trng::uniform01_dist<> uni,
                  int startRow, int endRow, int tid) {
  int i, j;
  double randN;

  for (i = startRow; i < endRow; i++) {
    for (j = 0; j < w; j++) {    
      randN = uni(RNengine);     // inside loop

      int id = i * w + j;    // flattened 2D index

      if (w <= 8 && l <= 8) {// for debugging, print the random number and indices
        printf("%0.3f %2d %2d %d %d |\n", randN, id, tid, i, j);
      }

      grid[id] = randN;
    }
  }
}

// Traverse column by column per thread to populate the grid.
// PREREQUISITE:  using leapfrog method of splitting random number
// stream among threads.
//
void populateColumns(double *grid, int w, int l, int numThreads,
                     trng::yarn2 RNengine, trng::uniform01_dist<> uni, int tid) {
  int i, j;
  double randN;

  for (i = 0; i < l; i++) {
    for (j = tid; j < w; j += numThreads) {  
      randN = uni(RNengine);     // inside loop

      int id = i * w + j;    // flattened 2D index

      if (w <= 8 && l <= 8) {// for debugging, print the random number and indices
        printf("%0.3f %2d %d %d %d |\n", randN, id, tid, i, j);
      }
      grid[id] = randN;
    }
  }
}

// Print a sequential set of random numbers for debugging.
void seqSet(int repetitions, long unsigned int seedValue) {
  
  // number generation needs two things: a generator and a distribution of the numbers
  // declare the generator object
  trng::yarn2 randGen;
  // declare the distribution to use (here it is uniform with vallues in the range min to max)
  trng::uniform01_dist<> uniform;

  // Set the starting point of the generator by seeding it
  randGen.seed(seedValue);

   // ////////////////////// end PRNG setup //////////////////////////////////

  // ///////////////////// get a portion of the stream ///////////////////////
  double nextRandValue;   // holds the next value as we go through the loop

  // loop to get each number in the PRNG stream and print it
  // Note here the ubiquitous for loop construction of incrementing by 1
  int i;
  for (i=0; i < repetitions; i++) {
    // get next number in the stream from the distribution
    nextRandValue = uniform(randGen);
     // print tid(i):nextRandValue
    printf("%0.3f ",  nextRandValue);
    if ((i+1) % 20 == 0) {
      printf("\n");
    }
  }
  printf("\n");
}