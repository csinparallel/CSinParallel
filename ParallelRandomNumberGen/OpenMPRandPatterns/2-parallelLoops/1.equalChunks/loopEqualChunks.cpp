/*
 * Parallel version of a loop that generates random integers
 * using the trng library.
 * This version uses the data decomposition method that splits
 * a for loop into equal chunks.
 *
 * Libby Shoop
 *  September, 2022
 */
#include <stdio.h>  // printf()
#include <time.h>   // time()
#include <string.h> // C++ string comparison

#include <omp.h>

// separate file for handling command line arguments
#include "../utils/getCommandLine.h"

// trng YARN (yet another random number) generator class
#include <trng/yarn2.hpp>
#include <trng/uniform_dist.hpp> // we'll use a uniform distribution
                                 // of the random numbers

#define LEAPFROG 0
#define BLOCKSPLIT 1

////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{

  // set up what conditions we will use:
  //  - default size of repetitions in the loop
  //  - whether to use a constant seed so we can repeat and
  //    get same random values in stream

  int repetitions = 8;
  int useConstantSeed = 0; // for same stream, set this to 1 on cammand line with -c
 
  // rnage for uniform distribution of random numbers
  int min = 1;
  int max = 99;

  // method the random generator will use to dole out the numbers
  int doleOut = LEAPFROG;

  // for openMP
  int numThreads = 1;

  // gather command line arguments
  getArguments(argc, argv, &numThreads, &repetitions,
               &useConstantSeed, &doleOut);

  // openMP additions +++++++++++++++++++++++++++++++++++++++++++
  int tid = 0;
  omp_set_num_threads(numThreads);
  // Print out info
  printf("trng random number stream will be split ");
  if (doleOut == LEAPFROG)
  {
    printf("using leapfrog.\n");
  }
  else
  {
    printf("into blocks.\n");
  }
  printf("The loop is partitioned into equal chunks per thread.\n");
  printf("This means that the loop indices should be consecutive per thread.\n");
  printf("the output is printed like this:\n");
  printf("thread (loop index):randNumber\n");
  // openMP additions +++++++++++++++++++++++++++++++++++++++++++++

  // ///////////////  random generator setup /////////////////////

  // random numbers start from a seed value
  long unsigned int seedValue; // note for trng this is long unsigned

  // same constant seed will generate the same sequence of rndom numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed)
  {
    seedValue = 888777666;
  }
  else
  {                                            // variable seed based on computer clock time
    seedValue = (long unsigned int)time(NULL); // enables variation; use for simulations
  }

// For threaded openMP version, each thread will fork and have
// a private copy of:
//
//  its thread id
//
// These are also automatically private because they are
// declared inside the pragma block:
//  -the generator and the distribution
//  -the next value that will comback from the generator via the distribution
//  -the loop counter
//
//  Since seedValue and repetitions are read but not written,
//  they are shared.
//
// //////////////   begin fork here by using pragma for the compiler
#pragma omp parallel default(none) private(tid) \
    shared(seedValue, repetitions, min, max, numThreads, doleOut)
  {
    // number generation needs two things: a generator and a distribution of the numbers
    // declare the generator object
    trng::yarn2 randGen;
    // declare the distribution to use (here it is uniform with vallues in the range min to max)
    trng::uniform_dist<> uniform(min, max);

    // Set the starting point of the generator by seeding it
    randGen.seed(seedValue);

    // OpenMP addition: get my thread number
    tid = omp_get_thread_num();

    // Addition for openMP and parallel in general: the generator must be set to give
    // the thread its portion of the random numbers.
    // Note if single thread, this isn't necessary.
    if (numThreads > 1)
    {
      if (doleOut == LEAPFROG)
      {
        // thread will get substream tid from numThread separate substreams
        randGen.split((unsigned)numThreads, tid); // this is the leapfrog setup
      }
      else
      {
        // thread will get substream as a block:
        // thread 0 starts at 0, thread 1 starts at repetitions / numThreads, etc.
        // Note: this works correctly only if repetitions is evenly divisible by numThreads.
        // Uncomment the next line to check the jump value
        // printf("tid, jump val: %d %d\n", tid, repetitions / numThreads);
        randGen.jump(tid * (repetitions / numThreads)); // block split
      }
    }
    // //////////////// end PRNG setup /////////////////////////////////

    // //////////////// get a portion of the stream /////////////////////
    int nextRandValue; // holds the next value as we go through the loop

    // loop to get each number in the PRNG stream and print it
    // Note here the ubiquitous for loop construction of incrementing by 1-
    //   this leads itself nicely to letting the openMP compiler split the
    //   loop into equal chunks using the pragma shown.
    int i;
#pragma omp for
    for (i = 0; i < repetitions; i++)
    {
      // get next number in the stream from the distribution
      nextRandValue = uniform(randGen);
      // print tid(i):nextRandValue
      printf("t%2d (%2d):%2d \n", tid, i, nextRandValue);
    }

  } // end of parallel block

  return 0;
}
