// A simple simulation of rolling a variable number of dice a chosen 
// number of times.

#include <stdio.h>        // printf()
#include <time.h>        // time()
#include <omp.h>         // OpenMP functions and pragmas

// Here are 2 of many different generators in the trng library.
#include <trng/yarn2.hpp>  // trng YARN (yet another random number) class
#include <trng/lcg64.hpp>  // trng 64-bit linear congruential generator class
#include <trng/uniform_dist.hpp>   // we'll use a uniform distribution 
                                   // of the random numbers

// separate file for handling command line arguments
#include "./utils/getCommandLine.h"

int main(int argc, char* argv[]) {
  
  int numThreads = 1;  // default number of threads to use

  /////////// Default values for the program parameters
  // Use a unsigned int because we could try a great deal of rolls for some tests.
  unsigned int rolls = 20;  // default number of times to roll dice
  int numDice = 2;          // default number of dice to roll
  int verbose = 0;        // when testing, set this to one on cammand line with -v
  int useConstantSeed = 0; // when testing, set this to one on cammand line with -c

	// gather command line arguments
  getArguments(argc, argv, &verbose, &rolls, &numDice, 
              &numThreads, &useConstantSeed);

  // verbose output of values chosen for debugging
  if (verbose) {
    printf("Simulation of rolling %d dice %u times.\n", numDice, rolls);
    if (useConstantSeed) {
      printf("Constant seed value used for each roll.\n");
    }
  }
  // set number of threads to use
  if (numThreads > 1) {
    omp_set_num_threads(numThreads);
  }

  //  set up random number generator for dice rolls
  // random numbers start from a seed value
  long unsigned int seed;  // note for trng this is long unsigned

  // same constant seed will generate the same sequence of random numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed) {
    seed = (long unsigned int)503895321;     
  } else {  // variable seed based on computer clock time
    seed = (long unsigned int)time(NULL); // enables variation; use for simulations
  }

  int i, j;  // loop counters
  int dieValue; // for each roll
  

  // TODO  add forking of threads here with proper shared, private variables
  // #pragma omp parallel default(none) \
  // shared(numThreads, numDice, rolls, seed, verbose) \
  // private(i,j, dieValue)
  {
  trng::yarn2 rand;      // random number generator
  rand.seed(seed);
  int min = 1;     // min die roll
  int max = 7;     // max die roll
  trng::uniform_dist<> uniform(min, max);  // six values for any die roll

  // get our tid
  int tid;
  if (numThreads > 1) {
    tid = omp_get_thread_num();
  } else {
    tid = 0;
  }
  // loop through number of rolls chosen getting random values for
  //  each die for the number of dice chosen.
  
  // TODO use blocks of values per thread 
  // Note that you must use this method with equal blocks
  // of values per thread.

  // if (numThreads > 1) {
  //     rand.jump(tid * ((numDice * rolls)/numThreads));
  // }
    
  // TODO decompose problem: break up the rolls between threads
  // #pragma omp for
  for (i=0; i<rolls; i++) {
   
    for (j=0; j<numDice; j++) {
      dieValue = uniform(rand);
      if (verbose) {
        printf("t%d roll %d die %d %d  \n", tid, i, j, dieValue);
      }
    }
    if (verbose) {
      printf("\n");
    }
    
  }

  } // end of forked threads block

}