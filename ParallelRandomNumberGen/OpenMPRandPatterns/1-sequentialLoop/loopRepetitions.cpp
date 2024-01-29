/*
 * Sequential version of a loop that generates random integers
 * using the trng library.
 *
 * Libby Shoop
 *  September, 2022
*/

#include <stdio.h>        // printf()
#include <time.h>        // time()

// separate file for handling command line arguments
#include "getCommandLine.h"

#include <trng/yarn2.hpp>  // trng YARN (yet another random number) generator class
#include <trng/uniform_dist.hpp>   // we'll use a uniform distribution 
                                   // of the random numbers

////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

  // set up what conditions we will use:
  //  - defult size of repetitions in the loop
  //  - whether to use a constant seed so we can repeat and 
  //    get same random values in stream
  //  - fixed range of random numbers from 1 to 99
  int repetitions = 8;
  int useConstantSeed = 0; // for same stream, set this to 1 on cammand line with -c
  
  // range for uniform distribution of random numbers
  int min = 1;
  int max = 99;

	// gather command line arguments
  getArguments(argc, argv, &repetitions, &useConstantSeed);

  // ////////////////  random generator setup /////////////////
  
  // random numbers start from a seed value
  long unsigned int seedValue;  // note for trng this is long unsigned

  // same constant seed will generate the same sequence of rndom numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed) {
    seedValue = 888777666;     
  } else {  // variable seed based on computer clock time
    seedValue = (long unsigned int)time(NULL); // enables variation; use for simulations
  }
  // number generation needs two things: a generator and a distribution of the numbers
  // declare the generator object
  trng::yarn2 randGen;
  // declare the distribution to use (here it is uniform with vallues in the range min to max)
  trng::uniform_dist<> uniform(min, max);

  // Set the starting point of the generator by seeding it
  randGen.seed(seedValue);

   // ////////////////////// end PRNG setup //////////////////////////////////

  // ///////////////////// get a portion of the stream ///////////////////////
  int nextRandValue;   // holds the next value as we go through the loop

  // loop to get each number in the PRNG stream and print it
  // Note here the ubiquitous for loop construction of incrementing by 1
  int i;
  for (i=0; i < repetitions; i++) {
    // get next number in the stream from the distribution
    nextRandValue = uniform(randGen);
     // print tid(i):nextRandValue
    printf("t0 (%2d):%2d \n", i, nextRandValue);
  }

}

