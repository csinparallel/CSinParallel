/*
  Original code provided by Dave Valentine, Slippery Rock University.
  Changed to use trng by Libby Shoop, Macalester College.
*/

//
// Simulate many coin flips 
// to determine how random the values are.
// The program starts with a trial of 256 random coin
// flips, where a randomly-generated integer is 'heads'
// if it is even and 'tails' if it is odd.
// The user can indicate how many trials to try, where
// each succesive trial flips twice as many coins as the 
// one before it. 
// You can specify the following on the command line:
//   The number of trials. -n 
//   Whether to use a constant value for the seed of the 
//   random number generator. 
//      When  -c used, the seed will be constant, otherwise
//       the seed is based on the clock time of the machine.
//   Verbose output for debugging. -v
//   Usage of program. -h
//

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

// functions to show different ways of generating the random numbers in parallel
void doFlipsSequential(long unsigned int seed, 
                       unsigned int trialFlips, 
                       int verbose);

// Standard chi sqaure test
// We can use this to check the random number distribution. See:
// http://datanuggets.org/wp-content/uploads/2014/03/Student-Guide-to-the-Chi-Square-Test.pdf
//
// Any value less than 3.84 means that the 'coin' is fair.
double chiSq(int heads, int tails) {
	double sum = 0;					//chi square sum
	double tot = heads+tails;		//total flips
	double expected = 0.5 * tot;	//expected heads (or tails)

	sum = ((heads - expected)*(heads-expected)/expected) + \
		((tails - expected)*(tails-expected)/expected);
	return sum;
}

////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
 
  /////////// Default values for the program parameters
  // Use a unsigned int because we could try a great deal of flips for some trials.
  unsigned int trialFlips = 256;  // default number of times to 'flip', doubling each trial
  int numTrials = 20;      // default number of trials
  int verbose = 0;        // when testing, set this to one on cammand line with -v
  int useConstantSeed = 0; // when testing, set this to one on cammand line with -c

	  // gather command line arguments
  getArgumentsSeq(argc, argv, &verbose, &numTrials, &useConstantSeed);

  if (verbose) {
	  printf("Sequential Simulation of Coin Flip using the trng library.\n");
    if (useConstantSeed)  printf("Using constant starting seed.\n");
  }

	//print our heading text
	printf("\n\n%15s%15s%15s%15s%15s%15s%15s",
           "Method","Flips","numHeads","numTails","total",
           "Chi Squared", "Time (sec)\n");

  // Try several trials of different numbers of flips, 
  // doubling how many each round.
 
  // random numbers start from a seed value
  long unsigned int seed;  // note for trng this is long unsigned

  // same constant seed will generate the same sequence of rndom numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed) {
    seed = 503895321;     
  } else {  // variable seed based on computer clock time
    seed = (long unsigned int)time(NULL); // enables variation; use for simulations
  }

  int trial; 
  for (trial = 0; trial < numTrials; trial++) {
    if (verbose) {
      printf("Trial %d of %d start\n", trial, numTrials);
    }

    doFlipsSequential(seed, trialFlips, verbose);
   
    trialFlips *= 2;   // double the number of flips for the next trial
  } // end trials

	return 0;
}
///////////////////////////////////////////////// end main

void doFlipsSequential(long unsigned int seed, unsigned int trialFlips, 
                       int verbose) {
  int numFlips,			//loop control
		numHeads, numTails;	//counters
  
  double ompStartTime, ompStopTime;  // time each trial

  unsigned int tid; // thread id when forking threads

  unsigned int nextRandVal;
  
  // range of random unsigned ints to generate
  unsigned min = 0;
  unsigned max = (1<<28) - 1;  // a large odd number
  if (verbose) {
      printf("Random number range: %u to %u\n", min, max);
  }

  numHeads = 0;               //reset counters
  numTails = 0;
  ompStartTime = omp_get_wtime();   //get start time for this trial

  trng::lcg64 rand;
  // trng::yarn2 rand;       // could try this one; should perform about the same
  rand.seed(seed);
  // distribution is declared inside pragma block
  trng::uniform_dist<> uniform(min, max);  

  tid = omp_get_thread_num();

  for (numFlips=0;  numFlips<trialFlips; numFlips++) {
    nextRandVal = uniform(rand);
    if (verbose) {
      printf("t%u %lu ", tid, nextRandVal);
    }

    if (nextRandVal%2 == 0) // if random number is even, call it heads
      numHeads++;
    else
      numTails++;
  }
  ompStopTime = omp_get_wtime();

  // Finish this trial by printing out results

  printf("%15s%15d%15d%15d%15d%15.6f%15.6f\n", "sequential", trialFlips, numHeads, numTails,
          (numHeads+numTails), chiSq(numHeads, numTails),
          (double)(ompStopTime-ompStartTime));  
}

