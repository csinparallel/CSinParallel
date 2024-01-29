/*
  Original code provided by Dave Valentine, Slippery Rock University.
  Changed to use trng by Libby Shoop, Macalester College.
*/

//
// Simulate many coin flips on multiple threads
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
//   Number of threads to use: -t
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
void leapfrog1(long unsigned int seed, unsigned int trialFlips, 
              int numThreads, int verbose);
void leapfrog2(long unsigned int seed, unsigned int trialFlips, 
              int numThreads, int verbose);
void blocksplit(long unsigned int seed, unsigned int trialFlips, 
              int numThreads, int verbose);

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
  /***  OMP ***/
  int numThreads = 1;  // default number of threads to use

  /////////// Default values for the program parameters
  // Use a unsigned int because we could try a great deal of flips for some trials.
  unsigned int trialFlips = 256;  // default number of times to 'flip', doubling each trial
  int numTrials = 20;      // default number of trials
  int verbose = 0;        // when testing, set this to one on cammand line with -v
  int useConstantSeed = 0; // when testing, set this to one on cammand line with -c

	  // gather command line arguments
  getArguments(argc, argv, &verbose, &numTrials, &numThreads, &useConstantSeed);

/***** Initialization *****/

  /***  OMP ***/
  omp_set_num_threads(numThreads);
  /***  OMP ***/

  if (verbose) {
	  printf("Threaded Simulation of Coin Flip using the trng library and %d threads.\n", numThreads);
    if (useConstantSeed)  printf("Using constant starting seed.\n");
  }

	//print our heading text
	printf("\n%12s%12s%12s%12s%12s%12s%12s%12s",
           "Method","Trials","numHeads","numTails","total",
           "Chi Squared", "Threads", "Time (sec)\n");

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
    // TODO: comment out this when you understand how the program is working
    // printf("Seed is %lu\n", seed);   
    // TODO: eventually comment out 2 of these and just use one method,
    //       once you convince yourself that they all produce the same results.
    leapfrog1(seed, trialFlips, numThreads, verbose); 
    leapfrog2(seed, trialFlips, numThreads, verbose);
    blocksplit(seed, trialFlips, numThreads, verbose); 

    trialFlips *= 2;   // double the number of flips for the next trial
  } // end trials

	return 0;
}
///////////////////////////////////////////////// end main

// This version uses trng 'split' function to create the leapfrog
// streams of random numbers for each thread.
// It also uses 2 OpenMP pragmas to fork and signify that the for
// loop should be split up evenly amoung the threads.
// Study the pragma lines carefully and compare to the leapfrog2() version.
// IMPORTANT: this version of the loop matches this OpenMP patternlet:
//            06.parallelLoop-equalChunks
//         go to that example and compare it to this code's for loop.
void leapfrog1(long unsigned int seed, 
               unsigned int trialFlips, 
              int numThreads, int verbose) {
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

 
// fork threads to generate random numbers in parallel
  #pragma omp parallel default(none) \
  private(numFlips, tid, nextRandVal) \
  shared (trialFlips, numThreads, min, max, verbose, seed) \
  reduction(+:numHeads, numTails)
  {
    trng::lcg64 rand;
    // trng::yarn2 rand;       // could try this one; should perform about the same
    rand.seed(seed);
    // distribution is declared inside pragma block
    trng::uniform_dist<> uniform(min, max);  

    tid = omp_get_thread_num();
    
    if (numThreads > 1) {
      // thread will get substream tid from numThread separate substreams
      rand.split((unsigned)numThreads, tid);  // this is the leapfrog setup
    }

    // each thread will get trialFlips/numThreads total random values

    // VERSION 1 of for loop: openMP splits the work
    #pragma omp for
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
  }

  ompStopTime = omp_get_wtime();

  // Finish this trial by printing out results

  printf("%12s%12d%12d%12d%12d%12.6f%12d%12.6f\n", "leap1", trialFlips, numHeads, numTails,
          (numHeads+numTails), chiSq(numHeads, numTails), numThreads,
          (double)(ompStopTime-ompStartTime));  
}

// This version shows an alternative way to use pragmas and loops
// to enable each thread to split the work evenly.
// Study the pragma line and the loop control and compare to leapfrog1().
// IMPORTANT: this version of the loop matches this OpenMP patternlet:
//            07.parallelLoop-chunksOf1
void leapfrog2(long unsigned int seed, unsigned int trialFlips, 
              int numThreads, int verbose) {
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

  ///////////////// IMPORTANT
  // The random number generator can be declared 
  // outside of the parallel block and be declared private in the pragma
  trng::lcg64 rand;
  // trng::yarn2 rand;

// fork threads to generate random numbers in parallel
  #pragma omp parallel default(none) \
  private(numFlips, tid, nextRandVal, rand) \
  shared (trialFlips, numThreads, min, max, verbose, seed) \
  reduction(+:numHeads, numTails)
  {
    rand.seed(seed);
    // distribution is declared inside pragma block
    trng::uniform_dist<> uniform(min, max);  

    tid = omp_get_thread_num();
    
    if (numThreads > 1) {
      // thread will get substream tid from numThread separate substreams
      rand.split((unsigned)numThreads, tid);  // this is the leapfrog setup
    }
    
    // each thread will get trialFlips/numThreads total random values

    // VERSION 2 of for loop: programmer splits the work for each thread.
    // We start at its thread number and advance the loop to 
    // the thread's next leapfrog position by adding numThreads to the counter.
    for (numFlips=tid; numFlips<trialFlips; numFlips+=numThreads) {
      nextRandVal = uniform(rand);
      if (verbose) {
        printf("t%u %lu ", tid, nextRandVal);
      }

      if (nextRandVal%2 == 0) // if random number is even, call it heads
        numHeads++;
      else
        numTails++;
    }
  }  // end of forked threads

  ompStopTime = omp_get_wtime();

  // Finish this trial by printing out results

  printf("%12s%12d%12d%12d%12d%12.6f%12d%12.6f\n", "leap2", trialFlips, numHeads, numTails,
          (numHeads+numTails), chiSq(numHeads, numTails), numThreads,
          (double)(ompStopTime-ompStartTime));  
}


void blocksplit(long unsigned int seed, unsigned int trialFlips, 
              int numThreads, int verbose) {
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

  ///////////////// IMPORTANT
  // The random number generator can be declared 
  // outside of the parallel block and be declared private in the pragma
  trng::lcg64 rand;
  // trng::yarn2 rand;
 
// fork threads to generate random numbers in parallel
  #pragma omp parallel default(none) \
  private(numFlips, tid, nextRandVal, rand) \
  shared (trialFlips, numThreads, min, max, verbose, seed) \
  reduction(+:numHeads, numTails)
  {
    rand.seed(seed);

    // distribution is declared inside pragma block
    trng::uniform_dist<> uniform(min, max);  

    tid = omp_get_thread_num();
    
    if (numThreads > 1) {
      // block of values from the stream will start at point given to jump function
      rand.jump(tid * (trialFlips / numThreads));  
    }
    // each thread will get trialFlips/numThreads total random values

    // for block splitting, we set up the start and end range for each thread's
    // block of random numbers in the stream
    int start = tid * (trialFlips / numThreads);
    int end = (tid + 1) * (trialFlips / numThreads);
    for (numFlips = start;  numFlips<end; numFlips++) {
      nextRandVal = uniform(rand);
      if (verbose) {
        printf("t%u %lu ", tid, nextRandVal);
      }

      if (nextRandVal%2 == 0) // if random number is even, call it heads
        numHeads++;
      else
        numTails++;
    }
  }

  ompStopTime = omp_get_wtime();

  // Finish this trial by printing out results

  printf("%12s%12d%12d%12d%12d%12.6f%12d%12.6f\n", "blocksplit", trialFlips, numHeads, numTails,
          (numHeads+numTails), chiSq(numHeads, numTails), numThreads,
          (double)(ompStopTime-ompStartTime));  
}
