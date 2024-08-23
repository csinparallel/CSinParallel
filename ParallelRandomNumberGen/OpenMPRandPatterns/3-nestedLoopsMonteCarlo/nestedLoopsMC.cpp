/*
 * Parallel version of a nested for loop that simulates monte carlo simulations.
 * This program generates random floats between 0 and 1.0 using the trng library.
 *
 * The random numbers are determined to be above or below a probability threshold,
 * which is technique used in monte carlo simulations
 *
 * Inner loop:
 * This version uses the data decomposition method that splits 
 * a for loop so that each thread does one loop iteration, then
 * skips forward the number of threads for its next point in the
 * loop.
 *
 * Libby Shoop
 *  August, 2024
*/
#include <stdio.h>        // printf()
#include <time.h>        // time()

#include <omp.h>

// separate file for handling command line arguments
#include "getCommandLine.h"

#include <trng/yarn2.hpp>  // trng YARN (yet another random number) generator class
#include <trng/uniform_dist.hpp>   // we'll use a uniform distribution 
                                   // of the random numbers


////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

  // set up what conditions we will use:
  //  - defult size of time Steps in the outer loop
  //  - number of players, which is then the size of the vectors
  //  - whether to use a constant seed so we can repeat and 
  //    get same random values in stream

  int timeSteps = 100;
  int numPlayers = 512;
  int useConstantSeed = 0; // for same stream, set this to 1 on cammand line with -c

  // rnage for uniform distribution of random numbers
  int min = 0;
  int max = 1.0;
  
  // for openMP
  int numThreads = 1;

	// gather command line arguments
  getArguments(argc, argv, &numThreads, &timeSteps, &numPlayers, &useConstantSeed);

  // openMP additions +++++++++++++++++++++++++++++++++++++++++++
  int tid =0;
  omp_set_num_threads(numThreads);
  
  printf("This is a nested loop example for obtaining random number \nin parallel using the trng package.\n");
  // printf("The outer loop represents a fictional set of time steps, or rounds\n");
  // printf("for a game in which players get more tokens if they win during a round \nand lose tokens when they lose.");
  // printf("Rounds are designed so that a win happens\nwith 40 percent probability.\n");
  // printf("The inner loop goes through each player and is partitioned \ninto chunks of 1 per thread.\n");
  // printf("This means the loop index for each thread starts at its tid \nand updates by the number of threads.\n");
  
  // openMP additions +++++++++++++++++++++++++++++++++++++++++++++

  // /////////////////  random generator setup /////////////////////
  
  // random numbers start from a seed value
  long unsigned int seedValue;  // note for trng this is long unsigned

  // same constant seed will generate the same sequence of rndom numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed) {
    printf("using constant seed\n");
    seedValue = 888777666;     
  } else {  // variable seed based on computer clock time
    seedValue = (long unsigned int)time(NULL); // enables variation; use for simulations
  }
  printf("seed: %ld\n", seedValue);

  // number generation needs two things: a generator and a distribution of the numbers
  // declare the generator object
  trng::yarn2 randGen;
  // declare the distribution to use (here it is uniform with vallues in the range min to max)
  trng::uniform_dist<> uniform(min, max);

  // Set the starting point of the generator by seeding it
  // randGen.seed(seedValue);

  // set up the vector to hold the players token counts
  // and the vector to hold the newly calculated values during each round.

  // set up vectors for player tokens
  // holds player token values, initially 100
  std::vector<int> p(numPlayers, 100);
  // holds new token values computed during each time step
  std::vector<int> pnew(numPlayers, 0);


  // Outer loop of time steps
  // Note that each time through this loop, a set of random numbers
  // equal to the size of the vector, or numPlayers, is consumed from the stream.
  for (int t = 0; t < timeSteps; t++) {

    // For threaded openMP version, each thread will fork and have a private copy of:
    // 
    //  its thread id
    //  the random number generator
    //
    // These are also automatically private because they are declared inside the pragma block:
    //
    //  -the next value that will comback from the generator via the distribution
    //  -the loop counters 
    //
    //  The variable representing the uniform distribution, uniform, can be shared.
    //

    // ////////////////////   begin fork here by using pragma 
    #pragma omp parallel default (none) \
    private(tid, randGen) \
    shared(t, numThreads, numPlayers, p, pnew, uniform, seedValue)
    {
      randGen.seed(seedValue);   // for trng, seed should be inside parallel block
      // IMPORTANT: move ahead in the stream during each iteration.
      //            Doesn't do anything when t == 0
      // jump number of random nums per time step that have been used from the stream
      randGen.jump(t * numPlayers);

      // OpenMP addition: get my thread number
      tid = omp_get_thread_num();

      // Addition for openMP and parallel in general: the generator must be set to give
      // the thread its portion of the random numbers.
      // Note if single thread, this isn't necessary.
      if (numThreads > 1) {
          // thread will get substream tid from numThread separate substreams
          randGen.split((unsigned)numThreads, tid);  // this is the leapfrog setup
      } 
      
      // /////////////////////// end PRNG setup ///////////////////////////////////

      float nextRandValue;   // holds the next value as we go through the loop
      
      int i = 0;    //loop counter

      // inner loop to compute new values for player tokens
      // no openMP pragma for this version
      // note each thread starts at a different index, incremented by numThreads
      for (i=tid; i < numPlayers; i+=numThreads) {

        //reset these private these vars each player
        int right=0;   // index of neighbor to the right
        int win = 0;   // hold value of whether each player wins

        // get next number in the stream from the distribution
        // Use it to determine win with 40% probaility
        nextRandValue = uniform(randGen);
        if (nextRandValue > 0.6) {
          win = 1;
        }
        
        if (i < numPlayers -1) {
          right = i + 1;
        } else {
          right = 0;    // wrap around
        }

        // debug
        // printf("t%2d(%2d):%f %d %d\n", tid, i, nextRandValue, win, right);

        if (win) {
          // egalitarian: if your neighbor has more tokens, you get three
          //              this time. Otherwise, you get 2
          //
          // Here is a dependency issue:
          // Note when multiple threads are used, p[i] cannot be updated in place
          // by one thread, since it depends on another value in the array, which must not be
          // updated by another thread until the whole round completes.
          // Thus, we keep new computed values in a separate array during each round.
          // This is a common pattern for monte carlo simulations.
          int winnings = p[right] > p[i] ? 3 : 2;
          pnew[i] = p[i] + winnings;

        } else {
          pnew[i] = p[i] - 2;
          
        }
        // Note that players eventually go into debt and 
        // their number of tokens goes below zero.

      }  // end loop through each player to compute new values


    } // end of parallel block 

    // reset token values to new ones before next round
    //debug
    // printf("index\tp[i]\tpnew[i]\t\n");
    for (int j = 0; j < numPlayers; j++) {
      // debug
      // printf("%d\t%d\t%d\n", j, p[j], pnew[j]);
      p[j] = pnew[j];
    }

  }   // end of outer loop of time step repetitions 

  // show final numbers of tokens (negative means they are in debt)
  printf("\nEnd state after %d rounds\n", timeSteps);
  printf("index\tp[i]\t\n");
  for (int j = 0; j < numPlayers; j++) {
    printf("%d\t%d\n", j, p[j]);
  }

  return 0;
}

