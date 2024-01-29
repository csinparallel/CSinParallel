/*
 * Create random number sequence in parallel, placing each number
 * generated into a 'bin'.
 *
 * Usage:
 *  Print runtime: ./histogram_omp -t 16 -n 100000000 -i 15 -a 0 -b 100
 *  Print histogram data: add tag "-p"
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <omp.h>

// a few different generators and distributions
#include <trng/lcg64_shift.hpp>
#include <trng/lcg64.hpp>
#include <trng/yarn2.hpp>
#include <trng/mt19937_64.hpp>
#include <trng/normal_dist.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/exponential_dist.hpp>

// separate file for handling command line arguments
#include "./utils/getCommandLinePar.h"

using namespace std;   // for 'cout' style of printing in C++

///==============  main ======================
int main(int argc, char* argv[])
{
  int nThreads = 16;
  int N = 100000000;
  int numBins = 15;
  // use these to experiment with a distribution of doubles
  // double min = 0;
  // double max = 10000;
  int min = 0;
  int max = 10000;
  int print = 0;
  int useConstantSeed = 0;

  int index, i;
  // double randN;    // if want to try double values
  int randN;
  double d_index;

  getArguments(argc, argv, &nThreads, &N, &numBins, &min, &max, &print, &useConstantSeed);
  
  omp_set_num_threads(nThreads);

  //debug
  if (print)
    printf("%d random values from %d to %d placed into %d bins\n", N, min, max, numBins);

  int hist_array [numBins] = {};

  // random numbers start from a seed value
  long unsigned int seed;  // note for trng this is long unsigned

  if (useConstantSeed) {
    seed = (long unsigned int)503895321;     
  } else {  // variable seed based on computer clock time
    seed = (long unsigned int)time(NULL); // enables variation; use for simulations
  }

  double start = omp_get_wtime();

  ///// reduction on individual elements of an array
  // 
  // Note this clause below:
  //  reduction(+:hist_array[:numBins])
  //            ^ adding     ^ elements from index 0 to numBins-1
  ///             to elements
  ///             in array
  //
  // like we can reduce on an individual value, we can reduce when
  // adding at a particular index in an array
  //
  // If you need to do this for 2 arrays,use 2 full reduction clauses

#pragma omp parallel  \
  shared(N, max, min, numBins, nThreads, seed, useConstantSeed) \
  private(i, d_index, index, randN) \
  reduction(+:hist_array[:numBins]) \
  default(none)
  {
     // initialize random number engine
     // Note a different choice here from the sequential version.
     // Try different engines and distributions and note the results.
     // see page 25 of the trng.pdf document for list of parallel and
     // sequential engines.
    
    // trng::lcg64_shift RNengine1;
    trng::lcg64 RNengine1;
    // trng::mt19937_64 RNengine1; // a Mersenne twister one
    // trng::yarn2 RNengine1;
    RNengine1.seed(seed);

    //////// set up the leapfrogging method
    int rank = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    // Addition for openMP and parallel in general: the generator must be set to give
    // the thread its portion of the random numbers.
    // Note if single thread, this isn't necessary.
    if (numThreads > 1)
    {
      // choose sub-stream no. rank out of nThreads stream
      RNengine1.split(nThreads, rank);
    }
    //////// end leapfrog setup

    // initialize uniform distribution
    trng::uniform_dist<> uni(min, max);

    for (i = rank; i < N; i+=nThreads)
    {
      randN = uni(RNengine1);
      // printf("r %d ", randN);   // debug

      // for doubles
      // printf("r %f ", randN);   // debug
      // d_index = ((randN - min) / (max - min)) * numBins;
      // index = (int) d_index;
      
      int index = (randN - min) * numBins / (max - min);
      hist_array[index] ++;
    }
  }

  double end = omp_get_wtime(); //ends timer
  double runtime = end - start;

  if (print == 0){
    cout << endl << "RUNTIME: " << runtime;
  }
  else{
    cout << endl;
    for (int k = 0; k < numBins; k++)
    {
      cout << hist_array[k] << "\t";
    }
  }
  cout << endl;
}


