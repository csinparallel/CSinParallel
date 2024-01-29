/*
 * Generate pseudo-random numbers using TRNG library and creates
 * an array of "bins" that stores the histogram information
 * for Usage: ./histogram_seq -h
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <omp.h>

//////////// NOTE  how we need to include the generator and distribution include
////////////       files from trng library
#include <trng/mt19937_64.hpp>
#include <trng/mt19937.hpp>
#include <trng/lcg64_shift.hpp>
#include <trng/yarn2.hpp>
#include <trng/normal_dist.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/exponential_dist.hpp>

// separate file for handling command line arguments
#include "./utils/getCommandLineSeq.h"

using namespace std;  // for 'cout' style of printing in C++

int main(int argc, char *argv[])
{
  //initialize parameters and set their default values
  int N = 10000000;
  int numBins = 12;
  
  int min = 0;
  int max = 10000;
  int print = 0;
  int useConstantSeed = 0;

///////////////////==========================================
  // command line arguments
  getArguments(argc, argv, &N, &numBins, &min, &max, &print, &useConstantSeed);
  //debug
  if (print)
  printf("%d values between %d and %d placed into %d bins\n", 
         N, min, max, numBins);

  //initialize the array that will hold counts in 'bins'
  int bins [numBins] = {};

  // random numbers start from a seed value
  long unsigned int seed;  // note for trng this is long unsigned

  double start = omp_get_wtime();  // start timing

/////////////////// random engine ==================================
   // initialize random number engine
   // Note there are several choices- see trng documentation:
   // trng.pdf, p. 25
  // trng::mt19937_64 rGen; // a Mersenne twister one
  // trng::mt19937 rGen;
  // trng::lcg64_shift rGen;

  trng::yarn2 rGen;

  // same constant seed will generate the same sequence of random numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed) {
    seed = (long unsigned int)503895321;     
  } else {  // variable seed based on computer clock time
    seed = (long unsigned int)time(NULL); // enables variation; use for simulations
  }

  rGen.seed(seed);

///////////////////  distribution ===================================
  // initialize normal distribution
  trng::normal_dist<> normal(4000.0, 1000.0);   

/////////////////// new random value placed in a 'bin' ==============
// find the correct bin for a new random number generated from the
// distribution.
  for (int i = 0; i < N; i++)
  {
    // follow normal distribution for integers
    int randN = normal(rGen);

    // index for bin where value belongs
    int index = (randN - min) * numBins / (max - min);
    bins[index] ++;
  }

  double end = omp_get_wtime();
	double elapsed_time = end - start;
  
  if (print == 0){
    cout << endl << "RUNTIME: " << elapsed_time;
  }
  else{
    cout << endl;
    for (int k = 0; k < numBins; k++)
    {
      cout << bins[k] << "\t";
    }
  }
  cout << endl;
}


