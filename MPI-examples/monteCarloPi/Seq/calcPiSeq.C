/*
 * Hannah Sonsalla, Macalester College, 2017
 * Libby Shoop updated to use trng
 *
 *  calcPiSeq.C
 *
 *   ...sequential program to calculate the value of Pi using
 *       Monte Carlo Method.
 *
 * Usage:  ./calcPiSeq <number of tosses>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>

void Get_input(int argc, char* argv[], long* totalNumTosses_p);
long Toss (long numProcessTosses);

int main(int argc, char** argv) {
	long numTosses, numInCircle;
  double piEstimate;
  clock_t start, finish;
  double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/

  Get_input(argc, argv, &numTosses);  // Read total number of tosses from command line

  start = clock();
  numInCircle = Toss(numTosses);
  finish = clock();

  piEstimate = (4*numInCircle)/((double) numTosses);
  printf("Elapsed time = %f seconds \n", (double)(finish-start)/CLOCKS_PER_SEC);
	printf("Pi is approximately %.16f, Error is %.16f\n", piEstimate, fabs(piEstimate - PI25DT));

  return 0;
}

/* Function gets input from command line for totalNumTosses */
void Get_input(int argc, char* argv[], long* numTosses_p){
	if (argc!= 2){
		fprintf(stderr, "usage:  %s <number of tosses> \n", argv[0]);
        fflush(stderr);
        *numTosses_p = 0;
    } else {
		*numTosses_p = atoi(argv[1]);
	}

	// 0 totalNumTosses ends the program
    if (*numTosses_p == 0) {
        exit(-1);
    }
}

// sequential version still using trng library
long Toss(long numTosses) {
  long numInCircle = 0;        
	double x,y;
	unsigned long int seed = (unsigned long int) time(NULL);
  trng::yarn2 r;
  r.seed(seed);
  trng::uniform01_dist<> u;

  for (int toss = 0; toss < numTosses; toss++) {
	  x=u(r);
    y=u(r); // choose random x− and y−coordinates
	  if((x*x+y*y) <= 1.0 ) {
       numInCircle++;
    }
  }
  return numInCircle;
}

/* Function implements Monte Carlo version of tossing darts at a board */
long oldToss (long numTosses){
	long toss, numInCircle = 0;
	double x,y;
  unsigned int seed = (unsigned) time(NULL);
	srand(seed);
	for (toss = 0; toss < numTosses; toss++) {
	   x = rand_r(&seed)/(double)RAND_MAX;
	   y = rand_r(&seed)/(double)RAND_MAX;
	   if((x*x+y*y) <= 1.0 ) numInCircle++;
    }
    return numInCircle;
}
