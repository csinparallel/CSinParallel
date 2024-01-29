 /*   
 * Hannah Sonsalla, Macalester College, 2017
 * Libby Shoop updated to use trng
 * 
 *  calcPiMPI.C
 *
 *   ...program uses MPI to calculate the value of Pi
 *
 * Usage:  mpirun -np N ./calcPiMPI <number of tosses>
 * 
 * Note that this uses the block splitting technique for
 * generating random numbers. This requires that the number
 * of tosses be equally divisible by the number of processes.
 *
 */
 
#include <mpi.h>  
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>

void Get_input(int argc, char* argv[], int myRank, long* totalNumTosses_p);
long Toss (long numProcessTosses, int myRank);

int main(int argc, char** argv) {
  int myRank, numProcs;
  long totalNumTosses, numProcessTosses, processNumberInCircle, totalNumberInCircle;
  double start, finish, loc_elapsed, elapsed, piEstimate;
  double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // Read total number of tosses from command line
  Get_input(argc, argv, myRank, &totalNumTosses);  
  
  // check for equal chunks per processor, since
  // block splitting requires this
  // only one process needs to print the error
  //
  if ((totalNumTosses % numProcs) == 0) {
    // how many tosses each process will complete
    numProcessTosses = totalNumTosses/numProcs;
  } else {
    if (myRank == 0) {
      printf("Number of tosses must be divisible by number of processors. Exiting.\n");
    }
    MPI_Finalize();
    exit(-1);
  }
  

  MPI_Barrier(MPI_COMM_WORLD);  // start timing
  start = MPI_Wtime();

  processNumberInCircle = Toss(numProcessTosses, myRank);
  

  MPI_Reduce(&processNumberInCircle, &totalNumberInCircle, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  // Get the highest time as the final end time
  finish = MPI_Wtime();
  loc_elapsed = finish-start;
  MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
  
  if (myRank == 0) {
    piEstimate = (4*totalNumberInCircle)/((double) totalNumTosses);
    printf("Elapsed time = %f seconds \n", elapsed);
    printf("Pi is approximately %.16f, Error is %.16f\n", piEstimate, fabs(piEstimate - PI25DT));
  }
  MPI_Finalize(); 
  return 0;
}  

/* Function implements Monte Carlo version of tossing darts at a board */
// Each process runs this.
// Each time through a loop the process creates a set of random x, y values
// between 0.0 and 1.0 from its block determined by the trng jump function.
long Toss(long processTosses, int myRank){
  long numberInCircle = 0;        
	double x,y;
	unsigned long int seed = (unsigned long int) time(NULL);
  trng::yarn2 r;

  r.seed(seed);

  trng::uniform01_dist<> u; // random number distribution
  r.jump(2*(myRank*processTosses)); // jump ahead to set of
                                    // random values for my process:
                                    // this is block splitting
  // throw random points into square and distribute workload over all processes
  for (long i=myRank*processTosses; i<(myRank+1)*processTosses; ++i) {
    x=u(r);
    y=u(r); // choose random x, y coordinates
    if (x*x+y*y<=1.0) {      // is point in unit circle ?
      ++numberInCircle; // increase counter
    }
  } 
  return numberInCircle;
}

/* Function gets input from command line for totalNumTosses */
void Get_input(int argc, char* argv[], int myRank, long* totalNumTosses_p){
	if (myRank == 0) {
		if (argc!= 2){
		    fprintf(stderr, "usage: mpirun -np <N> %s <number of tosses> \n", argv[0]);
            fflush(stderr);
            *totalNumTosses_p = 0;
		} else {
			*totalNumTosses_p = atoi(argv[1]);
		}
	}
	// Broadcasts value of totalNumTosses to each process
	MPI_Bcast(totalNumTosses_p, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	
	// 0 totalNumTosses ends the program
    if (*totalNumTosses_p == 0) {
        MPI_Finalize();
        exit(-1);
    }
}

