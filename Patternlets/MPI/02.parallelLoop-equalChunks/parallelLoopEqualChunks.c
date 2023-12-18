/* parallelLoopEqualChunks.c
 * ... illustrates the parallel for loop pattern in MPI
 *	in which processes perform the loop's iterations in equal-sized 'chunks'
 *	(preferable when loop iterations access memory/cache locations) ...
 * Joel Adams, Calvin College, November 2009.
 *    updated by Libby Shoop, Macalester College, 2017
 *
 * Usage: mpirun -np N ./parallelForEqualChunks
 *
 * Exercise:
 * - Compile and run, varying N: 1, 2, 4, and 8
 * - Change REPS to 16, save, recompile, rerun, varying N again.
 * - Explain how this pattern divides the iterations of the loop
 *    among the processes.
 */

#include <stdio.h> // printf()
#include <mpi.h>   // MPI

int main(int argc, char** argv) {
    const int REPS = 8;                      // repetitions in a loop
    int id = -1, numProcesses = -1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // In this example, ensure that the number of processes doesn't exceed REPS.
    if (numProcesses <= REPS) {
      int chunkSize = REPS / numProcesses;      // find minimum chunk size
      int extra = REPS % numProcesses;     // spread one more to this many processes
      int firstLargerId = numProcesses - extra;  // first proc getting chunkSuze + 1

      int start = 0; // start index in loop for my process
      int stop = 0;  // end index of loop for my process

      if (id < firstLargerId) {    // these processes get chunkSize steps
        start = id * chunkSize;     // skip by chunkSize unless I'm process 0
        stop = start + chunkSize;        
      } else {                      // these get chunksize + 1 steps
        start = (firstLargerId * chunkSize) + ((id - firstLargerId) * (chunkSize + 1));   
        stop = start + chunkSize + 1;  
      }

      for (int i = start; i < stop; i++) {      // iterate through our range
          printf("Process %d is performing iteration %d\n", id, i);
      }

    } else {  // more processes than REPS
      if (id == 0) {
          printf("Please run with -np less than or equal to %d\n.", REPS);
      }
    }

    MPI_Finalize();
    return 0;
}
