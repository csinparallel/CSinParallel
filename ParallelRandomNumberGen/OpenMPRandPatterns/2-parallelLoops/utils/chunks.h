/* chunks.h
 *  ... defines getChunkStartStopValues() to divide a range of zero-based values
 *  (0..N-1, inclusive) into P contiguous chunks that differ in size by at most 1.
 *  Example: A PE (process or thread) can call it to compute the start and stop values
 *  of its contiguous 'chunk' of a parallel loop's iterations.
 *
 *  This header file makes the function easy to reuse.
 *
 * Joel Adams, Calvin University, Fall 2023.
 */

#ifndef CHUNKS_H
#define CHUNKS_H

#include <stdio.h>  // printf()
#include <stdlib.h> // exit()
#include <math.h>   // ceil()

/* Calculate the start and stop values for this PE's 
 *  contiguous chunk of a range of values, 0..N-1,
 *  so that each process chunk-sizes are equal (or nearly so).
 *
 * @param: id, an int
 * @param: P, an int
 * @param: N, a const unsigned
 * Precondition: id == this process's MPI rank or thread ID.
 *            && P == the number of PEs (threads or MPI processes)
 *            && N == the total number of 0-based loop iterations needed
 *            && P <= N 
 *            && N < 2^32
 * @param: start, the address of the unsigned variable through which the 
 *          starting value of this process's chunk should be returned
 * @param: stop, the address of the unsigned variable through which the
 *          stopping value of this process's chunk should be returned
 * Postcondition: *start = this process's first iteration value 
 *             && *stop = this process's last iteration value + 1.
 */
void getChunkStartStopValues(int id, int P, const unsigned N,
                              unsigned* start, unsigned* stop) {

   // compute the chunk size that works in many cases
   unsigned chunkSize1 = (long)ceil(((double)N) / P);
   unsigned begin = id * chunkSize1;
   unsigned end = begin + chunkSize1;

   // see if there are any leftover iterations
   unsigned remainder = N % P;

   // If remainder == 0, chunkSize1 = chunk-size for all processes;
   // If remainder != 0, chunkSize1 = chunk-size for p_0..p_remainder-1
   //   but for processes p_remainder..p_numProcs-1
   //   recompute begin and end using a smaller-by-1 chunk size, chunkSize2.
   if (remainder > 0 && id >= remainder) {
     unsigned chunkSize2 = chunkSize1 - 1;
     unsigned remainderBase = remainder * chunkSize1;
     unsigned processOffset = (id-remainder) * chunkSize2;
     begin = remainderBase + processOffset;
     end = begin + chunkSize2;
   } 

   // pass back this PE's begin and end values via start and stop
   *start = begin;
   *stop = end;
}

/*
 * For 2D grids:
 *  Obtain the starting and stopping row values for each thread's
 *  block of rows when using blocking in trng's
 *  assignment of random numbers to threads.
 *
 * Libby Shoop, Macalester Colege, Fall 2025
 *  with special thanks to Rocky Slaymaker for inspiration
 *
 * 
 *  @param: tid: thread ID
 *  @param: numThreads: total number of threads
 *  @param: length: total number of rows in the 2D grid
 *  @param: startRow: pointer to store starting row int
 *  @param: endRow: pointer to store ending row int
 * 
 *  trng's jump() function can be used to skip ahead
 *  in the random number sequence, but it requires
 *  the number of random numbers to skip. This function
 *  helps compute that value for each thread.
 *
 *  Note: length is the number of rows, not including
 *  ghost rows. However, the end row value
 *  returned is still relative to the full grid with
 *  ghost rows. This end row value is then used in
 *  the nested for loop in calcNewGrid.cpp.
 * 
  */

void getStartStopRow(int tid, int numThreads, int length, int *startRow, int *endRow) {
    int rowsPerThread = length / numThreads;
    int extraRows = length % numThreads;

    if (tid < extraRows) {
        *startRow = tid * (rowsPerThread + 1);
        *endRow  = *startRow + rowsPerThread +1;
    } else {
        *startRow = (tid * rowsPerThread) + extraRows;
        * endRow = *startRow + rowsPerThread;
    }
    
}

#endif

