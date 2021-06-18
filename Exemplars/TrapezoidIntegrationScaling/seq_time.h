/*
 * Use the high-resolution clock measurement provided in time.h of C
 * to return a current clock time as a double in seconds,
 * with as close to nanosecond precision as the hardware can provide.
 *
 * This is designed to be used to obtain a delta elapsed time
 * between two points in a sequential program., like this:
 *
 *  dounble start, end;
 *  start = c_get_wtime();
 *  // do some work
 *  end = c_get_wtime();
 *  printf("time taken: %f seconds\n", end - start);
 *
 * Author: Libby Shoop
 */

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

// Note: this is the struct in C that holds time info:
//
// struct timespec {
//    time_t   tv_sec;        /* seconds */
//    long     tv_nsec;       /* nanoseconds */
// };
//

/**
 *  Return current wall clock time in seconds, with a
 *  precision of nanoseconds (one billionth of a second).
 *
 *  Technically, it is the time since 'the epoch' in unix
 *  (00:00 1 January, 1970 UTC).
 */
double c_get_wtime() {
    struct timespec start;
//     clock_gettime(CLOCK_REALTIME, &start); // could go backward
    clock_gettime(CLOCK_MONOTONIC, &start); //recommended for elapsed time
    long seconds = start.tv_sec;
    long nanoseconds = start.tv_nsec;

    double wtime = (double)seconds + (double)nanoseconds/(double)1000000000;
    return wtime;
}
