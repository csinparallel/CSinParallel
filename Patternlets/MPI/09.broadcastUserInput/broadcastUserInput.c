/* broadcastUserInput.c
 * ... illustrates the use of MPI_Bcast() with a scalar value
 *     obtained via a command line argument.
 *
 * Hannah Sonsalla, Macalester College 2017
 * Modeled from code by Joel Adams, Calvin College, April 2016.
 * Updated by Libby Shoop, 2023, for simplicity and use in Intermediate book
 *
 * Usage: mpirun -np N ./broadcastUserInput <integer>
 *
 * Exercise:
 * - Compile and run several times varying the number
 *   of processes and integer value
 * - Explain the behavior you observe
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CONDUCTOR 0

int main(int argc, char** argv) {
    int answer = 0, length = 0;
    int myRank = 0;

    char myHostName[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Get_processor_name (myHostName, &length);

    if (myRank == CONDUCTOR){ 
        if (argc == 2){
             answer = atoi(argv[1]);
        }
    }

    printf("BEFORE the broadcast, process %d on host '%s' has answer = %d\n",
             myRank, myHostName, answer);

    MPI_Bcast(&answer, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("AFTER the broadcast, process %d on host '%s' has answer = %d\n",
             myRank, myHostName, answer);

    MPI_Finalize();

    return 0;
}
