#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define COLLECTION_SIZE 32

int main(int argc, char **argv)
{
    int i;
    int lsize;
    int lmin, gmin;
    int world_rank, world_size;

    /* PREPARATIONS */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int Collection[COLLECTION_SIZE]={18, 83, 80, 12, 86, 66, 68, 41, 91, 84, 57, 93, 67, 6, 50, 75, 58, 85, 45, 96, 72, 33, 77, 48, 73, 10, 99, 29, 19, 65, 26, 25};

    /* 1. Divide Collection amongst tasks */
    /* Compute size of local collections */
    lsize = COLLECTION_SIZE / world_size;

    // For each process, create a buffer for local collection
    int *lcollection = (int *)malloc( sizeof(int) * lsize );

    // Scatter collection from root process to all others
    MPI_Scatter(Collection, lsize, MPI_INT, lcollection, lsize, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Initialize each task's local minimum
    lmin=lcollection[0];

    // 3. Each task compares its local minimum to each element in its local collection.
    for( i = 0; i < lsize; i++){
        
        // 3.a If element is less than minimum, set minimum to element
        if( lcollection[i] < lmin ){
            lmin = lcollection[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 4. Simultaneously, collect all local minima and find the global minimum from the local minima
    //    Replaces steps 4 and 5 in previous MPI solution
    MPI_Reduce(&lmin, &gmin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);


    // 5. Print global minimum value */
    if (world_rank ==0) {
        printf("The minimum value in the collection is: %d\n", gmin);
    }

    // Clean up
    free(lcollection);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}