#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define COLLECTION_SIZE 32

int main(int argc, char **argv)
{
    int i;
    int lsize;
    int min;
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
    min=lcollection[0];

    // 3. Each task compares its local minimum to each element in its local collection.
    for( i = 0; i < lsize; i++){
        // 3.a If element is less than minimum, set minimum to element
        if( lcollection[i] < min ){
            min = lcollection[i];
        }
    }

    // 4. Collect all local minima
    int *lmins = (int *)malloc(sizeof(char) * world_size);
    MPI_Allgather(&min, 1, MPI_INT, lmins, 1, MPI_INT, MPI_COMM_WORLD);
                                                                                                                                    
    // 5. Find the global minimum from the local minima
    if (world_rank == 0) {
        min=lmins[0];
        for( i = 0; i < world_size; i++){
            if( lmins[i] < min ){
                min = lmins[i];
            }
        }
    

        // 6. Print global minimum value */
        printf("The minimum value in the collection is: %d\n", min);
    }

    // Clean up
    free(lcollection);
    free(lmins);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
   