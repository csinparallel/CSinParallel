#include <stdio.h>
#include <omp.h>

#define COLLECTION_SIZE 32

int main( ) {
    int i, min;

    int Collection[COLLECTION_SIZE]={18, 83, 80, 12, 86, 66, 68, 41, 91, 84, 57, 93, 67, 6, 50, 75, 58, 85, 45, 96, 72, 33, 77, 48, 73, 10, 99, 29, 19, 65, 26, 25};

    omp_set_num_threads(4);

    /* 1. Initialize min to first element of collection */
    min=Collection[0];

    /* 2. Compare minimum to each element in collection. */
    #pragma omp parallel for reduction(min:min)
    for( i = 0; i < COLLECTION_SIZE; i++){

        /* 2.a If element is less than minimum, set minimum to element */
        if( Collection[i] < min ){
            min = Collection[i];
        }

    }

    /* 3. Print minimum value */
    printf("The minimum value in the collection is: %d\n", min);
}