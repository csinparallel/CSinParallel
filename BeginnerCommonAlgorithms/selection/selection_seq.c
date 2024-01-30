#include <stdio.h>

#define COLLECTION_SIZE 32

int main( ) {
    int min, time_steps=0;

    int Collection[COLLECTION_SIZE]={18, 83, 80, 12, 86, 66, 68, 41, 91, 84, 57, 93, 67, 6, 50, 75, 58, 85, 45, 96, 72, 33, 77, 48, 73, 10, 99, 29, 19, 65, 26, 25};

    /* 1. Initialize min to first element of collection */
    min=Collection[0];

    /* 2. Compare minimum to each element in collection. */
    for( int i = 0; i < COLLECTION_SIZE; i++){
        /* increment time_step each computation/iteration */
        time_steps++;

        /* If element is less than minimum, set minimum to element */
        if( Collection[i] < min ){
            min = Collection[i];
        }
    }

    /* 3. Print minimum value */
    printf("The minimum value in the collection is: %d\n", min);
    printf("It took %d 'time steps' to process %d elements in the collection.\n", time_steps, COLLECTION_SIZE);
}