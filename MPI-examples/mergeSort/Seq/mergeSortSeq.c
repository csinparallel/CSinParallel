/*
 * mergeSortSeq.c
 * ...illustrates sequential merge sort.
 *
 * Hannah Sonsalla, Macalester College 2017
 *
 * Usage:  ./mergeSort <arraySize>
 *
 * Notes:
 *  - To view initial unsorted array uncomment line A
 *  - To view final sorted array uncomment line B
 */

#include <stdio.h>   // printf
#include <stdlib.h>  // malloc, free, rand(), srand()
#include <time.h>    // time for random generator and timing
#include <string.h>  // memcpy
#include <limits.h>  // INT_MAX

/* Declaration of functions */
void getInput(int argc, char* argv[], int* arraySize);
void fillArray(int array[], int arraySize);
void printList(char arrayName[], int array[], int arraySize);
int compare(const void* a_p, const void* b_p);
int* mergeSort(int array[], int arraySize);



/*-------------------------------------------------------------------
 * Function:   getInput
 * Purpose:    Get input from user for array size
 * Params:     argc, argument count
 * 			   argv[], points to argument vector
 * 			   arraySize, points to array size
 */

void getInput(int argc, char* argv[], int* arraySize){
	if (argc!= 2){
		fprintf(stderr, "usage:  %s <number of tosses> \n", argv[0]);
        fflush(stderr);
        *arraySize = -1;
    } else {
		*arraySize = atoi(argv[1]);
	}

	// 0 totalNumTosses ends the program
    if (*arraySize <= 0) {
        exit(-1);
    }
}

/*-------------------------------------------------------------------
 * Function:   fillArray
 * Purpose:    Fill array with random integers
 * Params:     array, the array being filled
 *   		   arraySize, size of the array
 */

void fillArray(int array[], int arraySize) {
	int i;
	// use current time as seed for random generator
	srand(time(0));
	for (i = 0; i < arraySize; i++) {
		array[i] = rand() % 100; //INT_MAX
	}
}

/*-------------------------------------------------------------------
 * Function:   printList
 * Purpose:    Prints the contents of a given list of a process
 * Params:     arrayName, name of array
 *     		   array, array to print
 *   		   arraySize, size of array
 */

void printList(char arrayName[], int array[], int arraySize) {
    printf(" %s: ", arrayName);
    for (int i = 0; i < arraySize; i++) {
        printf(" %d", array[i]);
    }
    printf("\n");
}

/*-------------------------------------------------------------------
 * Function:    Compare - An Introduction to Parallel Programming by Pacheco
 * Purpose:     Compare 2 ints, return -1, 0, or 1, respectively, when
 *              the first int is less than, equal, or greater than
 *              the second.  Used by qsort.
 */

int compare(const void* a_p, const void* b_p) {
   int a = *((int*)a_p);
   int b = *((int*)b_p);

   if (a < b)
      return -1;
   else if (a == b)
      return 0;
   else /* a > b */
      return 1;
}

/*-------------------------------------------------------------------
 * Function:    mergeSort
 * Purpose:     implements merge sort: merges sorted arrays from 
 *              processes until we have a single array containing all 
 *  			integers in sorted order
 * Params:		array, array on which to perform merge sort
 * 				arraySize, size of array
 */

int* mergeSort(int array[], int arraySize){

    qsort(array, arraySize, sizeof(int), compare); // sort
	return array;
}

/*-------------------------------------------------------------------*/

int main(int argc, char** argv) {
    int arraySize;
    int *array;
    clock_t startTime, endTime;

    // get size of array
    getInput(argc, argv, &arraySize);

    // allocate memory for global array and fill with values
    array = (int*) malloc (arraySize * sizeof(int));
    fillArray(array, arraySize);
    //printList("UNSORTED ARRAY", array, arraySize);  // Line A
    
    //Start timing
    startTime = clock();
    
    //Merge sort
	array = mergeSort(array, arraySize);

    //End timing
    endTime = clock();

    //printList("FINAL SORTED ARRAY", array, arraySize);  // Line B
    printf("Sorting %d integers took %f seconds \n", arraySize, (double)(endTime-startTime)/CLOCKS_PER_SEC);
	free(array);
	
    return 0;
}
