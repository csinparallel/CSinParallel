/*
 * mergeSort.c
 * ...illustrates parallel merge sort in MPI.
 *
 * Hannah Sonsalla, Macalester College 2017
 *
 * Usage: mpirun -np N ./mergeSort <arraySize>
 *  - arraySize must be a multiple of N
 *  - N must be postive and a power of 2
 *
 * Notes:
 *  - To view initial unsorted array uncomment line A
 *  - To view local arrays of processes before sorting uncomment line B
 *  - To view final sorted array uncomment line C
 *
 */

#include <mpi.h>     // MPI
#include <stdio.h>   // printf
#include <stdlib.h>  // malloc, free, rand(), srand()
#include <time.h>    // time for random generator
#include <math.h>    // log2
#include <string.h>  // memcpy
#include <limits.h>  // INT_MAX

/* Declaration of functions */
void powerOfTwo(int id, int numberProcesses);
void getInput(int argc, char* argv[], int id, int numProcs, int* arraySize);
void fillArray(int array[], int arraySize, int id);
void printList(int id, char arrayName[], int array[], int arraySize);
int compare(const void* a_p, const void* b_p);
int* merge(int half1[], int half2[], int mergeResult[], int size);
int* mergeSort(int height, int id, int localArray[], int size, MPI_Comm comm, int globalArray[]);

/*-------------------------------------------------------------------
 * Function:   powerOfTwo
 * Purpose:    Check number of processes, if not power of 2 prints message
 * Params:     id, rank of the current process
 * 			   numberProcesses, number of processes
 */

void powerOfTwo(int id, int numberProcesses) {
	int power;
	power = (numberProcesses != 0) && ((numberProcesses & (numberProcesses - 1)) == 0);
	if (!power) {
		if (id == 0) printf("number of processes must be power of 2 \n");
		MPI_Finalize();
		exit(-1);
	}
}

/*-------------------------------------------------------------------
 * Function:   getInput
 * Purpose:    Get input from user for array size
 * Params:     argc, argument count
 * 			   argv[], points to argument vector
 *     		   id, rank of the current process
 * 			   numProcs, number of processes
 * 			   arraySize, points to array size
 */

void getInput(int argc, char* argv[], int id, int numProcs, int* arraySize){
    if (id == 0){
        if (id % 2 != 0){
			fprintf(stderr, "usage: mpirun -n <p> %s <size of array> \n", argv[0]);
            fflush(stderr);
            *arraySize = -1;
        } else if (argc != 2){
            fprintf(stderr, "usage: mpirun -n <p> %s <size of array> \n", argv[0]);
            fflush(stderr);
            *arraySize = -1;
        } else if ((atoi(argv[1])) % numProcs != 0) {
		    fprintf(stderr, "size of array must be divisible by number of processes \n");
            fflush(stderr);
            *arraySize = -1;
		} else {
            *arraySize = atoi(argv[1]);
        }
    }
    // broadcast arraySize to all processes
    MPI_Bcast(arraySize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // negative arraySize ends the program
    if (*arraySize <= 0) {
        MPI_Finalize();
        exit(-1);
    }
}

/*-------------------------------------------------------------------
 * Function:   fillArray
 * Purpose:    Fill array with random integers
 * Params:     array, the array being filled
 *   		   arraySize, size of the array
 *     		   id, rank of the current process
 */

void fillArray(int array[], int arraySize, int id) {
	int i;
	// use current time as seed for random generator
	srand(id + time(0));
	for (i = 0; i < arraySize; i++) {
		array[i] = rand() % 100; //INT_MAX
	}
}

/*-------------------------------------------------------------------
 * Function:   printList
 * Purpose:    Prints the contents of a given list of a process
 * Params:     id, rank of the current process
 * 			   arrayName, name of array
 *     		   array, array to print
 *   		   arraySize, size of array
 */

void printList(int id, char arrayName[], int array[], int arraySize) {
    printf("Process %d, %s: ", id, arrayName);
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
 * Function:    merge
 * Purpose:     Merges half1 array and half2 array into mergeResult
 * Params:      half1, first half of array to merge
 *   			half2, second half of array to merge
 * 				mergeResult, array to store merged result
 * 				size, size of half1 and half2
 */
int* merge(int half1[], int half2[], int mergeResult[], int size){
    int ai, bi, ci;
    ai = bi = ci = 0;
    // integers remain in both arrays to compare
    while ((ai < size) && (bi < size)){
        if (half1[ai] <= half2[bi]){
			mergeResult[ci] = half1[ai];
			ai++;
		} else {
			mergeResult[ci] = half2[bi];
			bi++;
		}
			ci++;
	}
	// integers only remain in rightArray
	if (ai >= size){
        while (bi < size) {
			mergeResult[ci] = half2[bi];
			bi++; ci++;
		}
	}
	// integers only remain in localArray
	if (bi >= size){
		while (ai < size) {
			mergeResult[ci] = half1[ai];
			ai++; ci++;
		}
	}
	return mergeResult;
}

/*-------------------------------------------------------------------
 * Function:    mergeSort
 * Purpose:     implements merge sort: merges sorted arrays from 
 *              processes until we have a single array containing all 
 *  			integers in sorted order
 * Params:		height, height of merge sort tree
 * 				id, rank of the current process
 * 				localArray, local array containing integers of current process
 * 				size, size of localArray on current process
 * 				comm, MPI communicator
 * 				globalArray, globalArray contains either all integers
 *    				if process 0 or NULL for other processes
 */

int* mergeSort(int height, int id, int localArray[], int size, MPI_Comm comm, int globalArray[]){
    int parent, rightChild, myHeight;
    int *half1, *half2, *mergeResult;

    myHeight = 0;
    qsort(localArray, size, sizeof(int), compare); // sort local array
    half1 = localArray;  // assign half1 to localArray
	
    while (myHeight < height) { // not yet at top
        parent = (id & (~(1 << myHeight)));

        if (parent == id) { // left child
		    rightChild = (id | (1 << myHeight));

  		    // allocate memory and receive array of right child
  		    half2 = (int*) malloc (size * sizeof(int));
  		    MPI_Recv(half2, size, MPI_INT, rightChild, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  		    // allocate memory for result of merge
  		    mergeResult = (int*) malloc (size * 2 * sizeof(int));
  		    // merge half1 and half2 into mergeResult
  		    mergeResult = merge(half1, half2, mergeResult, size);
  		    // reassign half1 to merge result
            half1 = mergeResult;
			size = size * 2;  // double size
			
			free(half2); 
			mergeResult = NULL;

            myHeight++;

        } else { // right child
			  // send local array to parent
              MPI_Send(half1, size, MPI_INT, parent, 0, MPI_COMM_WORLD);
              if(myHeight != 0) free(half1);  
              myHeight = height;
        }
    }

    if(id == 0){
		globalArray = half1;   // reassign globalArray to half1
	}
	return globalArray;
}

/*-------------------------------------------------------------------*/

int main(int argc, char** argv) {
    int numProcs, id, globalArraySize, localArraySize, height;
    int *localArray, *globalArray;
    double startTime, localTime, totalTime;
    double zeroStartTime, zeroTotalTime, processStartTime, processTotalTime;;
    int length = -1;
    char myHostName[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Get_processor_name (myHostName, &length); 

    // check for odd processes
    powerOfTwo(id, numProcs);

    // get size of global array
    getInput(argc, argv, id, numProcs, &globalArraySize);

    // calculate total height of tree
    height = log2(numProcs);

    // if process 0, allocate memory for global array and fill with values
    if (id==0){
		globalArray = (int*) malloc (globalArraySize * sizeof(int));
		fillArray(globalArray, globalArraySize, id);
		//printList(id, "UNSORTED ARRAY", globalArray, globalArraySize);  // Line A
	}
	
    // allocate memory for local array, scatter to fill with values and print
    localArraySize = globalArraySize / numProcs;
    localArray = (int*) malloc (localArraySize * sizeof(int));
    MPI_Scatter(globalArray, localArraySize, MPI_INT, localArray, 
		localArraySize, MPI_INT, 0, MPI_COMM_WORLD);
    //printList(id, "localArray", localArray, localArraySize);   // Line B 
    
    //Start timing
    startTime = MPI_Wtime();
    //Merge sort
    if (id == 0) {
		zeroStartTime = MPI_Wtime();
		globalArray = mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, globalArray);
		zeroTotalTime = MPI_Wtime() - zeroStartTime;
		printf("Process #%d of %d on %s took %f seconds \n", 
			id, numProcs, myHostName, zeroTotalTime);
	}
	else {
		processStartTime = MPI_Wtime();
	        mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, NULL);
		processTotalTime = MPI_Wtime() - processStartTime;
		printf("Process #%d of %d on %s took %f seconds \n", 
			id, numProcs, myHostName, processTotalTime);
	}
    //End timing
    localTime = MPI_Wtime() - startTime;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE,
        MPI_MAX, 0, MPI_COMM_WORLD);

    if (id == 0) {
		//printList(0, "FINAL SORTED ARRAY", globalArray, globalArraySize);  // Line C
		printf("Sorting %d integers took %f seconds \n", globalArraySize,totalTime);
		free(globalArray);
	}

    free(localArray);  
    MPI_Finalize();
    return 0;
}
