/*
 * broadcastSendReceive.c
 * ... illustrates basic send receive functions.
 * Conductor process sends filled array to each process.
 *
 * Hannah Sonsalla, Macalester College 2017
 * fill and print function from code by Joel Adams, Calvin College
 *
 * Usage: mpirun -np N ./broadcastSendReceive
 *
 * Exercise:
 * - Compile and run, using 2, 4, and 8 processes
 * - Use source code to trace execution and output
 * 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* fill an array with some arbitrary values 
 * @param: a, an int*.
 * @param: size, an int.
 * Precondition: a is the address of an array of ints.
 *              && size is the number of ints a can hold.
 * Postcondition: a has been filled with arbitrary values 
 *                { 11, 12, 13, ... }.
 */
void fill(int* a, int size) {
	int i;
	for (i = 0; i < size; i++) {
		a[i] = i+11;
	}
}

/* display a string, a process id, and its array values 
 * @param: str, a char*
 * @param: id, an int
 * @param: a, an int*.
 * Precondition: str points to either "BEFORE" or "AFTER"
 *              && id is the rank of this MPI process
 *              && a is the address of an 8-element int array.
 * Postcondition: str, id, and a have all been written to stdout.
 */
void print(char* str, int id, int* a) {
	printf("%s array sent, process %d has: {%d, %d, %d, %d, %d, %d, %d, %d}\n",
	   str, id, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
}

#define MAX 8

int main(int argc, char** argv) {
	int id = -1, numProcesses = -1;
	int array[MAX] = {0};
    

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
    	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    
	if (id == 0) fill(array, MAX);
     
	print("BEFORE", id, array);
	
	// conductor process sends array to every process
	if (id == 0) {
		for (int i = 1; i < numProcesses; i++) {
			MPI_Send(&array, MAX, MPI_INT, 
			    i, 1, MPI_COMM_WORLD);
	    }
	}
	else {      // worker
	    MPI_Recv(&array, MAX, MPI_INT, 0, 
	        1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	
    	print("AFTER", id, array);
 	MPI_Finalize();

	return 0;
}
