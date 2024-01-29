/* taskQueue.c
 * ... illustrates how to use the Conductor-Worker and Task Queue
 *     patterns to achieve dynamic load balancing.
 * This code is based on the dynamicLoadBalance.py script,
 * that was written by Libby Shoop (Macalester College)
 * 
 * Joel C. Adams, Calvin University, June 2021.
 *
 * Usage: mpirun -np 4 ./taskQueue
 *   or:  mpirun -np 8 --oversubscribe ./taskQueue
 *        (assuming a quad-core CPU and OpenMPI)
 *
 * Exercise:
 * - Compile and run, varying N = 4, 8.
 * - Explain how this approach balances the work loads dynamically
 */

#include <stdio.h>    // printf()
#include <stdlib.h>   // rand(), srand()
// #include <unistd.h>   // usleep()
#include <time.h>     // nanosleep()
#include <mpi.h>      // MPI functions

const int CONDUCTOR        = 0;
const int WORK_TAG      = 0;
const int DONE_TAG      = 1;

/* generate the task queue
 * @param: numTasks, an int
 * @PRE: numTasks > numProcesses
 * @return: a dynamic array of numTasks randomly generated 
 *           integers from the range MIN_TASK_TIME..MAX_TASK_TIME.
 */
int* generateQueue(int numTasks) {
    const int MIN_TASK_TIME = 100000000;    // nanoseconds
    const int MAX_TASK_TIME = 1000000000;   // nanoseconds

                                          // allocate the dynamic array
    int* taskQ = malloc(numTasks * sizeof(int));

    srand(1000);                          // seed the RNG
    for (int i = 0; i < numTasks; ++i) {  // fill the array 
        taskQ[i] = (rand() % (MAX_TASK_TIME - MIN_TASK_TIME + 1)) 
                    + MIN_TASK_TIME;
        printf("Conductor: added task %d to queue\n", taskQ[i]);
    }
    return taskQ;
}

/* procedure used by the Conductor to hand out tasks...
 * @param: taskQ, an array of integer times
 * @param: numTasks, the length of the array
 * @param: numProcs, the number of MPI processes
 * @POST: all tasks in taskQ have been sent to workers
 *       && a response has been received for each task
 *       && 'done' messages have been sent to each worker.
 */
void handOutTasks(int* taskQ, int numTasks, int numProcs) {
    int taskCount         = 0;
    int recvCount         = 0;
    int workerID          = -1;
    unsigned int task     = -1;
    unsigned int taskTime = -1;
    MPI_Status msgStatus;

    printf("Conductor: sending initial %d tasks\n", numProcs - 1);
    for (int id = 1; id < numProcs; ++id) {
        task = taskQ[taskCount++];
        MPI_Send(&task, 1, MPI_INT, id, WORK_TAG, MPI_COMM_WORLD);
    }

    while (taskCount < numTasks) {
        MPI_Recv(&taskTime, 1, MPI_INT, MPI_ANY_SOURCE,
                   WORK_TAG, MPI_COMM_WORLD, &msgStatus);
        workerID = msgStatus.MPI_SOURCE;
        printf("Conductor: received %d from worker %d\n", taskTime, workerID);
        ++recvCount;
        task = taskQ[taskCount++];
        MPI_Send(&task, 1, MPI_INT, workerID, WORK_TAG, MPI_COMM_WORLD);
    }

    while (recvCount < numTasks) {
        MPI_Recv(&taskTime, 1, MPI_INT, MPI_ANY_SOURCE,
                   WORK_TAG, MPI_COMM_WORLD, &msgStatus);
        workerID = msgStatus.MPI_SOURCE;
        printf("Conductor: received %d from worker %d\n", taskTime, workerID);
        ++recvCount;
    }

    for (int id = 1; id < numProcs; ++id) {
        task = 0;
        MPI_Send(&task, 1, MPI_INT, id, DONE_TAG, MPI_COMM_WORLD);
        printf("Conductor: sent 'done' message to worker %d\n", id);
    }
    printf("Conductor: terminating normally\n");
}

/* procedure performed by Worker processes
 * @param: id, the process's MPI rank
 * @POST: 1 or more tasks have been received from the Conductor process
 *        && performing that task has been simulated 
 *        && a response has been sent to the Conductor
 *        && a 'done' message has been received from the Conductor.
 */
void performWork(int id) {
    unsigned int taskTime = 0;
    int done = 0;
    MPI_Status msgStatus;

    while (!done) {
        MPI_Recv(&taskTime, 1, MPI_INT, CONDUCTOR,
                   MPI_ANY_TAG, MPI_COMM_WORLD, &msgStatus);
        int tag = msgStatus.MPI_TAG;
        if (tag != DONE_TAG) {
            printf("Worker %d: received %d from Conductor\n", id, taskTime);

            // usleep(taskTime);      // simulate performing the task
            struct timespec remaining, request = { 0, taskTime };
           nanosleep(&request, &remaining);
            // sleep(taskTime);
                                   // then notify the Conductor
            MPI_Send(&taskTime, 1, MPI_INT, CONDUCTOR, WORK_TAG, MPI_COMM_WORLD);
        } else {
            done = 1;
        }
    } 
    printf("Worker %d: done\n", id);
}

int main(int argc, char** argv) {
    int numProcesses        = -1;
    int id                  = -1;
    int* taskQueue = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == CONDUCTOR) {
        int numTasks = (numProcesses - 1) * 4;
        taskQueue = generateQueue(numTasks);
        handOutTasks(taskQueue, numTasks, numProcesses);
        free(taskQueue);
    } else {
        performWork(id);
    }

    MPI_Finalize();
}

