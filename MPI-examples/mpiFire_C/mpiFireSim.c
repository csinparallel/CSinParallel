/* mpiFireSim.c
 * Libby Shoop
 *
 * Show the results of running many trials of a burning forest
 * of a given size over a range of probability of spread values.
 * Split the work using MPI processes to complete a portion of the trials.
 *
 * Usage, with default values in ():
 *
 * mpiFireSim [forestSize(20)] [numTrials(100)] [numProbabilities(101)] [showGraph(1)]
 *
 * The forest is simulated as forestSize*forestSize trees.
 * numTrials is the number of simulations of one fire to run,
 *    over a set of probability thresholds.
 * numProbabilities is the number of proability threshold values
 *    to use, ranging from 0.0 to 1.0. Using 101 means the values
 *    will increase by .01, and 101 single fire simulations will be
 *    run for every trial.
 * showGraph: non-zero value means two gnuplot graphs will display
 *    results for 1) average percent burned and 2) average number of
 *    iterations of the fire simulation before it burns out,
 *    over all the trials at each probability threshold.
 *
 */
#include <stdio.h>
#include <stdlib.h>

// code to simulate burning of each fire at some probability of
// spread threshold
#include "fire_functions.h"

// functions from fireSimPlot.c to display results
#include "fireSimPlot.h"

#include "mpi.h"

#define true 1
#define false 0

// #define DEBUG 1

int main(int argc, char **argv)
{
  // initial conditions and variable definitions
  int forest_size = 20;
  double prob_min = 0.0;
  double prob_max = 1.0;
  double prob_step;
  // arrays for data
  int **forest;
  double *percent_burned; // sum of computed value for all trials
  //    int * num_iterations; // how many steps before fire burns out
  double *avg_iterations; // average num_iterations over trials

  int i_trial;
  int n_trials = 100;
  int i_prob;
  int n_probs = 101;
  int do_display = 1;

  // new values for MPI
  int numtasks;      // total processes
  int taskid;        // id of task running a process
  int numworkers;    // number of tasks that will report to conductor
  MPI_Status status; // status of MPI send/recv

  // arrays the conductor uses
  double *prob_spread;     // for x axis display
  double *burn_recvBuffer; // for data received from each worker
  double *iter_recvBuffer; // for data received from each worker

  /* First, find out my taskid and how many tasks are running */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  numworkers = numtasks - 1;

  if (numworkers < 1)
  {
    printf("Please use at least two processes. Quitting.\n");
    exit(1);
  }

  // check command line arguments
  // note  each process does this
  if (argc > 1)
  {
    sscanf(argv[1], "%d", &forest_size);
  }
  if (argc > 2)
  {
    sscanf(argv[2], "%d", &n_trials);
  }
  if (argc > 3)
  {
    sscanf(argv[3], "%d", &n_probs);
  }
  if (argc > 4)
  {
    sscanf(argv[4], "%d", &do_display);
  }
  if (do_display != 0)
    do_display = 1;

  // all processes have arrays to hold data calculations
  percent_burned = (double *)malloc(n_probs * sizeof(double));
  //    num_iterations = (int *) malloc (n_probs*sizeof(int));
  avg_iterations = (double *)malloc(n_probs * sizeof(double));
  for (i_prob = 0; i_prob < n_probs; i_prob++)
  {
    percent_burned[i_prob] = 0.0;
    //      num_iterations[i_prob] = 0;
    avg_iterations[i_prob] = 0.0;
  }

  // for timing
  double startTime, finishTime;

  startTime = MPI_Wtime();

  /////////////////////////////////////////////////////////////////
  // coordinator/conductor tells each worker how many trials to do
  // and waits for result from each worker.
  /////////////////////////////////////////////////////////////////
  if (taskid == 0)
  {
    // conductor determines number of trials for each worker process,
    // ensuring work for each is no more than 1 above any other
    //
    int trunc_num_trials = n_trials / numworkers;
    int extra = n_trials % numworkers;
    int trials_per_worker;
    for (int id = 1; id <= numworkers; id++)
    {
      if (id <= extra)
      {
        trials_per_worker = trunc_num_trials + 1;
      }
      else
      {
        trials_per_worker = trunc_num_trials;
      }
      // send number of trials to worker i
      MPI_Send(&trials_per_worker, 1, MPI_INT, id, 1, MPI_COMM_WORLD);
    }

    // burn_recvBuffer for holding percent burned
    // data received from each worker
    burn_recvBuffer = (double *)malloc(n_probs * sizeof(double));
    for (i_prob = 0; i_prob < n_probs; i_prob++)
    {
      burn_recvBuffer[i_prob] = 0.0;
    }
    // iter_recvBuffer for holding average number of iterations
    // data received from each worker
    iter_recvBuffer = (double *)malloc(n_probs * sizeof(double));
    for (i_prob = 0; i_prob < n_probs; i_prob++)
    {
      iter_recvBuffer[i_prob] = 0.0;
    }

    // prob_spread used for drawing graph
    // x axis is proability threshold values in this array
    prob_spread = (double *)malloc(n_probs * sizeof(double));
    prob_step = (prob_max - prob_min) / (double)(n_probs - 1);
    for (i_prob = 0; i_prob < n_probs; i_prob++)
    {
      prob_spread[i_prob] = prob_min + (double)i_prob * prob_step;
    }

    // conductor waits to receive completed work from each worker task
    // conductor sums values as received from each worker
    for (int i = 1; i <= numworkers; i++)
    {

      MPI_Recv(burn_recvBuffer, //  msg received
               n_probs,         //  buffer size
               MPI_DOUBLE,      //  type
               MPI_ANY_SOURCE,  //  sender (anyone)
               1,               //  tag for percent_burned
               MPI_COMM_WORLD,  //  communicator
               &status);        //  recv status

      MPI_Recv(iter_recvBuffer, //  msg received
               n_probs,         //  buffer size
               MPI_DOUBLE,      //  type
               MPI_ANY_SOURCE,  //  sender (anyone)
               2,               //  tag for avg iterations
               MPI_COMM_WORLD,  //  communicator
               &status);        //  recv status

#ifdef DEBUG
      printf("conductor received percent burned data %d\n", i);
      printf("prob      burn_recv       burn_sum     iter_recv      iter_sum\n");
#endif

      // note that the order of which worker an array was received
      // from is not important, as long as we get all of each array
      for (i_prob = 0; i_prob < n_probs; i_prob++)
      {
        percent_burned[i_prob] += burn_recvBuffer[i_prob];
        avg_iterations[i_prob] += iter_recvBuffer[i_prob];
#ifdef DEBUG
        printf("%f   %f    %f\n",
               prob_spread[i_prob],
               burn_recvBuffer[i_prob], percent_burned[i_prob],
               iter_recvBuffer[i_prob], avg_iterations[i_prob]);
#endif
      }
    }

    // after receiving all data, then compute final averages
    for (i_prob = 0; i_prob < n_probs; i_prob++)
    {
      percent_burned[i_prob] /= numworkers;
      avg_iterations[i_prob] /= numworkers;
#ifdef DEBUG
      printf("%f   %f     %f\n",
             prob_spread[i_prob],
             percent_burned[i_prob], avg_iterations[i_prob]);
#endif
    }

    finishTime = MPI_Wtime();
    printf("Total running time of process %d : %f seconds\n",
           taskid, finishTime - startTime);

    /////////////////////////////////////////////////////////////////
  }
  else
  { // workers do the fire simulation trials ////////////////
    /////////////////////////////////////////////////////////////////
    // setup problem
    // each worker process does this- to start random number
    // at different seed add taskid into the seed value by replacing 0
    seed_by_time(taskid);

    // each worker task will have a forest
    forest = allocate_forest(forest_size);

    // if worker, recv the number of trials it is to do
    int my_n_trials = 0;
    MPI_Recv(&my_n_trials, 1, MPI_INT, 0, 1,
             MPI_COMM_WORLD, &status);

    // Useful to see how work is partitioned. Comment out if you wish.
    printf("worker %d will do %d trials over %d probs\n",
           taskid, my_n_trials, n_probs);

    // worker, do your stuff...

    double next_prob = 0.0; // holds next probablility threshold
    // value of spacing between probability thresholds
    prob_step = (prob_max - prob_min) / (double)(n_probs - 1);

    for (i_trial = 0; i_trial < my_n_trials; i_trial++)
    {
      next_prob = prob_min; // reset per trial as precaution

      // For a number of trials, calculate average
      // percent burn per each probability of spread.
      // Note the loops are swapped comared to sequential version.
      for (i_prob = 0; i_prob < n_probs; i_prob++)
      {
        next_prob = prob_min + (double)i_prob * prob_step;

        // burn until fire is gone; add results to cumlative arrays
        avg_iterations[i_prob] +=
            burn_until_out(forest_size, forest,
                           next_prob,
                           forest_size / 2, forest_size / 2);

        percent_burned[i_prob] += get_percent_burned(forest_size, forest);
        ;
      }
    }

    // note difference from sequential version when loops swap
    // calculate averages when trials are over
    for (i_prob = 0; i_prob < n_probs; i_prob++)
    {

      // avg percent burned. Note using my_n_trials
      percent_burned[i_prob] /= my_n_trials;

      // avg number of iterations
      avg_iterations[i_prob] /= my_n_trials;
    }

    // worker sends data back to conductor
    MPI_Send(percent_burned,  //  msg sent
             n_probs,         //  num chars + NULL
             MPI_DOUBLE,      //  type
             0,               //  destination is conductor
             1,               //  tag for percent_burned
             MPI_COMM_WORLD); //  communicator

    MPI_Send(avg_iterations,  //  msg sent
             n_probs,         //  num chars + NULL
             MPI_DOUBLE,      //  type
             0,               //  destination is conductor
             2,               //  tag for avg iterations
             MPI_COMM_WORLD); //  communicator

    // end of worker's main work
    finishTime = MPI_Wtime();
    printf("Total running time of process %d : %f seconds\n",
           taskid, finishTime - startTime);
  }

  // if conductor, plot graph
  // Note there is is just debug  printing because if you choose not
  // to display this, you my be timing the code to this point.
  if (taskid == 0)
  {
// #ifdef DEBUG
    for (i_prob = 0; i_prob < n_probs; i_prob++)
    {
      printf("%f   %f     %f\n",
             prob_spread[i_prob], percent_burned[i_prob],
             avg_iterations[i_prob]);
    }
// #endif

    // draw graph using gnuplot_i.c functions
    if (do_display == 1)
    {
      drawSimGraphs(forest_size, n_probs, n_trials,
                    prob_spread, percent_burned,
                    avg_iterations);
    }

  } // end of graphs displayed by conductor task

  //////////  clean up ///////////////////////////////
  // Note we distinguish which is conductor data,
  // which is worker data, and which were used in both
  ////////////////////////////////////////////////////
  // used in both
  free(percent_burned);
  free(avg_iterations);

  if (taskid == 0)
  { // conductor
    free(prob_spread);
    free(burn_recvBuffer);
    free(iter_recvBuffer);
  }
  else
  { // worker
    delete_forest(forest_size, forest);
  }

  MPI_Finalize();
  return 0;
}
