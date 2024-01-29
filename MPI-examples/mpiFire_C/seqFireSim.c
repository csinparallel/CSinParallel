/* seqFireSim.c
 *
 * original code by David Joiner
 * updated by Libby Shoop for gnuplot display and clarifying comments
 *
 * Show the results of running many trials of a burning forest
 * of a given size over a range of probability of spread threshold values.
 *
 * Usage, with default values in ():
 *
 *  ./seqFireSim [forestSize(20)] [numTrials(100)] [numProbabilities(101)] [showGraph(1)]
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
#include "mpi.h" // use MPI timing

// code to simulate burning of each fire at some probability of
// spread threshold
#include "fire_functions.h"

// functions from fireSimPlot.c to display results
#include "fireSimPlot.h"

#define true 1
#define false 0

// uncomment and use small values for trials, numPrababilities to
// get more information.
// #define DEBUG 1

int main(int argc, char **argv)
{
    // initial conditions and variable definitions
    int forest_size = 20;
    double *prob_spread;
    double prob_min = 0.0;
    double prob_max = 1.0;
    double prob_step;
    int **forest;
    double *percent_burned; // sum/avg of computed value for all trials
    int *num_iterations;    // how many steps before fire burns out
    double *avg_iterations; // average num_iterations over trials

    int i_trial;
    int n_trials = 100;
    int i_prob;
    int n_probs = 101;
    int do_display = 1;

    // check command line arguments

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

    // so we can use MPI timinng function with MPICH
    int numtasks; // total processes
    int taskid;   // id of task running a process
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    if (numtasks > 1 && taskid == 0)
    {
        printf("Please use one process for this sequential version. Quitting.\n");
        exit(1);
    }
    // for timing
    double startTime, finishTime;

    startTime = MPI_Wtime();

    // setup problem
    seed_by_time(0);
    forest = allocate_forest(forest_size);
    prob_spread = (double *)malloc(n_probs * sizeof(double));
    percent_burned = (double *)malloc(n_probs * sizeof(double));
    num_iterations = (int *)malloc(n_probs * sizeof(int));
    avg_iterations = (double *)malloc(n_probs * sizeof(double));

    // for a number of probabilities, calculate
    // average burn and output
    prob_step = (prob_max - prob_min) / (double)(n_probs - 1);

    if (!do_display)
    {
        printf("Probability of fire spreading, Average percent burned\n");
    }

    for (i_prob = 0; i_prob < n_probs; i_prob++)
    {
        num_iterations[i_prob] = 0;
        // for a number of trials, calculate average
        // percent burn
        prob_spread[i_prob] = prob_min + (double)i_prob * prob_step;
        percent_burned[i_prob] = 0.0;
        for (i_trial = 0; i_trial < n_trials; i_trial++)
        {
            // burn until fire is gone
            num_iterations[i_prob] +=
                burn_until_out(forest_size, forest,
                               prob_spread[i_prob],
                               forest_size / 2, forest_size / 2);
            percent_burned[i_prob] += get_percent_burned(forest_size, forest);

            
        }

        // avg percent burned
        percent_burned[i_prob] /= n_trials;
        // avg number of iterations
        avg_iterations[i_prob] = num_iterations[i_prob] / n_trials;
        
    }

    // Note there is is just debug  printing because if you choose not
    // to display the graph, you my be timing the code to this point.
    if (!do_display)
    {
#ifdef DEBUG
        for (i_prob = 0; i_prob < n_probs; i_prob++)
        {
            printf("%f   %f     %f\n",
                   prob_spread[i_prob], percent_burned[i_prob],
                   avg_iterations);
        }
#endif
    }

    finishTime = MPI_Wtime();
    printf("Total running time of process: %f seconds\n",
           finishTime - startTime);

    // plot graph
    if (do_display == 1)
    {
        drawSimGraphs(forest_size, n_probs, n_trials,
                      prob_spread, percent_burned,
                      avg_iterations);
    }

    // clean up
    delete_forest(forest_size, forest);
    free(prob_spread);
    free(percent_burned);
    free(num_iterations);
    free(avg_iterations);

    // Just for the timing function
    MPI_Finalize();
    return 0;
}
