#
# Run multiple simulations of a fire burning at several probability thresholds.
#
# Ported to python from the original Shodor foundation example:
#  https://www.shodor.org/refdesk/Resources/Tutorials/BasicMPI/
#
# Libby Shoop     Macalester College
#
import argparse      # for command-line arguments
import matplotlib.pyplot as plt
import math
import time

from fire_functions import *
from sim_functions import *


############################# main() ##########################
def main():
    start = time.process_time()  # start the timing

    row_size, prob_spread_increment, num_trials = parseArguments()

    # determine how many probabilities between .1 up to but not including 1.0
    # will be tried, based on increment given on command line.
    tot_prob_trials = int(math.ceil((1.0 - 0.1)/prob_spread_increment))
    # Note: num_trials simulations will be run, where an iteration
    #       will be for tot_prob_trials.
    #
    # print("total probability trials: {}".format(tot_prob_trials))  #debug
    # set up result data arrays to hold:
    #   sums of each value computed for each probability while iterating
    percent_burned_data = np.zeros( (tot_prob_trials, 2) )
    iters_per_sim_data = np.zeros( (tot_prob_trials, 2) )

    # The primary work: run the trials using each set of proabilities.
    # There will be num_trials x tot_prob_trials individual fire simulations
    # run, each with a new forest.
    for i in range(num_trials):
        idx = 0     # index into result data array
        for prob_spread in np.arange(0.1, 1.0, prob_spread_increment):
            forest = initialize_forest(row_size)
            iter, percent_burned = burn_until_out(row_size, forest, prob_spread)

            if i == 0: #put proability for x axis in result once
                percent_burned_data[(idx,0)] = prob_spread
                iters_per_sim_data[(idx,0)] = prob_spread
            #add data for this trial to running total for each trial
            percent_burned_data[(idx,1)] += percent_burned
            iters_per_sim_data[(idx,1)] += iter

            idx += 1

    # find average percent burned and number of iterations
    # for each probability threashold
    for row in range(tot_prob_trials):
        percent_burned_data[(row,1)] = percent_burned_data[(row,1)]/num_trials
        iters_per_sim_data[(row,1)] = iters_per_sim_data[(row,1)]/num_trials

    finish = time.process_time()  # end the timing
    total_time = finish - start
    print("Running time: {0:12.4f} seconds".format(total_time))

    # Create a figure with 2 plots of the simulation results
    upper_title = "Simulation: {0} trials for each probability\n {1}x{1} forest\nRun time on 1 process: {2:12.4f} seconds"
    upper_title = upper_title.format(num_trials, row_size, total_time)

    # 2 plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0.24}, figsize=(8.5,6))
    fig.canvas.set_window_title("Forest Fire Simulation with Multiple Trials (1 process)")
    fig.suptitle(upper_title)

    ax1.plot(percent_burned_data[:,0], percent_burned_data[:,1], linestyle='-', marker='.', color='tab:blue')
    ax1.set(xlabel="Probability threshold", ylabel="Avg percent burned")

    ax2.plot(iters_per_sim_data[:,0], iters_per_sim_data[:,1], linestyle='-', marker='.', color='xkcd:maroon')
    ax2.set(xlabel="Probability threshold", ylabel="Avg iterations per simulation")

    plt.show()

########## Run the main function
main()
