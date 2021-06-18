#
# Run multiple trial simulations of a fire burning at several
# probability thresholds.
# This version distributes the trials across multiple MPI processes.
#
# Ported to python from the original Shodor foundation example:
#  https://www.shodor.org/refdesk/Resources/Tutorials/BasicMPI/
#
# Libby Shoop     Macalester College
#

import matplotlib.pyplot as plt
import math
import time
from mpi4py import MPI

from fire_functions import *
from sim_functions import *


############################# main() ##########################
def main():
    # MPI information
    comm = MPI.COMM_WORLD
    id = comm.Get_rank()            #number of the process running the code
    numProcesses = comm.Get_size()  #total number of processes running
    myHostName = MPI.Get_processor_name()  #machine name running the code

    np.random.seed()    # each process starts with seed from its /dev/urandom

    start = MPI.Wtime() # start the timing
    # master process will get the arguments
    # each process gets sent row_size, prob_spread_increment, and
    # its number of trials to perform (via a broadcast)
    if id == 0:
        # row_size, prob_spread_increment, tot_num_trials
        args = parseArguments()
    else:
        args = None

    # all processes participate in the broadcast
    sim_data = comm.bcast(args, root=0)
    # set the simulation values
    row_size = sim_data[0]
    prob_spread_increment =sim_data[1]
    tot_num_trials = sim_data[2]

    # determine number of trials that each process will do
    # by checking whether trials are divisible by number of processes
    # and if not spreading the work so that some do one extra trial
    remainder = tot_num_trials%numProcesses
    num_trials = int(tot_num_trials/numProcesses)
    if remainder !=0 and id >= numProcesses - remainder:
        num_trials += 1

    # determine how many probabilities between .1 up to but not including 1.0
    # will be tried, based on increment given on cammand line.
    tot_prob_trials = int(math.ceil((1.0 - 0.1)/prob_spread_increment))
    # Note: tot_num_trials simulations will be run, where an iteration
    #       will be for tot_prob_trials.

    # set up result data arrays to hold:
    #   sums of each value computed for each probability while iterating
    percent_burned_data = np.zeros( (tot_prob_trials, 2) )
    iters_per_sim_data = np.zeros( (tot_prob_trials, 2) )
    # master holds arrays for receving data from workers
    if id == 0:
        recv_percent_burned_data = np.zeros( (tot_prob_trials, 2) )
        recv_iters_per_sim_data = np.zeros( (tot_prob_trials, 2) )

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

    # each worker will send its computed data to the master, who receives
    # it in turn from each worker and updates its copy
    if id !=0:
        comm.Send(percent_burned_data, dest=0, tag=1)
        comm.Send(iters_per_sim_data, dest=0, tag=2)
        proc_time = MPI.Wtime() - start
        print("Process {0} ({1}) Running time: {2:12.4f} seconds".format(id,
                myHostName, proc_time))
    else:  #master
        # get each worker's arrays and add the contents to its arrays
        for proc in range(1, numProcesses):
            comm.Recv(recv_percent_burned_data, source=proc, tag=1)
            comm.Recv(recv_iters_per_sim_data, source=proc, tag=2)

            for row in range(tot_prob_trials):
                percent_burned_data[(row,1)] += recv_percent_burned_data[(row,1)]
                iters_per_sim_data[(row,1)] += recv_iters_per_sim_data[(row,1)]
        # determine the averages
        for row in range(tot_prob_trials):
            percent_burned_data[(row,1)] = percent_burned_data[(row,1)]/numProcesses
            iters_per_sim_data[(row,1)] = iters_per_sim_data[(row,1)]/numProcesses
    # barrier here since master waited until all are finished sending

    if id == 0: #master will have overall time and can print the results
        finish = MPI.Wtime()  # end the timing
        total_time = finish - start
        print("Total Running time: {0:12.4f} seconds".format(total_time))

        # Create a figure with 2 plots of the simulation results
        upper_title = "Simulation: {0} trials for each probability\n {1}x{1} forest\nRun time on {2} processes: {3:12.4f} seconds"
        upper_title = upper_title.format(tot_num_trials, row_size,  numProcesses, total_time)

        # 2 plots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0.24}, figsize=(8.5,6))
        fig.canvas.manager.set_window_title("Forest Fire Simulation with Multiple Trials in Parallel")
        print("figure width: {}".format(fig.get_figwidth()))
        # fig.set_figwidth(fig.get_figwidth() + 1.0, forward=True)
        fig.suptitle(upper_title)

        ax1.plot(percent_burned_data[:,0], percent_burned_data[:,1], linestyle='-', marker='.', color='tab:blue')
        ax1.set(xlabel="Probability threshold", ylabel="Avg percent burned")

        ax2.plot(iters_per_sim_data[:,0], iters_per_sim_data[:,1], linestyle='-', marker='.', color='xkcd:maroon')
        ax2.set(xlabel="Probability threshold", ylabel="Avg iterations per simulation")

        plt.show()

########## Run the main function
main()
