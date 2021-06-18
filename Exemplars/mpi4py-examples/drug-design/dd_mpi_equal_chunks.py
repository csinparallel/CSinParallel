# Drug Design Exemplar:
# Equal number of tasks assigned to workers
#
#  This version of the 'drug 'design' program uses the master-worker
#  pattern.  The master generates a list of ligands to be matched to
#  a given protein and scored based on simple matching of letters.
#  Each worker gets assigned an equal number of ligands from the list and
#  computes a score for each one, sending it back to the master.
#  In this version, the master is keeping track of the overall
#  maximum scoring ligands.
#
#  To run a small example:
#        mpirun -np 4 python ./dd_mpi_equal_chunks.py 18 -verbose
#  where  18 is the number of ligands to create
#  To see all the options:
#       python dd_mpi_equal_chunks.py --help

import math
from mpi4py import MPI

# Functions in common between this and the 'dynamic' version
from dd_functions import *


# main program

def main():
    # set up MPI and retrieve basic data
    comm = MPI.COMM_WORLD
    id = comm.Get_rank()            #number of the process running the code
    numProcesses = comm.Get_size()  #total number of processes running
    myHostName = MPI.Get_processor_name()  #machine name running the code

    start = MPI.Wtime() # start timer

    args = getCommandLineArgs()

    if numProcesses <= 1:
        print("Need at least two processes, aborting")
        return

    # if ((args.nLigands%(numProcesses-1)) != 0):
    #     print("Number of ligands should be divisible by number of worker processes. Exiting")
    #     exit()


    if id == 0:    # master

        ligands = genLigandList(args)

        # print details if chose verbose
        printIf(args.verbose, "master created {} ligands : \n{}".format(len(ligands), ligands), flush=True)
        printIf(args.verbose, "to be scored against protein: {}".format(args.protein), flush=True)

        totalWork = len(ligands)
        recvcount = 0  # number of ligand scores from workers

        # used to determine ligands with highest score
        maxScore = -1
        maxScoreLigands = []

        ######################## send chunks to workers
        # workers will get as equal chunks as possible, differing by 1
        #
        # ligands per worker
        n = math.ceil(len(ligands)/(numProcesses-1))
        printIf(args.verbose, "Each worker process will do at most {} ligands".format(n), flush=True)

        remainder = args.nLigands%(numProcesses-1)

        # the workers who will get n ligands (rest will get n-1)
        if (remainder == 0):
            n_workers = numProcesses
        else:
            n_workers = remainder + 1
        # send n to these workers
        for p in range(1, n_workers):
            comm.send(ligands[(p-1)*n:p*n], dest=p)
        # send n-1 to the rest of the workers
        for p in range(n_workers, numProcesses):
            comm.send(ligands[(p-1)*(n-1):p*(n-1)], dest=p)
        ############################################ end of send chunks

        # get a score for a ligand from each worker until all completed
        while (recvcount < totalWork) :
            results, workerId = masterReceiveResults(comm, args)
            rcv_score = results[0]
            lig = results[1]
            recvcount += 1

            # keep track of maximum
            maxScore, maxScoreLigands = updateMaximum(rcv_score, lig, maxScore, maxScoreLigands)

        # print results
        print('The maximum score is', maxScore)
        print('Achieved by ligand(s)', maxScoreLigands)
        finish = MPI.Wtime()  # end the timing
        total_time = finish - start
        print("Total Running time: {0:12.3f} sec".format(total_time))

    else:       # worker

        ligandList = comm.recv(source=0)

        printIf(args.verbose, "Process {} ligandList: {}".format(id, ligandList), flush=True)

        for lig in ligandList:
            s = score(lig, args.protein)
            result = [s, lig]
            comm.send(result, dest=0)

        finish = MPI.Wtime()  # end the timing
        proc_time = finish - start
        print("Process {0:} running time: {1:12.3f} sec".format(id, proc_time))



main()
