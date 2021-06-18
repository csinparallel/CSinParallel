# Drug Design Exemplar:
# Dynamic assignment of tasks to workers
#
#  This version of the 'drug 'design' program uses the master-worker
#  pattern.  The master generates a list of ligands to be matched to
#  a given protein and scored based on simple matching of letters.
#  The master assigns one ligand at a time to each worker, who will
#  compute a score for it.
#  After the worker completes one, they send the resulting score and
#  request another. This dynamic assignmnet means that if one ligand takes
#  a worker a long time to compute, another worker can complete
#  additional ones that might take less time.
#  In this version, the master is keeping track of the overall
#  maximum scoring ligands.
#
#  To run a small example:
#        mpirun -np 4 python ./dd_mpi_dynamic.py 18 -verbose
#  where  18 is the number of ligands to create
#  To see all the options:
#        python ./dd_mpi_dynamic.py --help

import math
from mpi4py import MPI

# Functions in common between this and the 'equal chunks' version
from dd_functions import *

# tags that can be applied to messages
WORKTAG = 1
DIETAG = 2

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

    if (id == 0) :
        # create an array of ligands of varying size
        ligands = genLigandList(args)

        # print details if chose verbose
        printIf(args.verbose, "master created {} ligands : \n{}".format(len(ligands), ligands), flush=True)
        printIf(args.verbose, "to be scored against protein: {}".format(args.protein), flush=True)

        handOutWork(ligands, comm, numProcesses, args, myHostName)

        finish = MPI.Wtime()  # end the timing
        total_time = finish - start
        # print("Total Running time: {0:12.3f} sec".format(total_time))
        print("Total Running time: {0:12.3f} sec".format(total_time))

    else:
        worker(comm, args, myHostName)

        finish = MPI.Wtime()  # end the timing
        proc_time = finish - start
        print("Process {0:} running time: {1:12.3f} sec".format(id, proc_time))


def handOutWork(ligands, comm, numProcesses, args, myHostName):
    totalWork = len(ligands)
    workcount = 0
    recvcount = 0
    # used to determine ligands with highest score
    maxScore = -1
    maxScoreLigands = []

    printIf(args.verbose, "master sending first tasks", flush=True)
    # send out the first tasks to all workers
    for id in range(1, numProcesses):
        if workcount < totalWork:
            work=ligands[workcount]
            comm.send(work, dest=id, tag=WORKTAG)
            workcount += 1
            printIf(args.verbose,"master on {} sent {} to {}".format(myHostName, work, id), flush=True)

    # while there is still work,
    # receive result from a worker, which also
    # signals they would like some new work
    while (workcount < totalWork) :
        results, workerId = masterReceiveResults(comm, args)
        score = results[0]
        lig = results[1]
        recvcount += 1

        #send next work
        work=ligands[workcount]
        comm.send(work, dest=workerId, tag=WORKTAG)
        workcount += 1
        printIf(args.verbose,"master on {} sent {} to {}".format(myHostName, work, workerId), flush=True)

        # keep track of maximum
        maxScore, maxScoreLigands = updateMaximum(score, lig, maxScore, maxScoreLigands)


    # Receive results for outstanding work requests.
    while (recvcount < totalWork):
        results, workerId = masterReceiveResults(comm, args)
        score = results[0]
        lig = results[1]
        recvcount += 1

        # keep track of maximum
        maxScore, maxScoreLigands = updateMaximum(score, lig, maxScore, maxScoreLigands)

    # Tell all workers to stop
    for id in range(1, numProcesses):
        comm.send(-1, dest=id, tag=DIETAG)

    # print results
    print('The maximum score is', maxScore)
    print('Achieved by ligand(s)', maxScoreLigands)



#
# Actions of the worker: receive ligand, compute score, and return them
#
def worker(comm, args, myHostName):
    # keep receiving messages and do work, unless tagged to 'die'
    while(True):
        stat = MPI.Status()
        nextLigand = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        printIf(args.verbose, "worker {} on {} got {}".format(comm.Get_rank(), myHostName, nextLigand), flush=True)
        # stop if message has special tag
        if (stat.Get_tag() == DIETAG):
            printIf(args.verbose, "worker {} dying".format(comm.Get_rank()), flush=True)
            return
        # do work of scoring the ligand
        s = score(nextLigand, args.protein)
        # indicate done with work by sending to Master
        result = [s, nextLigand]
        comm.send(result, dest=0)

########## Run the main function
main()
