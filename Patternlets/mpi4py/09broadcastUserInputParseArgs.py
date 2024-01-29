#
#    Illustrates broacasting data entered at the command line from
#    the conductor node to the rest of the nodes.
#
#  Libby Shoop, Macalester College, Jan 2023
#
# Example usage:
#      python run.py ./09broadcastUserInput.py N 22 64.5
#  Here the N signifies the number of processes to start up in mpi,
#  which must be greater than one. The dat must be supplied
#  and represents the list that will be broadcast from the conductor
#  process to the workers.
#
#  run.py executes this program within mpirun using
#         the number of processes given and the dataString argument.
#
#
# Exercise:
#
#    - Run, using N = from 1 through 8 processes, an in and a float
#       your choosing.
#     - Use source code to trace execution- note which processes execute
#       the broadcast command.
#     - Explain the behavior you observe.
#
import argparse      # for command-line arguments
from mpi4py import MPI


def parseArguments():
    """Handle 2 command line arguments

    Returns:
        A list containing each argument provided
    """
    # process command line arguments
    # see https://docs.python.org/3.3/howto/argparse.html#id1
    parser = argparse.ArgumentParser()
    parser.add_argument("age", help="Your age (integer)")
    parser.add_argument("height", help="Your height in inches (float)")

    args = parser.parse_args()
    age = int(args.age)
    height = float(args.height)
    return [age, height]

def main():
    comm = MPI.COMM_WORLD
    id = comm.Get_rank()            #number of the process running the code
    numProcesses = comm.Get_size()  #total number of processes running
    myHostName = MPI.Get_processor_name()  #machine name running the code

    if numProcesses > 1 :

        if id == 0:        # conductor
            #conductor: get the command line arguments
            data = parseArguments()
            print("Conductor Process {} of {} on {} broadcasts \"{}\""\
            .format(id, numProcesses, myHostName, data))

        else :
            # worker: start with empty data
            data = None
            print("Worker Process {} of {} on {} starts with \"{}\""\
            .format(id, numProcesses, myHostName, data))

        #initiate and complete the broadcast
        data = comm.bcast(data, root=0)
        #check the result
        print("Process {} of {} on {} has \"{}\" after the broadcast"\
        .format(id, numProcesses, myHostName, data))

    else :
        print("Please run this program with the number of processes \
greater than 1")

########## Run the main function
main()
