/* ConductorWorker.java
 * ... illustrates the conductor-worker pattern
 *      using basic MPI commands
 *      and OpenMPI's Java interface.
 *
 * Joel Adams, Calvin University, November 2019.
 *
 * Usage: mpirun -np 4 java ./ConductorWorker
 *
 * Exercise:
 * - Compile and run the program, varying N from 1 through 8.
 * - Explain what stays the same and what changes as the
 *    number of processes changes.
 */

import mpi.*;

public class ConductorWorker {

 public static final int CONDUCTOR = 0;

 public static void main(String [] args) throws MPIException {
    MPI.Init(args);

    int id           = MPI.COMM_WORLD.getRank();
    int numProcesses = MPI.COMM_WORLD.getSize();
    String hostName  = MPI.getProcessorName();
    String message   = "Greetings from ";

    if (id == CONDUCTOR) {
       message += "the conductor, #" + id
                   + " (" + hostName + ")"
                   + " of " + numProcesses + "\n";
    } else {
       message += "a worker, #" + id
                   + " (" + hostName + ")"
                   + " of " + numProcesses + "\n";
    }

    System.out.print(message);

    MPI.Finalize();
  }
}

