/* MessagePassing.java
 * ... illustrates how use of MPI's send and receive commands
 *      can lead to deadlock.
 *
 * Goal: Have MPI processes pair up and exchange their id numbers.
 *
 * Note: Values are sent/received in Java using arrays or buffers.
 *       Buffers are preferred b/c they work for all communication calls.
 *
 * Joel Adams, Calvin University, November 2019,
 *  with error-handling from Hannah Sonsalla, Macalester College 2017.
 *
 * Usage: mpirun -np 4 java ./MessagePassing
 *
 * Exercise:
 * - Compile, then run using 1 process, then 2 processes.
 *    (Use Cntl-c to terminate.)
 * - Use source code to trace execution.
 * - Why does this fail?

 */

import mpi.*;
import java.nio.IntBuffer;

public class MessagePassingDeadlock {

 public static void main(String [] args) throws MPIException {
    MPI.Init(args);

    Comm comm         = MPI.COMM_WORLD;
    int numProcesses  = comm.getSize();
    int id            = comm.getRank();

    if ( numProcesses <= 1 || (numProcesses % 2) != 0)  {
        if (id == MASTER) {
            System.out.print("\nPlease run this program using -np N where N is positive and even.\n\n");
        }
    } else {
        IntBuffer sendBuf = MPI.newIntBuffer(1);
        sendBuf.put(id);
        IntBuffer receiveBuf = MPI.newIntBuffer(1);

        if ( odd(id) ) { // odd processes receive from their 'left' neighbor, then send
            comm.recv(receiveBuf, 1, MPI.INT, id-1, 0); 
            comm.send(sendBuf, 1, MPI.INT, id-1, 0);
        } else {         // even processes receive from their 'right' neighbor, then send
            comm.recv(receiveBuf, 1, MPI.INT, id+1, 0); 
            comm.send(sendBuf, 1, MPI.INT, id+1, 0);
        }

        String message = "Process " + id + " sent '" + sendBuf.get(0)
                         + "' and received '" + receiveBuf.get(0) + "'\n";
        System.out.print(message);
    }

    MPI.Finalize();
  }

  public static boolean odd(int number) { return number % 2 != 0; }

  private static final int MASTER = 0;
}

