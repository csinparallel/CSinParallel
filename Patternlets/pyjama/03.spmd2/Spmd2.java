/* Spmd2.java
 * ... illustrates the SPMD pattern in OpenMP,
 * 	using the commandline arguments 
 *      to control the number of threads.
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: 
 *   make run ARGS=[numThreads]
 *   Example: make run ARGS=4
 *
 * Exercise:
 * - Compile & run with no commandline args 
 * - Rerun with different commandline args,
 *    until you see a problem with thread ids
 * - Fix the race condition
 *    (if necessary, compare to 02.spmd)
 */

import java.util.Random;

class Spmd2 {
    public static void main(String[] args) {
        if (args.length >= 1) {
            Pyjama.omp_set_num_threads(Integer.parseInt(args[0]));
        }
        System.out.println();


        int id, numThreads;
        Random rand;
        //#omp parallel shared(id, numThreads) private(rand)
        {
            // To make it easier to observe the race condition, uncomment the code below:
            //
            rand = new Random(); // use a fixed seed for reproduceability
            try {
                Thread.sleep(rand.nextInt());
            } catch(Exception ex) {};

            numThreads = Pyjama.omp_get_num_threads();
            id = Pyjama.omp_get_thread_num();
            System.out.println("Hello from thread "+ id +" of " + numThreads);
        }

        System.out.println();
    }
}
