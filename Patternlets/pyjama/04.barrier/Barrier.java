/* Barrier.java
 * ... illustrates the use of the OpenMP barrier command,
 * 	using the commandline to control the number of threads...
 *
 * Joel Adams, Calvin College, May 2013.
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run ARGS=[numThreads]
 *   Example: make run ARGS=8
 * 
 * Exercise:
 * - Compile & run several times, noting interleaving of outputs.
 * - Remove the barrier directive, recompile, rerun,
 *    and note the change in the outputs.
 */
class Barrier {
    public static void  main(String[] args) {
        if (args.length >= 1) {
            Pyjama.omp_set_num_threads( Integer.parseInt(args[0]) );
        }

        System.out.println();
        //#omp parallel 
        {
            int id = Pyjama.omp_get_thread_num();
            int numThreads = Pyjama.omp_get_num_threads();
            System.out.println("Thread "+id + " of "+numThreads+" is BEFORE the barrier.");

            //Try this with and without the barrier
            //#omp barrier 

            System.out.println("Thread "+id + " of "+numThreads+" is AFTER the barrier.");
        }

        System.out.println();
    }
}
