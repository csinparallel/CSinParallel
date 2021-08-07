/* Spmd2.java
 * ... illustrates the SPMD pattern in OpenMP,
 * 	using the commandline arguments 
 *      to control the number of threads.
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: ./spmd2 [numThreads]
 *
 * Exercise:
 * - Compile & run with no commandline args 
 * - Rerun with different commandline args,
 *    until you see a problem with thread ids
 * - Fix the race condition
 *    (if necessary, compare to 02.spmd)
 */

class Spmd2 {
    static int id, numThreads;

    public static void main(String[] args) {
        if (args.length >= 1) {
            Pyjama.omp_set_num_threads(Integer.parseInt(args[0]));
        }
        System.out.println();

        //#omp parallel 
        {
            id = Pyjama.omp_get_thread_num();
            numThreads = Pyjama.omp_get_num_threads();
            System.out.println("Hello from thread "+ id +" of " + numThreads);
        }

        System.out.println();
    }
}
