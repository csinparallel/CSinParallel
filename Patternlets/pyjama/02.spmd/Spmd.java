/* Spmd.java
 * ... illustrates the single-program-multiple-data (SPMD)
 *      pattern using two basic OpenMP commands...
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: ./spmd
 *
 * Exercise:
 * - Compile & run 
 * - Uncomment pragma, recompile & run, compare results
 */

class Spmd {
    public static void main(String[] args) {
        System.out.println();

        //#omp parallel 
        {
            int id = Pyjama.omp_get_thread_num();
            int numThreads = Pyjama.omp_get_num_threads();
            System.out.println("Hello from thread " + id + " of " + numThreads);
        }

        System.out.println();
    }
}
