/* ParallelLoopEqualChunks.java
 * ... illustrates the use of OpenMP's default parallel for loop in which
 *  	threads iterate through equal sized chunks of the index range
 *	(cache-beneficial when accessing adjacent memory locations).
 *
 * Joel Adams, Calvin College, November 2009.
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run [numThreads]
 *   Example: make run 4
 *
 * Exercise
 * - Compile and run, comparing output to source code
 * - try with different numbers of threads, e.g.: 2, 3, 4, 6, 8
 */

class ParallelLoopEqualChunks {
    final static int REPS = 16;
    public static void main(String[] args) {
        if (args.length >= 1) {
            Pyjama.omp_set_num_threads(Integer.parseInt(args[0]));
        }
        System.out.println();

        //#omp parallel for  
        for (int i = 0; i < REPS; i++) {
            int id = Pyjama.omp_get_thread_num();
            System.out.println("Thread "+id+" performed iteration "+i);
        }

        System.out.println();
    }
}

