import pj.Pyjama;

/* ParallelLoopChunksOf1.java
 * ... illustrates how to make OpenMP map threads to 
 *	parallel loop iterations in chunks of size 1
 *	(use when not accesssing memory).
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run [numThreads]
 *   Example: make run 4
 *
 * Exercise:
 * 1. Compile and run, comparing output to source code,
 *    and to the output of the 'equal chunks' version.
 * 2. Uncomment the "commented out" code below,
 *    and verify that both loops produce the same output.
 *    The first loop is simpler but more restrictive;
 *    the second loop is more complex but less restrictive.
 */

class ParallelLoopChunksOf1 {
    final static int REPS = 16;
    public static void main(String[] args) {
        if (args.length >= 1) {
            Pyjama.omp_set_num_threads(Integer.parseInt(args[0]));
        }
        System.out.println();

        //#omp parallel for schedule(static,1)
        for (int i = 0; i < REPS; i++) {
            int id = Pyjama.omp_get_thread_num();
            System.out.println("Thread "+id+" performed iteration "+i);
        }

        /*
        System.out.println("--\n\n");

        //#omp parallel
        {
            int numThreads = Pyjama.omp_get_num_threads();
            int id = Pyjama.omp_get_thread_num();
            for (int i = id; i < REPS; i+=numThreads) {
                System.out.println("Thread "+id+" performed iteration "+i);
            }
        }
        */

        System.out.println();
    }
}

