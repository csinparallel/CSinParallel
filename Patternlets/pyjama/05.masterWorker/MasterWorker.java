/* MasterWorker.java
 * ... illustrates the master-worker pattern in OpenMP
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: 
 *   make run [numThreads]
 *   Example: make run 4
 *
 * Exercise: 
 * - Compile and run as is.
 * - Remove the #omp directive, re-compile and re-run
 * - Compare and trace the different executions.
 */

class MasterWorker {
    public static void main(String[] args) {
        if (args.length >= 1) {
            Pyjama.omp_set_num_threads(Integer.parseInt(args[0]));
        }
        System.out.println();


        //#omp parallel 
        {
            int id = Pyjama.omp_get_thread_num();
            int numThreads = Pyjama.omp_get_num_threads();

            if ( id == 0 ) {  // thread with ID 0 is master
                System.out.println("Greetings from the master, #"+ id +" of " +  numThreads + " threads");
            } else {          // threads with IDs > 0 are workers 
                System.out.println("Greetings from a worker, #"+ id +" of " +  numThreads + " threads");
            }
        }

        System.out.println();
    }
}

