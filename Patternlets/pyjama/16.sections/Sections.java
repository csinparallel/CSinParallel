/* sections.java
 * ... illustrates the use of OpenMP's parallel section/sections directives,
 *      which can be used for task parallelism...
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: ./sections
 *
 * Exercise: Compile, run (several times), compare output to source code.
 */

public class Sections {

    public static void main(String[] args) {

        int numThreads = Pyjama.omp_get_num_procs();
        if (args.length >= 1) {
            numThreads = Integer.parseInt(args[0]);
        }

        System.out.println("\nBefore...\n");

        //#omp parallel sections num_threads(4)
        {
            //#omp section 
            {
                System.out.println("Task/section A performed by thread " +  Pyjama.omp_get_thread_num() ); 
            }
            //#omp section 
            {
                System.out.println("Task/section B performed by thread " + Pyjama.omp_get_thread_num() ); 
            }
            //#omp section
            {
                System.out.println("Task/section C performed by thread " + Pyjama.omp_get_thread_num() ); 
            }
            //#omp section 
            {
                System.out.println("Task/section D performed by thread " + Pyjama.omp_get_thread_num() ); 
            }
        }

        System.out.println("\nAfter...\n");
    }
}

