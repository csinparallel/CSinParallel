/* ForkJoin2.java
 * ... illustrates the fork-join pattern 
 *      using multiple OpenMP parallel directives,
 *      and changing the number of threads two ways.
 *
 * Joel Adams, Calvin College, May 2013.
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run
 *
 * Exercise:
 * - Compile & run, compare results to source.
 * - Predict how many threads will be used in 'Part IV'?
 * - Uncomment 'Part IV', recompile, rerun.
 */

class ForkJoin2 {
    public static void main(String[] args) {

        System.out.print("\nBeginning\n");

        //#omp parallel 
        System.out.print("\nPart I");

        System.out.print("\n\nBetween I and II...\n");

        Pyjama.omp_set_num_threads(3);

        //#omp parallel 
        System.out.print("\nPart II...");

        System.out.print("\n\nBetween II and III...\n");

        //#omp parallel num_threads(5)
        System.out.print("\nPart III...");

        System.out.print("\n\nEnd\n\n");
    }
}

