/* ForkJoin.java
 * ... illustrates the fork-join pattern 
 *      using OpenMP's parallel directive.
 *
 * Joel Adams, Calvin College, November 2009.
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run
 *
 * Exercise:
 * - Compile & run, uncomment the pragma,
 *    recompile & run, compare results.
 */

class ForkJoin {

    public static void main(String[] args) {

        System.out.println("\nBefore...");

        //#omp parallel num_threads(4)
        System.out.println("\nDuring...");

        System.out.println("\nAfter...\n");
    }

}