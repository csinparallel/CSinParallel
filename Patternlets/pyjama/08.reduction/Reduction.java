/* Reduction.java
 * ... illustrates the OpenMP parallel-for loop's reduction clause
 *
 * Joel Adams, Calvin College, November 2009.
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run [numThreads]
 *   Example: make run 4 
 *
 * Exercise:
 * - Compile and run.  Note that incorrect output is produced by parallelSum()
 * - Uncomment 'reduction(+:sum)' clause of #omp in parallelSum()
 * - Recompile and rerun.  Note that correct output is produced again.
 */

import java.util.Random;

class Reduction {
    final static int SIZE=1000000;

    public static void main(String[] args) {
        if (args.length >= 1) {
            Pyjama.omp_set_num_threads(Integer.parseInt(args[0]));
        }
        System.out.println();

        // generate SIZE random values in [0..1000) range
        int[] array = new Random().ints(SIZE, 0, 1000).toArray();
        System.out.println("Seq. sum: \t" + sequentialSum(array));
        System.out.println("Par. sum: \t" + parallelSum(array));
    } 


    /* sum the array sequentially */
    static int sequentialSum(int[] a) {
        int sum = 0;
        int i;
        for (i = 0; i < a.length; i++) {
            sum += a[i];
        }
        return sum;
    }

    /* sum the array using multiple threads */
    static int parallelSum(int[] a) {
        int sum = 0;
        //#omp parallel shared(a,sum) 
        {
            //#omp for /* reduction(+:sum) */
            for(int i = 0; i < a.length;i++) {
                sum += a[i];
            }
        }
        return sum;
    }
}