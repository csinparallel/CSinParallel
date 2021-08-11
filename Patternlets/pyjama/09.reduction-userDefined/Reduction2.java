/* Reduction2.java computes a table of factorial values,
 *  OpenMP's reduction using the multiplication operator.
 *
 *  Joel Adams, Calvin College, December 2015.
 *  Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 *  Usage: 
 *    make run
 *    make run ARGS="numThreads n"
 *    For example: make run ARGS="4 100" 
 *
 * 
 *  Exercise:
 *  - Build and run, record sequential time in a spreadsheet
 *  - Uncomment #pragma omp parallel for directive, rebuild,
 *     and read the error message carefully.
 *  - Uncomment the #pragma omp declare directive, rebuild,
 *     and note the user-defined * reduction for a BigInt.
 *  - Rerun, using 2, 4, 6, 8, ... threads, recording
 *     the times in the spreadsheet.
 *  - Create a chart that plots the times vs the # of threads.
 *  - Experiment with different n values
 */

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class Reduction2 {
    static long factorial(int numThreads, int n) {
        long acc = 1;

        //#omp parallel shared(acc, n) num_threads(numThreads) 
        {
            //#omp for reduction(*:acc)
            for(int i=2; i <= n; i++) {
                acc *= i;
            }
        }
        return acc;
    }

    public static void main(String[] args) {
        // check and parse argument
        int numThreads = Runtime.getRuntime().availableProcessors();
        if (args.length < 1) {
            System.out.println("Usage " + Reduction2.class.getName() + " numThreads n.");
            System.out.println("Using default number of Threads " + numThreads);
        } else {
            numThreads = Integer.parseInt(args[0]);
        }

        int n = 20;
        if (args.length == 2) {
            n = Integer.parseInt(args[1]);
            if (n > 20) {
                System.out.println("n cannot be greater than 20 (overflow will happen)");
                return;
            }
        }

        long startTime = System.currentTimeMillis();
        long result = factorial(numThreads, n);
        long duration = System.currentTimeMillis() - startTime;

        System.out.println("Result = " + result);
        System.out.println("Time = " + duration + " ms");
    }
}
