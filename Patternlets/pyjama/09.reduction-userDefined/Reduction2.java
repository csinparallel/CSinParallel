/* Reduction2.java computes the maximum using reduction.
 *
 * Pyjama OpenMP's reduction using the max function.
 * Note that Pyjama currently does NOT support arbitrary 
 * user-defined function in the reduction clause. 
 * 
 * Adapted from Thomas Hines (Tennessee Tech) Pyjama example 
 * by Ruth Kurniawati, Westfield State University, July, 2021
 *
 *  Usage: 
 *    make run
 *    make run ARGS="numThreads n"
 *    For example: make run ARGS="4 100" 
 * 
 *  Exercise:
 *  - Build and run, record sequential time in a spreadsheet
 *  - Uncomment the #pragma omp declare directive, rebuild,
 *     and note the user-defined max function used in the reduction.
 *  - Rerun, using 2, 4, 6, 8, ... threads, recording
 *     the times in the spreadsheet.
 *  - Create a chart that plots the times vs the # of threads.
 *  - Experiment with different n values
 */

import java.util.concurrent.ThreadLocalRandom;

public class Reduction2 {

    public static void main(String[] args) {

        int numThreads = Pyjama.omp_get_num_procs();
        if (args.length >= 1) {
            numThreads = Integer.parseInt(args[0]);
        }
        int n = 100000000;
        if (args.length >= 2) {
            n = Integer.parseInt(args[1]);
        }
        
        int max_val = 0;
        int [] arr = new int[n];
        for (int i = 0; i < n; i++) 
            arr[i] = ThreadLocalRandom.current().nextInt(0, 101);
        
        long startTime = System.currentTimeMillis();

        //#omp parallel num_threads(numThreads) shared(n, arr) reduction(max:max_val)
        {
            //#omp for
            for (int i = 0; i < n; i++)
            {
                if (arr[i] > max_val)
                    max_val = arr[i];
            }
        }

        //for (int i = 0; i < n; i++) System.out.print(arr[i] + " ");
        long endTime = System.currentTimeMillis();
        
        System.out.println("\nmax value = " + max_val);
        System.out.println("Time = " + (endTime-startTime) + " ms");
    }
}
