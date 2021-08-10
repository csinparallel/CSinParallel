/* Reduction2.java computes a table of factorial values,
 *  OpenMP's user-defined reductions.
 *
 *  Joel Adams, Calvin College, December 2015.
 *
 *  Usage: ./reduction2 [numThreads] [n]
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

import java.math.BigInteger;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class Reduction2 {
    static class MyBigInteger {
        private BigInteger myValue;

        public MyBigInteger() {
            myValue = BigInteger.valueOf(1);
        }

        public MyBigInteger(int val) {
            myValue = BigInteger.valueOf(val);
        }

        public MyBigInteger(BigInteger val) {
            myValue = val;
        }

        public MyBigInteger multiply(MyBigInteger arg) {
            return new MyBigInteger(myValue.multiply(arg.myValue));
        }

        @Override
        public String toString() {
            return myValue.toString();
        }
    }

    static MyBigInteger multiply(MyBigInteger arg1, MyBigInteger arg2) {
        return arg1.multiply(arg2);
    }

    static MyBigInteger factorial(int numThreads, int n) {
        MyBigInteger acc = new MyBigInteger(1);

        //#omp parallel shared(acc, n) num_threads(numThreads) 
        {
            //#omp for reduction(multiply:acc)
            for(int i=2; i <= n; i++) {
                MyBigInteger x = new MyBigInteger(i);
                acc = acc.multiply(x);
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

        int n = 32;
        if (args.length == 2) {
            n = Integer.parseInt(args[1]);
        }

        long startTime = System.currentTimeMillis();
        MyBigInteger result = factorial(numThreads, n);
        long duration = System.currentTimeMillis() - startTime;

        System.out.println("Result = " + result.toString());
        System.out.println("Time = " + duration + " ms");
    }
}
