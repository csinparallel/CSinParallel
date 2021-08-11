/*  DynamicScheduling.java
 *  Explore OpenMP's schedule() clause by counting the number of 
 *  prime numbers between 0 and n. 
 *
 *  Adapted from OpenMP example from Shaikh Ghafoor and Mike Rogers (Tennessee Tech) 
 *  CDER workshop summer 2021. 
 *  Ruth Kurniawati, Westfield State University, July, 2021
 *
 *  Usage: 
 *     make run ARGS="numThreads n"
 *     Example: make run ARGS=4 10000000
 *
 *  Exercise:
 *  - Build and run, record sequential run time in a spreadsheet
 *  - Uncomment #pragma omp parallel for, rebuild,
 *      run using 2, 4, 6, 8, ... threads, record run times.
 *  - Uncomment schedule(dynamic), rebuild,
 *      run using 2, 4, 6, 8, ... threads, record run times.
 *  - Create a line chart plotting run times vs # of threads.
 */

class DynamicScheduling {

    static boolean isPrime(int n) {
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        int half = n / 2;
        for(int i = 3; i < half; i+=2) {
            if (n % i == 0) return false;
            if (i*i > n) break;
        }
        return true;
    }

    public static void main(String[] args) {

        int numThreads = Pyjama.omp_get_num_procs();
        if (args.length >= 1) {
            numThreads = Integer.parseInt(args[0]);
        }

        int n = 1000000; // one million, cannot 1_000_000 since we're limited to Java 1.5 syntax
        if (args.length >= 2) {
            n = Integer.parseInt(args[1]);
        }

        long startTime = System.currentTimeMillis();
        int count = 1;
        //#omp parallel for shared(n) num_threads(numThreads) reduction(+:count) /* schedule(dynamic) */
        for(int i = 3; i <= n; i++) {
            if (isPrime(i)) count++;
        }
    
        long endTime = System.currentTimeMillis();
        
        System.out.println("The number between of prime numbers between 0 and "+ n + " is " + count);
        System.out.println("Time = " + (endTime-startTime) + " ms");

    }
}