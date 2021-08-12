/*  SimpleDynamicScheduling.java
 *  Explore OpenMP's schedule() clause by counting the number of 
 *  prime numbers between 0 and n. 
 *
 *  Ruth Kurniawati, Westfield State University, July, 2021
 *
 *  Usage: 
 *     make run ARGS=numThreads
 *     Example: make run ARGS=4
 *
 *  Exercise:
 *  - Build and run, record sequential run time in a spreadsheet
 *  - Uncomment #pragma omp parallel for, rebuild,
 *      run using 2, 4, ... threads, record run times.
 *  - Uncomment schedule(dynamic), rebuild,
 *      run using 2, 4, ... threads, record run times.
 *  - Create a line chart plotting run times vs # of threads.
 */

class SimpleDynamicScheduling {

    static void sleepALittle(int numMillis) {
        try { 
            Thread.sleep(numMillis); 
        } catch(InterruptedException e) {
            // do nothing
        }
    }

    public static void main(String[] args) {
        int numThreads = Pyjama.omp_get_num_procs();
        if (args.length >= 1) {
            numThreads = Integer.parseInt(args[0]);
        }

        long startTime = System.currentTimeMillis();
        int count = 1;

        //#omp parallel for num_threads(numThreads) schedule(dynamic) 
        for(int i = 1; i <= 100; i++) {
            sleepALittle(i);
        }
    
        long endTime = System.currentTimeMillis();        
        System.out.println("Time = " + (endTime-startTime) + " ms");

    }
}