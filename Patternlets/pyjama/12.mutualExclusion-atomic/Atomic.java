/* Atomic.java
 * ... illustrates a race condition when multiple threads read from / 
 *  write to a shared variable (and explores OpenMP atomic operations).
 * 
 * NOTE: In Pyjama, atomic is translated to a critical region, instead of atomic hardware operation.
 * There is not a real support for atomic operations in Java.
 * In native OpenMP library, support for atomic the clause will depend on the available atomic operations
 * on the machine - make sure that you check the documentation.
 *
 * Joel Adams, Calvin College, November 2009.
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run 
 *
 * Exercise:
 *  - Compile and run 10 times; note that it always produces the correct balance: $1,000,000.00
 *  - To parallelize, uncomment A, recompile and rerun multiple times, compare results
 *  - To fix: uncomment B, recompile and rerun, compare
 */

class Atomic {
    final static int REPS = 1000000;

    public static void main(String[] args) {
        double balance = 0.0;
  
        System.out.println("\nYour starting bank account balance is "+ 
               balance);

        // simulate many deposits
        /* //#omp parallel for shared(balance)          // A */
        for (int i = 0; i < REPS; i++) {
            /* //#omp atomic                            // B */
            balance += 1.0;
        }

        System.out.println("\nAfter "+REPS+" $1 deposits, your balance is "+balance+"\n");

    }
}

