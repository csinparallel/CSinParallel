/* Critical.java
 * ... fixes a race condition when multiple threads read from / 
 *  write to a shared variable	using the OpenMP critical directive.
 * 
 * NOTE: In Pyjama, atomic is translated to a critical region, instead of atomic hardware operation.
 * There is not a real support for atomic operations in Java.
 *
 * Joel Adams, Calvin College, November 2009.
 * Adapted for Java/Pyjama by Ruth Kurniawati, Westfield State University, July, 2021
 *
 * Usage: 
 *   make run 
 *
 * Exercise:
 *  - Compile and run several times; note that it always produces the correct balance $1,000,000.00 
 *  - Comment out A; recompile/run, and note incorrect result
 *  - To compare: uncomment B1+B2+B3, recompile and rerun, compare
 *  - Compare the code generated using A vs B1/B2/B3 by running "make j2j". The generated code can be found in the "gen" subdirectory.
 */

class Critical {
    final static int REPS = 1000000;

    public static void main(String[] args) {
        double balance = 0.0;
  
        System.out.println("\nYour starting bank account balance is "+ 
               balance);

        // simulate many deposits
        //#omp parallel for shared(balance)  
        for (int i = 0; i < REPS; i++) {
            //#omp atomic                           // A  
            // //#omp critical                         // B1 
            // {                                       // B2
                    balance += 1.0;
            // }                                       // B3
        }

        System.out.println("\nAfter "+REPS+" $1 deposits, your balance is "+balance+"\n");
    }
}


