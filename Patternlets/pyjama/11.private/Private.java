import pj.Pyjama;

/* private.c
 * ... illustrates why private variables are needed with OpenMP's parallel for loop
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: ./private 
 *
 * Exercise: 
 * - Run, noting that the sequential program produces correct results
 * - Uncomment line A, recompile/run and compare
 * - Recomment line A, uncomment line B, recompile/run and compare
 */

/* there is a bug in Pyjama when loop control variable i is declared as private (private(i)) */

class Private {
    final static int SIZE = 1000;

    public static void main(String[] args) {
        int j;
        boolean ok = true;
        int[][] m = new int[SIZE][SIZE];

        System.out.println();
        // set all array entries to 1
        
        /* //#omp parallel for shared(m,j)                 // A  */
        /* //#omp parallel for private(j) shared(m)        // B */
        for (int i = 0; i < SIZE; i++) {
            for (j = 0; j < SIZE; j++) {
                //System.out.println("Thread " + Pyjama.omp_get_thread_num() + " setting m["+i+","+j+"]");
                m[i][j] = 1;
            }
        }

        // test (without using threads)
        for (int ii = 0; ii < SIZE; ii++) {
            for (int jj = 0; jj < SIZE; jj++) {
                if ( m[ii][jj] != 1 ) {
                    System.out.println("Element ["+ii+","+jj+"] not set... \n");
                    ok = false;
                }
            }
        }

        if ( ok ) {
            System.out.println("\nAll elements correctly set to 1\n");
        }
    }
}

