
#include <omp.h>
#include <stdio.h>
#include "../calcNewGrid.hpp"
#include "../gol_rules.hpp"
#include "../chunks.h"

#include <trng/uniform01_dist.hpp>
#include <trng/lcg64.hpp>


void calcNewGrid(unsigned long int seed, int *grid, int *newGrid, int w, int l, int it) {

    trng::lcg64 RNengine;       // generator
    trng::uniform01_dist<> uni;  // distribution for random numbers

    // for debugging, print the iteration number
    // this is to check that random numbers are being generated correctly
    if (w <= 8 && l <= 8) {
            printf("Iteration: %d\n", it);
    }

    #pragma omp parallel default(none) \
    shared(grid, newGrid, w, l, it, seed) private(RNengine, uni)
    {
        int i, j;        

        int tid = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        double randN;
        // for block split of random numbers
        // initialize to full grid 
        int startRow =0;
        int endRow = l+1; 
#ifdef STOCHASTIC
        RNengine.seed((long unsigned int)(seed+it));  // reseed for each iteration

        // Use block splitting to partition random numbers among threads
        getStartStopRow(tid, numThreads, l, &startRow, &endRow);   // enables unequal blocks per thread
        long unsigned int numsToSkip = (long unsigned int)startRow * (long unsigned int)w;
        RNengine.jump(numsToSkip);
#endif

        // iterate over the grid (not the ghost rows and columns)
        // each thread works on a block of rows
        for (i = startRow+1; i <= endRow; i++) {
            for (j = 1; j <= w; j++) { 
                int id = i * (w + 2) + j;    // cell index in the flattened grid

#ifdef STOCHASTIC
                if (w <= 8 && l <= 8) {// for debugging, print the random number and indices
                    printf("%f %d %d %d %d |\n", randN, id, tid, i, j);
                }

                randN = uni(RNengine);              // get new number inside loop
                // Implementing the Stochastic Game of Life Rules
                apply_rules(randN, grid, newGrid, id, w);
#else
                // Implementing the classic Game of Life rules
                apply_rules(-1.0, grid, newGrid, id, w);
#endif
            }
        }

    }  // end of parallel region
}