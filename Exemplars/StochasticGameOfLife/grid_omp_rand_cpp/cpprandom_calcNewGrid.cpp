
#include <omp.h>

#include <random>

#include <stdio.h>
#include "../calcNewGrid.hpp"
#include "../gol_rules.hpp"
#include "../chunks.h"


void calcNewGrid(long unsigned int seed, int *grid, int *newGrid, int w, int l, int it) {
    int i, j, id;

    // for debugging, print the iteration number
    // this is to check that random numbers are being generated correctly
    if (w <= 8 && l <= 8) {
        printf("Iteration: %d\n", it);
    }

    // Parallel region with OpenMP
#pragma omp parallel default(none) private(i, j, id) \
    shared(grid, newGrid, w, l, it, seed)
    {
        int tid = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        // for block of random numbers by rows
        // initialize to full grid
        int startRow =0;
        int endRow = l+1; 

        getStartStopRow(tid, numThreads, l, &startRow, &endRow);   // enables unequal blocks per thread

 
        // iterate over the grid (not the ghost rows and columns)
        // each thread works on a block of rows
        for (i = startRow+1; i <= endRow; i++) {
            for (j = 1; j <=w; j++) { 

                id = i * (w + 2) + j;    // cell index in the flattened grid
#ifdef STOCHASTIC
                // thread-local generator seeded with it and tid
                thread_local std::mt19937 generator(seed + it + tid);  
                // distribution for random numbers
                std::uniform_real_distribution<double> dis(0.0, 1.0);

                double randN = dis(generator);

                if (w <= 8 && l <= 8) {// for debugging, print the random number and indices
                    printf("%f %d %d %d %d |\n", randN, id, tid, i, j);
                }

                // Implementing the Stochastic game of life Rules
                apply_rules(randN, grid, newGrid, id, w);
#else
                // Implementing the Game of Life rules
                apply_rules(-1.0, grid, newGrid, id, w);
#endif
            }
        }
    }
}
