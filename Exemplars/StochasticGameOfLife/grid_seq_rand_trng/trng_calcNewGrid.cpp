
#include <omp.h>
#include <stdio.h>
#include "../calcNewGrid.hpp"
#include "../gol_rules.hpp"

#include <trng/uniform01_dist.hpp>
#include <trng/lcg64.hpp>
#include <trng/mt19937.hpp>


void calcNewGrid(unsigned long int seed, int *grid, int *newGrid, int w, int l, int it) {

    // trng::mt19937 RNengine1;  // generator you could experiment with
    trng::lcg64 RNengine2;    // another generator; using this because it is fast
    trng::uniform01_dist<> uni;  // distribution for random numbers

    // for debugging, print the iteration number
    // this is to check that random numbers are being generated correctly
    if (w <= 8 && l <= 8) {
            printf("Iteration: %d\n", it);
    }

        int i, j;
        double randN;

#ifdef STOCHASTIC        
        RNengine2.seed((long unsigned int)(seed+it));  // reseed for each iteration
#endif

        // iterate over the grid (not the ghost rows and columns)
        for (i = 1; i <= l; i++) {
            for (j = 1; j <= w; j ++) {  // goal: each thread works on a different column            

                int id = i * (w + 2) + j; // cell index in the flattened grid
                
#ifdef STOCHASTIC
                if (w <= 8 && l <= 8) {// for debugging, print the random number and indices
                    printf("%f %d %d %d |\n", randN, id, i, j);
                }

                randN = uni(RNengine2);              // inside loop
                // Implementing the Stochastic Game of Life Rules
                apply_rules(randN, grid, newGrid, id, w);
#else
                // Implementing the classic Game of Life rules
                apply_rules(-1.0, grid, newGrid, id, w);
#endif
            }
        }

}