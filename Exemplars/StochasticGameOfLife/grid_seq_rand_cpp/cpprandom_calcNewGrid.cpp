
#include <omp.h>
#include <stdio.h>

#include <random>

#include "../calcNewGrid.hpp"
#include "../gol_rules.hpp"

void calcNewGrid(long unsigned int seed, int *grid, int *newGrid, int w, int l, int it) {
    int i, j, id;

    // for debugging, print the iteration number
    // this is to check that random numbers are being generated correctly
    if (w <= 8 && l <= 8) {
        printf("Iteration: %d\n", it);
    }

    for (i = 1; i <= l; i++) {
        for (j = 1; j <= w; j++) {
            

            id = i * (w + 2) + j;
#ifdef STOCHASTIC
            std::mt19937 generator(seed + it);  // start a new stream for each iteration

            // distribution for random numbers
            std::uniform_real_distribution<double> dis(0.0, 1.0);

            double randN = dis(generator);

            if (w <= 8 && l <= 8) {  // for debugging, print the random number and indices
                printf("%f %d %d %d |\n", randN, id, i, j);
            }

            // Implementing theStochastic  Game of Life Rules
            apply_rules(randN, grid, newGrid, id, w);
#else

            // Implementing the Game of Life Rules
            apply_rules(-1.0, grid, newGrid, id, w);
#endif
        }
    }
}
