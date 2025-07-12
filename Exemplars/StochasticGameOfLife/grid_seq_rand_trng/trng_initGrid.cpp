#include "../initGrid.hpp"
#include <random>

// NOTE: Initially this code uses the C++ random number
//       generator capability to create the initial spores.
//       You could practice using trng here insead.


// assign initial population randomly, based on PROB_SPORE
void initGrid(long unsigned int seed, int *grid, int w, int l) {
    int i, j;

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (i = 1; i <= l; i++) {
        for (j = 1; j <= w; j++) {
            double randN = dis(generator);  // next random num

            if (randN < PERCENT_INITIALLY_ALIVE) {
                grid[i * (w + 2) + j] = 1;  // ALIVE
            } else {
                grid[i * (w + 2) + j] = 0;  // DEAD
            }
        }
    }

}