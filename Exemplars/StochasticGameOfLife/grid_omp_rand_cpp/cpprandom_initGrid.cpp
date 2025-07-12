
#include <omp.h>

#include <random>

#include "../initGrid.hpp"

// assign initial population randomly, based on PROB_SPORE
void initGrid(long unsigned int seed, int *grid, int w, int l) {
    int i, j;

#pragma omp parallel default(none) private(i, j) \
    shared(grid, w, l, seed)
    {
        int tid = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        thread_local std::mt19937 generator(seed + tid);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        for (i = 1; i <= l; i++) {
            for (j = 1; j <= w; j += numThreads) {
                double randN = dis(generator);  // next random num

                if (randN < PERCENT_INITIALLY_ALIVE) {
                    grid[i * (w + 2) + j] = 1;  // ALIVE
                } else {
                    grid[i * (w + 2) + j] = 0;  // DEAD
                }
            }
        }
    }  // end of parallel region
}
