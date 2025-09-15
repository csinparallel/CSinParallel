
// A simulation of Game of Life using a  structured grid approach.
//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <random>

#include "command_line/getCommandLine.hpp"
#include "graphics/display.h"

// common functions for all variations of the mushroom simulation
#include "grid_common.hpp"

// common declarations for functions with different implementations,
// depending on the random number generator used and whether seuential or parallel
#include "calcNewGrid.hpp"
#include "initGrid.hpp"

// in this file
void update_grid(int *grid, int *newGrid, int w, int l);
void init_ghosts(int *grid, int *newGrid, int w, int l);
void initCenterGrid(int *grid, int w, int l);

#define SEED 948573288UL  // for debugging using the same grid

int main(int argc, char *argv[]) {
    int i, j;  // loop variables

    /////////////////// default values  ///////////////////////////////////
    // number of iterations
    int itEnd = 1 << 10;
    // dimension of square grid
    int dim = 800;     // Default square grid Grid dimension (excluding ghost cells)
    int width = dim;   // width of grid
    int length = dim;  // length of grid
    // if in debug mode, we will use a fixed seed to initialize the grid
    int debug = 0;
    // whether to print the grid at beginning and end for debugging
    int graphics = 0;
    // whather to create png files for movie animation
    int movie = 0;
    // whether to depict an animated gnuplot plot of the grid
    int animation = 0;
    // whether to place the initial glider pattern in the center of the grid
    int centerInit = 0;

    int numThreads = 1;  // not used in sequential version, but ready for openMP
    int experiment = 0;  // whether to run in experiment mode (just print time)

    // grab the command line arguments
    getArguments(argc, argv, &width, &length, &itEnd, &debug, &graphics,
                 &animation, &movie, &centerInit, &numThreads, &experiment);

    // !#!#!#!#!#!#!#!#
    // for sequential version, keep numThreads at 1. REMOVE FOR OpenMP
    // numThreads = 1;  // set to 1 for sequential version
    // !#!#!#!#!#!#!#!# FIX ABOVE LINE FOR openMP version by removing
    omp_set_num_threads(numThreads);
    if (!experiment)
        printf("Using %d threads\n", numThreads);

    // grid array with dimension width x length + ghost columns and rows
    int arraySize = (width + 2) * (length + 2);
    size_t bytes = arraySize * sizeof(int);
    int *grid = (int *)malloc(bytes);

    // allocate result grid
    int * /*restrict*/ newGrid = (int *)malloc(bytes);

    if (!experiment) {
        printf("debug %d   graphics %d    dim %d w x %d l   iter %d   \n", debug, graphics, width, length, itEnd);
        printf("centerGlider %d   movie %d   animation %d\n", centerInit, movie, animation);
    };

    long unsigned int seed;
    if (debug) {
        seed = (long unsigned int)SEED;
    } else {
        seed = time(NULL);
    }

    if (centerInit) {
        initCenterGrid(grid, width, length);
    } else {
        initGrid(seed, grid, width, length);
    }

    int total = 0;  // total sum of cell values for debugging
    int it;

    char data_filename[80];
    char png_filename[80];
    char iter_string[20];

    // initial state for movie images
    if (movie) {
        prtdat(width, length, grid, "movie_images/0_initial.dat");
        snprintf(data_filename, 80, "movie_images/0_initial.dat");
        snprintf(png_filename, 80, "movie_images/0_initial.png");
        drawIteration(-1, data_filename, png_filename, width, length);
    }

    if (graphics || debug || animation) {
        prtdat(width, length, grid, "initial.dat");
    }

    // for ascii debugging without using graphics
    if (debug && !graphics && width <= 10) {
        // print the final grid for debugging
        printf("Initial grid:\n\n");
        printGrid(grid, width, length);
    }

    // for replotting the grid
    gnuplot_ctrl *plt;
    if (animation) {
        plt = drawFirst(width, length);
    }

    // start timing
    double st = omp_get_wtime();

    //////////////   run the simulation
    for (it = 0; it < itEnd; it++) {
        init_ghosts(grid, newGrid, width, length);

        calcNewGrid(seed, grid, newGrid, width, length, it);

        if (width <= 10 && length <= 10) {
            printGrid(newGrid, width, length);  // print the new grid for debugging purposes
        }
   
        update_grid(grid, newGrid, width, length);

        if (animation) {
            prtdat(width, length, grid, "intermediate.dat");
            redrawIteration(plt);
            // usleep(500000);  // sleep for 0.5 seconds
            usleep(250000);
        }

        if (movie) {
            // make sure filenames are in order by zero-padding the iteration number
            if (it < 10) {
                snprintf(iter_string, 20, "0%d", it);
            } else {
                snprintf(iter_string, 20, "%d", it);
            }
            snprintf(data_filename, 80, "movie_images/iteration_%s.dat", iter_string);
            snprintf(png_filename, 80, "movie_images/iteration_%s.png", iter_string);

            prtdat(width, length, grid, data_filename);
            drawIteration(it, data_filename, png_filename, width, length);
        }
    }

    // end timing
    double runtime = omp_get_wtime() - st;

    if (!experiment) {
        printf("Total time: %f s\n", runtime);
    } else {
        // for experiments, print out just the time with a tab
        printf("%lf\t", runtime);
    }

    // sum up cells (low level check of correctness)
    for (i = 1; i <= length; i++) {
        for (j = 1; j <= width; j++) {
            total += grid[i * (width + 2) + j];
        }
    }

    if (!experiment) {
        printf("Total: %d\n", total);
    }

    if (graphics || debug || movie) {
        prtdat(width, length, grid, "final.dat");
    }

    if (graphics || movie) {
        // shows on screen and makes result.png
        drawLast(width, length); 
    }

    if (debug && !graphics && width <= 10) {
        // print the final grid for debugging
        printf("Final grid:\n\n");
        printGrid(grid, width, length);
    }

    if (animation && !graphics) {
        printf("Press Enter to exit...\n");
        getchar();
    }

    free(grid);
    free(newGrid);

    return 0;
}


// Transfers newGrid data to grid to prepare for next time step
void update_grid(int *grid, int *newGrid, int w, int l) {
    int i, j;
// copy new grid over to the new grid
    // Note: we do not copy the ghost rows and columns, they are initialized separately
    // in init_ghosts() and will be copied over to grid in the next iteration

    // OpenMP parallel for loop to update the grid
#pragma omp parallel for default(none) private(i, j) \
    shared(w, l, grid, newGrid)                      \
    collapse(2)
    for (i = 1; i <= l; i++) {
        for (j = 1; j <= w; j++) {
            int id = i * (w + 2) + j;
            grid[id] = newGrid[id];
        }
    }
}

// Initializes ghost rows and columns
void init_ghosts(int *grid, int *newGrid, int w, int l) {
    int i;

// ghost rows
#pragma omp parallel for default(none) private(i) \
    shared(grid, w, l)
    for (i = 1; i <= w; i++) {
        grid[(w + 2) * (l + 1) + i] = grid[(w + 2) + i];  // copy first row to bottom ghost row
        grid[i] = grid[(w + 2) * l + i];                  // copy last row to top ghost row
    }

// ghost columns
#pragma omp parallel for default(none) private(i) \
    shared(grid, w, l)
    for (i = 0; i <= l + 1; i++) {
        grid[i * (w + 2) + w + 1] = grid[i * (w + 2) + 1];  // copy first column to right most ghost column
        grid[i * (w + 2)] = grid[i * (w + 2) + w];          // copy last column to left most ghost column
    }
}

void initCenterGrid(int *grid, int w, int l) {
    int x, y;
    // Initialize the grid with glider pattern 
    // This is an interesting pattern for testing
/* from this site:
https://medium.com/better-programming/creating-conways-game-of-life-in-c-523db7404577#:~:text=The%20glider%20The%20glider%20with%20is%20defined,other%20gliders%20to%20produce%20more%20complex%20patterns.

"The glider with is defined by five cells located at 
(x, y), (x+1, y), (x+2, y), (x+2, y+1) and (x+1, y+2) coordinates, respectively. 
It glides across the screen diagonally and can collide with other gliders to produce
 more complex patterns. "

 This pattern and others like it are also described here:
 https://pi.math.cornell.edu/~lipa/mec/lesson6.html

*/

    // starting point for the glider pattern
    y = (w+2) * l/2;  // half way down and 5 cells in from the left
    x = 6;  // 6 cells in from the left
    grid[y + x] = 1;
    grid[y + x + 1] = 1; // x+1, y
    grid[y + x + 2] = 1; // x+2, y
    grid[y + x + 2 + (w + 2)] = 1; // x+2, y+1
    grid[y + x + 1 + 2*(w + 2)] = 1; // x+1, y+2

}
