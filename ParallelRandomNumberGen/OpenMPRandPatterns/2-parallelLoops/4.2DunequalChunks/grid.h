
#ifndef GRID_H
#define GRID_H
#include <stdio.h>
/*
  grid.h

  Utility grid functions for 4.2DunequalChunks example.

  Created by Libby Shoop, Jan, 2026.

*/

// printGrid for debugging
void printGrid(double *grid, int w, int l) {
    int i, j;
    for (i = 0; i < l; i++) {
        for (j = 0; j < w; j++) {
            int id = i * w + j;
            printf("%0.3f  ", grid[id]);
        }
        printf("\n");
    }
}
#endif
