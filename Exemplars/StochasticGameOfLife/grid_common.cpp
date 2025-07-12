
#include <stdio.h>
#include <omp.h>


// printGrid for debugging
void printGrid(int *grid, int w, int l) {
    int i, j;
    for (i = 1; i <= l; i++) {
        for (j = 1; j <= w; j++) {
            int id = i * (w + 2) + j;
            printf("%d  ", grid[id]);
        }
        printf("\n");
    }
}

// Prints the grid to a file in a formatted way
void prtdat(int w, int l, int *grid, const char *fnam) {
    int ix, iy;
    FILE *fp;

    fp = fopen(fnam, "w");

    for (ix = 1; ix <= l; ix++) {
        for (iy = 1; iy <= w; iy++) {
            int id = ix * (w + 2) + iy;
            fprintf(fp, "%4d", grid[id]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
