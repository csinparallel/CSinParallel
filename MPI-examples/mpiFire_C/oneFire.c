/* oneFire.c 
 * 
 * Libby Shoop
 *
 * Usage: one Fire [forestSize(20)] [probability(0.5) [showGraph(1)]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fire_functions.h"
#include "fireSimPlot.h"
#include "gnuplot_i.h"

#define true 1
#define false 0

// function declarations
void print_forest(int, int **);
void display_forest(int forest_size, int ** forest, double percent_burned);

int main(int argc, char ** argv) {
    // initial conditions and variable definitions
    int forest_size=20;
    double prob_spread = 0.5;
    //    double prob_min=0.0;
    //double prob_max=1.0;

    int **forest;
    double percent_burned = 0.0;
    int do_display=1;


    // check command line arguments

    if (argc > 1) {
        sscanf(argv[1],"%d",&forest_size);
    }
    if (argc > 2) {
        sscanf(argv[2],"%lf",&prob_spread);
    }
    if (argc > 3) {
        sscanf(argv[3],"%d",&do_display);
    }
    if (do_display!=0) do_display=1;

    // setup problem
    seed_by_time(0);
    forest=allocate_forest(forest_size);

    // need prob_spread, percent_burned as single values  
    //burn until fire is gone
    burn_until_out(forest_size,forest,prob_spread,
                forest_size/2,forest_size/2);
    percent_burned = get_percent_burned(forest_size,forest);
 
    // write data out to tempfile
    print_forest(forest_size, forest);
    printf("percent of trees burned: %f\n", percent_burned);
    
    // plot graph 
    if (do_display==1) {
      display_forest(forest_size, forest, percent_burned);
    }
    
    // clean up
    delete_forest(forest_size,forest);
    return 0;
}

// write the state of the forest out to a temporary file.
// print to screen if turn DEBUG flag on when compile
void print_forest(int forest_size,int ** forest) {
    int i,j;

    FILE *outfile = fopen("tmpout.dat", "w");

    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
#ifdef DEBUG
            if (forest[i][j]==BURNT) {
                printf(".");
            } else {
                printf("X");
            }
#endif
	    fprintf(outfile, "%d ", forest[i][j]);
        }
#ifdef DEBUG
        printf("\n");
#endif
	fprintf(outfile, "\n");
    }

    fclose(outfile);
    
}

