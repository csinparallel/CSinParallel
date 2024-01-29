#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "fire_functions.h"

#define UNBURNT 0
#define SMOLDERING 1
#define BURNING 2
#define BURNT 3

#define true 1
#define false 0


void seed_by_time(int offset) {
    time_t the_time;
    time(&the_time);
    srand((int)the_time+offset);
}



int burn_until_out(int forest_size,int ** forest, double prob_spread,
    int start_i, int start_j) {
    int count;

    initialize_forest(forest_size,forest);
    light_tree(forest_size,forest,start_i,start_j);

    // burn until fire is gone
    count = 0;
    while(forest_is_burning(forest_size,forest)) {
        forest_burns(forest_size,forest,prob_spread);
        count++;
    }

    return count;
}

double get_percent_burned(int forest_size,int ** forest) {
    int i,j;
    int total = forest_size*forest_size-1;
    int sum=0;

    // calculate pecrent burned
    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i][j]==BURNT) {
                sum++;
            }
        }
    }

    // return percent burned;
    return ((double)(sum-1)/(double)total);
}


int ** allocate_forest(int forest_size) {
    int i;
    int ** forest;

    forest = (int **) malloc (sizeof(int*)*forest_size);
    for (i=0;i<forest_size;i++) {
        forest[i] = (int *) malloc (sizeof(int)*forest_size);
    }

    return forest;
}

void initialize_forest(int forest_size, int ** forest) {
    int i,j;

    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            forest[i][j]=UNBURNT;
        }
    }
}

void delete_forest(int forest_size, int ** forest) {
    int i;

    for (i=0;i<forest_size;i++) {
        free(forest[i]);
    }
    free(forest);
}

void light_tree(int forest_size, int ** forest, int i, int j) {
    forest[i][j]=SMOLDERING;
}

boolean fire_spreads(double prob_spread) {
    if ((double)rand()/(double)RAND_MAX < prob_spread) 
        return true;
    else
        return false;
}

void forest_burns(int forest_size, int **forest,double prob_spread) {
    int i,j;
    extern boolean fire_spreads(double);

    //burning trees burn down, smoldering trees ignite
    for (i=0; i<forest_size; i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i][j]==BURNING) forest[i][j]=BURNT;
            if (forest[i][j]==SMOLDERING) forest[i][j]=BURNING;
        }
    }

    //unburnt trees catch fire
    for (i=0; i<forest_size; i++) {
        bool north_edge = i == 0;
        bool south_edge = i == forest_size - 1;

        for (j=0;j<forest_size;j++) {
            bool west_edge = j == 0;
            bool east_edge = j == forest_size - 1;

            
            // if (forest[i][j]==BURNING) {
            if (forest[i][j]==UNBURNT && fire_spreads(prob_spread)) {
                if (( (!north_edge) && forest[i-1][j]==BURNT)  ||
                      (!south_edge) && forest[i+1][j]==BURNT ||
                      (!west_edge) && forest[i][j-1]==BURNT ||
                      (!east_edge) && forest[i][j+1]==BURNT
                ) {
                    forest[i][j] = SMOLDERING;
                }
                // if (( (i!=0) && forest[i-1][j]==BURNT)  ||
                //       (i!=forest_size-1) && forest[i+1][j]==BURNT ||
                //       (j!=0) && forest[i][j-1]==BURNT ||
                //       (j!=forest_size-1) && forest[i][j+1]==BURNT
                // ) {
                //     forest[i][j] = SMOLDERING;
                // }
                // if (i!=0) { // North
                //     if (fire_spreads(prob_spread)&&forest[i-1][j]==UNBURNT) {
                //         forest[i-1][j]=SMOLDERING;
                //     }
                // }
                // if (i!=forest_size-1) { //South
                //     if (fire_spreads(prob_spread)&&forest[i+1][j]==UNBURNT) {
                //         forest[i+1][j]=SMOLDERING;
                //     }
                // }
                // if (j!=0) { // West
                //     if (fire_spreads(prob_spread)&&forest[i][j-1]==UNBURNT) {
                //         forest[i][j-1]=SMOLDERING;
                //     }
                // }
                // if (j!=forest_size-1) { // East
                //     if (fire_spreads(prob_spread)&&forest[i][j+1]==UNBURNT) {
                //         forest[i][j+1]=SMOLDERING;
                //     }
                // }
            }
        }
    }
}

boolean forest_is_burning(int forest_size, int ** forest) {
    int i,j;

    for (i=0; i<forest_size; i++) {
        for (j=0; j<forest_size; j++) {
            if (forest[i][j]==SMOLDERING||forest[i][j]==BURNING) {
                return true;
            }
        }
    }
    return false;
}

