//
// This file is part of the Game of Life Simulation
//
// It provides functions to visualize the simulation using gnuplot in an X11 window
// by drawing the initial and final grids or updating the display at each iteration.
// It also has a function to draw each iteration of the simulation to a png file in the
// movie_images directory for later creating an mpeg animation of the simulation.
//

#include <stdio.h>

#include "gnuplot_i.h"

// size the window for a 16 x 9 aspect ratio of the screen
#define MAX_WINDOW_SIZE_TALL 800  // maximum size of the gnuplot window when length>width
#define MAX_WINDOW_SIZE 1400      // maximum size of the gnuplot window when width>length

// Calculate a scaling factor for the gnuplot window
// based on the original width and length of the grid
// and vallues for maximum width or height of the window.
float calcScaleFactor(int width, int length) {
    float scaleFactor = 1.0;

    if (width <= length) {  // taller
        // scale to the maximum height of the window
        scaleFactor = (float)MAX_WINDOW_SIZE_TALL / (float)length;
    } else {  // wider
        // scale to the maximum width of the window
        scaleFactor = (float)MAX_WINDOW_SIZE / (float)width;
    }
    return scaleFactor;
}

// set the color palette for the gnuplot plot
// also changed the x and y axes so that the glider looks
// like the one shown here:
// https://pi.math.cornell.edu/~lipa/mec/lesson6.html
//
void setColorPalette(gnuplot_ctrl* plt, int w, int l) {
    char yrange_str[100];
    snprintf(yrange_str, 100, "set yrange [%d:0] reverse", l);
    char xrange_str[100];
    snprintf(xrange_str, 100, "set xrange [%d:0] reverse", w);
    gnuplot_cmd(plt, "set cbrange [0:1]");
    gnuplot_cmd(plt, "set xrange [0:*]");
    gnuplot_cmd(plt, yrange_str);
    gnuplot_cmd(plt, xrange_str);
    gnuplot_cmd(plt, "set key noautotitle");
    gnuplot_cmd(plt, "unset xtics");
    gnuplot_cmd(plt, "unset ytics");
    gnuplot_cmd(plt,
                " set palette defined (\
        0 '#FFFFFF', \
        1 '#000000'\
    )");
    // 0 empty/dead
    // 1 alive
    gnuplot_cmd(plt, "unset colorbox");  // do not show color box
    
 
}

// Redraws the current iteration of the mushroom simulation on the X11 window.
// This function assumes that the gnuplot control structure is already initialized
// using `drawFirst()` and that the terminal is set to x11 for animation.
//
void redrawIteration(gnuplot_ctrl* plt) {
    // this assumes the pointer to the plot is already initialized for x11 terminal
    gnuplot_cmd(plt, "replot '%s' matrix with image", "intermediate.dat");
}

// Draws the first grid of the simulation to X11 window
// and returns a pointer to the gnuplot control structure for use in
// subsequent redraws for each iteration of the simulation.
// This is designed to be called before the animated simulation starts.
gnuplot_ctrl* drawFirst(int width, int length) {

    char term_str[200];   // for x11 window setup
    char term_str2[200];  // for png file setup
    char title_str[200];

    float new_width = (float)width;
    float new_length = (float)length;

    float scaleFactor = calcScaleFactor(width, length);

    new_width *= scaleFactor;
    new_length *= scaleFactor;

    //////////////// initial state of grid
    snprintf(term_str, 200, "set terminal x11 size %d, %d", (int)new_width, (int)new_length);

    gnuplot_ctrl* plt;

    plt = gnuplot_init();

    snprintf(title_str, 200, 
            "set title 'Game of Life Simulation: %d x %d Grid'", width, length);

    gnuplot_cmd(plt, title_str);
    gnuplot_cmd(plt, term_str);
    setColorPalette(plt, width, length);
    gnuplot_cmd(plt, " plot 'initial.dat' matrix with image");

    return plt;
}

// This function draws the initial and final grids of the  simulation
// by reading the data from files and displaying them in separate X11 windows.
// this is designed to be called after the simulation has run for verifying the results.
// After the initial and final grids are displayed, it saves the final grid as a PNG image
// called result/png and waits for the user to press Enter before closing the windows.
//
void drawFirstAndLast(int width, int length) {
    char term_str[200];   // for x11 window setup
    char term_str2[200];  // for png file setup
    char title_str[200];

    float new_width = (float)width;
    float new_length = (float)length;

    float scaleFactor = calcScaleFactor(width, length);

    new_width *= scaleFactor;
    new_length *= scaleFactor;

    //////////////// initial state of grid
    snprintf(term_str, 200, "set terminal x11 size %d, %d", (int)new_width, (int)new_length);

    gnuplot_ctrl* plt;

    plt = gnuplot_init();

    snprintf(title_str, 200, 
            "set title 'Game of Life Simulation: Initial %d x %d Grid'", width, length);

    gnuplot_cmd(plt, title_str);
    gnuplot_cmd(plt, term_str);
    setColorPalette(plt, width, length);
    gnuplot_cmd(plt, " plot 'initial.dat' matrix with image");

    /////////////////////////// final state of grid
    gnuplot_ctrl* plt2;

    plt2 = gnuplot_init();
    snprintf(title_str, 200, 
            "set title 'Game of Life Simulation: Final %d x %d Grid'", width, length);

    gnuplot_cmd(plt2, title_str);
    snprintf(term_str2, 200, "set terminal x11 size %d, %d", (int)new_width, (int)new_length);
    gnuplot_cmd(plt2, term_str2);
    setColorPalette(plt2, width, length);
    gnuplot_cmd(plt2, "plot 'final.dat' matrix with image");

    printf("Press Enter to exit...\n");
    getchar();

    // Save the final grid as a PNG image
    snprintf(term_str2, 200, "set terminal png size %d, %d", (int)new_width, (int)new_length);

    gnuplot_cmd(plt2, term_str2);
    gnuplot_cmd(plt2, "set size 1, 1");
    gnuplot_cmd(plt2, "set output 'result.png'");
    gnuplot_cmd(plt2, "replot 'final.dat' matrix with image");
}

// Draws the final grid of the simulation to X11 window
void drawLast(int width, int length) {
    char term_str[200];   // for x11 window setup
    char term_str2[200];  // for png file setup
    char title_str[200];

    float new_width = (float)width;
    float new_length = (float)length;

    float scaleFactor = calcScaleFactor(width, length);

    new_width *= scaleFactor;
    new_length *= scaleFactor;

    gnuplot_ctrl* plt2;

    plt2 = gnuplot_init();
     snprintf(title_str, 200, 
            "set title 'Game of Life Simulation: Final %d x %d Grid'", width, length);

    gnuplot_cmd(plt2, title_str);
    snprintf(term_str2, 200, "set terminal x11 size %d, %d", (int)new_width, (int)new_length);
    gnuplot_cmd(plt2, term_str2);
    setColorPalette(plt2, width, length);
    gnuplot_cmd(plt2, "plot 'final.dat' matrix with image");

    printf("Press Enter to exit...\n");
    getchar();

    // Save the final grid as a PNG image
    snprintf(term_str2, 200, "set terminal png size %d, %d", (int)new_width, (int)new_length);

    gnuplot_cmd(plt2, term_str2);
    gnuplot_cmd(plt2, "set size 1, 1");
    gnuplot_cmd(plt2, "set output 'result.png'");
    gnuplot_cmd(plt2, "replot 'final.dat' matrix with image");
   
}

//////////////////////////////////////////////// For movie images
//
// This function draws the current iteration of the mushroom simulation
// by reading the data from a file and saving it as a PNG image.
// It is designed to be called after each iteration of the simulation and
// places the file into the movie_images directory.
// Each image can then be used to create an mpeg animation of the simulation.
void drawIteration(int it, char* data_filename, char* png_filename, int width, int length) {

    char term_str[200];   // for x11 window setup
    char term_str2[200];  // for png file setup
    char title_str[200];

    float new_width = (float)width;
    float new_length = (float)length;

    float scaleFactor = calcScaleFactor(width, length);

    new_width *= scaleFactor;
    new_length *= scaleFactor;

    gnuplot_ctrl* plt;

    plt = gnuplot_init();
    setColorPalette(plt, width, length);

    if (it == -1) {
        gnuplot_cmd(plt, "set title 'Initial %d x %d grid'", width, length);
    } else {
        gnuplot_cmd(plt, "set title 'Iteration %d, %d x %d grid'", it, width, length);
    }

    snprintf(term_str2, 200, "set terminal png size %d, %d", (int)new_width, (int)new_length);

    gnuplot_cmd(plt, term_str2);

    // gnuplot_cmd(plt, "set term png");
    gnuplot_cmd(plt, "set output '%s'", png_filename);
    gnuplot_cmd(plt, "plot '%s' matrix with image", data_filename);
}
