
#include "gnuplot_i.h"

extern "C" void drawFirstAndLast(int width, int length);
extern "C" void drawLast(int width, int length);
extern "C" void drawIteration(int it, char *filename, char *png_filename, int width, int length);
extern "C" gnuplot_ctrl* drawFirst(int width, int length);
extern "C" void redrawIteration(gnuplot_ctrl* plt);
