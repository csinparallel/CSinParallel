#include "gnuplot_i.h"
#include <stdio.h>

void draw2DHeat(float max, float ratio) {
    gnuplot_ctrl * plt;

    char buf[128];
    sprintf(buf, "set cbrange[0:%g]", max);
    char buf2[128];
    sprintf(buf2, "set size ratio %f", ratio);
        
    plt = gnuplot_init();
    gnuplot_cmd(plt,"set terminal x11 title 'initial temps'");
    gnuplot_cmd(plt, "set title 'Initial plate temps'");
    gnuplot_cmd(plt, buf);
    gnuplot_cmd(plt, "set xrange[0:*]");
    gnuplot_cmd(plt, "set yrange[0:*]");
    gnuplot_cmd(plt, buf2);
    gnuplot_cmd(plt, " plot 'initial.dat' matrix with image");

    gnuplot_ctrl * plt2;
        
    plt2 = gnuplot_init();

    gnuplot_cmd(plt2,"set terminal x11 title 'final temps'");
    gnuplot_cmd(plt2, "set title 'Final plate temps'");
    gnuplot_cmd(plt2, "set xrange[0:*]");
    gnuplot_cmd(plt2, "set yrange[0:*]");
    gnuplot_cmd(plt2, buf);
    gnuplot_cmd(plt2, buf2);
    gnuplot_cmd(plt2, " plot 'final.dat' matrix with image");
}