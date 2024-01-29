#include "gnuplot_i.h"

// Author: Libby Shoop
//


// Pause for user input before closing the plot displays
void wait_to_continue() {

  printf("Pausing to display plots. Press Enter to finish.");
  fflush(stdout);
  getchar();

}

// draw curves for average percent burned and 
// average number of iterations before the fire burned out
// using structs and funcions from gnplot_i.c
//
void drawSimGraphs(int forest_size, int n_probs, int n_trials,
		   double * prob_spread, double * percent_burned,
		   double * avg_iterations) {
// first percent_burned
      gnuplot_ctrl * plt;

      plt = gnuplot_init();
      gnuplot_cmd(plt,"set terminal x11 font 'Verdana,12' title 'Fire Simulation'");
      gnuplot_cmd(plt, "set title '%dx%d Trees, %d Probabilities, %d Trials'",
		  forest_size, forest_size, n_probs, n_trials);
      gnuplot_cmd(plt, "set nokey");
      gnuplot_setstyle(plt, "linespoints");
      gnuplot_set_xlabel(plt, "Probability of Spread");
      gnuplot_set_ylabel(plt, "Averge Percent Burned");

      gnuplot_plot_xy(plt, prob_spread, percent_burned, n_probs, "");

      // next iterations
       gnuplot_ctrl * plt2;
      plt2 = gnuplot_init();
      gnuplot_cmd(plt2,"set terminal x11 font 'Verdana,12' title 'Fire Simulation'");
      gnuplot_cmd(plt2, "set title '%dx%d Trees, %d Probabilities, %d Trials'",
		  forest_size, forest_size, n_probs, n_trials);
      gnuplot_cmd(plt2, "set nokey");
      gnuplot_setstyle(plt2, "linespoints");
      gnuplot_set_xlabel(plt2, "Probability of Spread");
      gnuplot_set_ylabel(plt2, "Averge Iterations");

      gnuplot_plot_xy(plt2, prob_spread, avg_iterations, n_probs, "");//, line_label);

      wait_to_continue();
      gnuplot_close(plt);
      gnuplot_close(plt2);

}

// use gnuplot to display a map of burned, unburned trees in one forest
//
void display_forest(int forest_size, int ** forest, double percent_burned) {
  // using structs and funcions from gnplot_i.c
    gnuplot_ctrl * plt;
        
    plt = gnuplot_init();
    gnuplot_cmd(plt,"set terminal x11 font 'Verdana,12' title 'One Fire'");
    gnuplot_cmd(plt, "set title '%dx%d trees, %f percent burned'",
		forest_size, forest_size, percent_burned);
    // hard-coding danger: 0 is unburnt, 3 is burnt
    gnuplot_cmd(plt, "set palette defined ( 0 '#006600', 3 'white')");
    gnuplot_cmd(plt, "set autoscale xfix");
    gnuplot_cmd(plt, "set autoscale yfix");
    gnuplot_cmd(plt, "set size square");
    gnuplot_cmd(plt, "set cblabel 'unburnt - burnt'");
    gnuplot_cmd(plt, "unset cbtics");
    // hard-coded output file name
    gnuplot_cmd(plt, "plot 'tmpout.dat' matrix with image");
    // wait for cntrl-c
    //    pause();
    wait_to_continue();  // fireSimPlot.c
    gnuplot_close(plt);
}
