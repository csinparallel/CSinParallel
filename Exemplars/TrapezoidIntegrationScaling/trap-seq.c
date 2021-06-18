#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <omp.h>    // only for using the omp_get_wtime() function
#include "seq_time.h"

/* Demo program : computes trapezoidal approximation of an integral*/

const double pi = 3.141592653589793238462643383079;

double f(double x);       // declaration of th function f, defined below

/*
 *  The integral from 0 to pi of the function sine(x) should be equal to 2.0
 *  This program computes that using the 'trapezoidal rule' by summing rectangles.
 */
int main(int argc, char** argv) {
  /* Variables */
  double a = 0.0, b = pi;  /* limits of integration */;
  unsigned long n = 1048576; /* maximum number of subdivisions default = 2^20 */

  double integral; /* accumulates answer */
  int threadct = 1;  /* number of threads to use */

  /* parse command-line arg for number of rectangles */
  if (argc > 1) {
    n = strtol(argv[1], NULL, 10);
  }

  double h = (b - a) / n; /* width of subdivision */
  // debug
  // printf("h = %15.14lf\n",h);

  integral = (f(a) + f(b))/2.0;  // initialize accumulator variable
  int i;

  // for timimg
  // (we will use the openmp library function for timing because it is easy to use)
  // double start = omp_get_wtime();
  // Try the standard C method as coded up in seq_time.h
  double start = c_get_wtime();

// compute each rectangle, adding area to the accumulator
  for(i = 1; i < n; i++) {
    integral += f(a+i*h);
  }

  integral = integral * h;

  // Measuring the elapsed time
  // double end = omp_get_wtime();
  double end = c_get_wtime();
  // Time calculation (in seconds)
  double elapsed_time = end - start;

  //output for debugging
  // printf("With %ld trapezoids, our esimate of the integral from %lf to %lf is %lf\n", n, a, b, integral);
  // printf("Sequential time: %lf seconds\n", elapsed_time);

  //output for sending to a spreadsheet using bash scripts: just the time
  // followed by a tab
  printf("%lf\t",elapsed_time);
}

/*
 *  Function that simply computes the sine of x.
 */
double f(double x) {
  return sin(x);
}
