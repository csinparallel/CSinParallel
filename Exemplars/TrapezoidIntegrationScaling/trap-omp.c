#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Demo program for OpenMP: computes trapezoidal approximation to an integral*/

const double pi = 3.141592653589793238462643383079;

double f(double x);   // declaration of th function f, defined below

/*
 *  The integral from 0 to pi of the function sine(x) should be equal to 2.0
 *  This program computes that using the 'trapezoidal rule' by summing rectangles.
 */
int main(int argc, char** argv) {
  /* Variables */
  double a = 0.0, b = pi;  /* limits of integration */;
  unsigned long n = 1048576; /* number of subdivisions default = 2^20 */

  double integral; /* accumulates answer */
  int threadct = 1;  /* number of threads to use */

  /* parse command-line arg for number of rectangles */
  if (argc > 1) {
    n = strtol(argv[1], NULL, 10);  // needs to be positive, though we have no check here for that
  }

  double h = (b - a) / n; /* width of subdivision */

  /* parse command-line arg for number of threads */
  if (argc > 2) {
    threadct = atoi(argv[2]);
  }


#ifdef _OPENMP
  // printf("OMP defined, threadct = %d\n", threadct);
  omp_set_num_threads(threadct);
#else
  printf("OMP not defined\n");
#endif

  integral = (f(a) + f(b))/2.0;
  int i;

  // for timimg
  double start = omp_get_wtime();

// compute each rectangle, adding area to the accumulator
///////////////// 
// We've added the proper additions to the openmp pragma here for correct output.
// Study the pragma line.
////////////////
#pragma omp parallel for default(none) \
private(i) shared(n, a, h) reduction(+:integral)
  for(i = 1; i < n; i++) {
    integral += f(a+i*h);
  }

  integral = integral * h;

  // Measuring the elapsed time
  double end = omp_get_wtime();
  // Time calculation (in seconds)
  double elapsed_time = end - start;

  //debugging output
  // printf("With %ld trapezoids, our esimate of the integral from %lf to %lf is %lf\n", n, a, b, integral);
  // printf("Parallel time: %lf seconds\n", elapsed_time);

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
