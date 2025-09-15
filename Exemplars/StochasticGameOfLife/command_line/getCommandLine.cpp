#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "getCommandLine.hpp"


void getArguments(int argc, char *argv[], int * width, int *length, int * iterations, 
                  int * debug, int *graphics, int *animation, int * movie, 
                  int *centerInit, int *numThreads, int *experiment)
{
  // arguments expected that have default values in the code: 
  //
  char *tvalue;  // number of threads
  char *dimvalue;      // number of rows, columns (square grid)
  char *iters_value; // number of iterations
    
  int c;        // result from getopt calls

  double converted;   // for floating point threshold value

// for threads later
// while ((c = getopt (argc, argv, "t:m:n:i:p:dv")) != -1) { 
// m for dimension, t for threads, i for iterations, p for probThreshold, 
// d for debug, v for verbose, a for animation, h for help
// c for centerInit
  while ((c = getopt (argc, argv, "w:l:i:t:dgahcme")) != -1) {

    switch (c)
      {
      case 't':
        if (isNumber(optarg)) {
          tvalue = optarg;
          *numThreads = atoi(tvalue);
        } else {
          exitWithError(c, argv);
        }
       break;

      case 'w':
        if (isNumber(optarg)) {
          dimvalue = optarg;
          *width= atoi(dimvalue);
        } else {
          exitWithError(c, argv);
        }
        break;
      
      case 'l':
        if (isNumber(optarg)) {
          dimvalue = optarg;
          *length = atoi(dimvalue);
        } else {
          exitWithError(c, argv);
        }
        break;
    
      case 'i':
        if (isNumber(optarg)) {
          iters_value = optarg;
          *iterations = atoi(iters_value);
        } else {
          exitWithError(c, argv);
        } 
        break;

      // case 'p':
      //   probThreshold_flag = 1;
      //   probThreshold_value = optarg;
      //   converted = strtod(probThreshold_value, NULL);
      //   if (converted != 0 ) {
      //     *thresh = converted;
      //   } else {
      //     exitWithError(c, argv);
      //   } 
      //   break;

      case 'd':
        *debug = 1;
        break;

      case 'g':
        *graphics = 1;
        break;
      
      case 'a':
        *animation = 1;
        break;
      
      case 'c':
        *centerInit = 1;
        break;
      
      case 'm':
        *movie = 1;
        break;

      case 'e':
        *experiment = 1;
        break;

      case 'h':
        Usage(argv[0]);
        exit(EXIT_SUCCESS);
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (
            (optopt == 'w') ||
            (optopt == 'l') ||
            (optopt == 't') ||
            (optopt == 'i') ||
            (optopt == 'd') ||
            (optopt == 'a') ||
            (optopt == 'm') ||
            (optopt == 'e') ||
            (optopt == 'c') ||
            (optopt == 'g') 
           ) 
        {
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        } else if (isprint (optopt)) {
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        } else {
          fprintf (stderr,
                   "Unknown non-printable option character `\\x%x'.\n",
                   optopt);
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        }
        break;
      
      }
  }
}

int isNumber(char s[])
{
    for (int i = 0; s[i]!= '\0'; i++)
    {
        if (isdigit(s[i]) == 0)
              return 0;
    }
    
    return 1;
}

void exitWithError(char cmdFlag, char ** argv) {
  fprintf(stderr, "Option -%c needs a number value\n", cmdFlag);
  Usage(argv[0]);
  exit(EXIT_FAILURE);
}

void Usage(char *program) {

  fprintf(stderr, "Usage: %s [-m dim] [-i iterations] [-t numThreads] [-d] [-v] [-a]\n", program);
  fprintf(stderr, "  -w dim         : width of the grid (default: 2048)\n");
  fprintf(stderr, "  -l dim         : vertical length of the grid (default: 2048)\n");
  fprintf(stderr, "  -i iterations  : number of iterations (default: 1024)\n");
  fprintf(stderr, "  -t numThreads  : number of threads to use (default: 1)\n");
  fprintf(stderr, "  -d             : debug mode (default: off)\n");
  fprintf(stderr, "  -g             : graphics mode for depicting grids (default: off)\n");
  fprintf(stderr, "  -a             : enable animation (default: off)\n");
  fprintf(stderr, "  -m             : create movie images (default: off)\n");
  fprintf(stderr, "  -c             : center initialization of glider pattern (default: off)\n");
  fprintf(stderr, "  -e             : experiment mode (default: off)\n");
  fprintf(stderr, "  -h             : display this help message\n");

}
