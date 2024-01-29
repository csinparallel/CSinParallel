// #include <iostream>
// #include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
// #include <cstring.h>
#include <ctype.h>
#include "getCommandLine.h"


void getArguments(int argc, char *argv[], int * nx, int * ny, 
                  int * steps, int * display, int * verbose)
{
  // arguments expected that have default values in the code: 
  // m = 10; n = 10; iterations = 15; thresh = 0.50;
  // m, n i, p
  //
  //char *tvalue;  // number of threads
  char *xvalue;       // number of rows
  char *yvalue;      // number of columns
  char *steps_value; // number of iterations
  
  int c;        // result from getopt calls

  // flags indicating that command line flag was encountered
  //
  //int tflag = 0; // number of threads
  // int xflag = 0;  // number of rows
  // int yflag = 0;  // number of columns
  // int steps_flag = 0; // number of bins
  
  // display gnplot results immediately 
  int display_flag = 0; 
  // for verbose printing output only to display
  int verbose_flag = 0;


  while ((c = getopt (argc, argv, "x:y:s:dhv")) != -1) {

    switch (c)
      {
    
      case 'x':
        if (isNumber(optarg)) {
          xvalue = optarg;
          *nx = atoi(xvalue);
        } else {
          exitWithError(c, argv);
        }
        break;
    
      case 'y':
        if (isNumber(optarg)) {
          yvalue = optarg;
          *ny = atoi(yvalue);
        } else {
          exitWithError(c, argv);
        }
        break;

      case 's':
        if (isNumber(optarg)) {
          steps_value = optarg;
          *steps = atoi(steps_value);
        } else {
          exitWithError(c, argv);
        } 
        break;

      case 'd':
        display_flag = 1;
        *display = display_flag;
        break;
      
      case 'v':
        verbose_flag = 1;
        *verbose = verbose_flag;
        break;

      case 'h':
        Usage(argv[0]);
        exit(0);
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (
            (optopt == 'x') ||
            (optopt == 'y') ||
            (optopt == 's') ||
            (optopt == 'd') 
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
    //std::string Str = std::string(s);  // a bit of C++ string use
    //if (Str.length() == 0) return 0;

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
  fprintf(stderr, "Usage: mpirun -np num_procs %s [-x rows] [-y cols] [-s steps] [-d]\n", program);
}
