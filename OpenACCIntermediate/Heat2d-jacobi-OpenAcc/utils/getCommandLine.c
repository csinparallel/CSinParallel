// #include <iostream>
// #include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
// #include <cstring.h>
#include <ctype.h>
#include "getCommandLine.h"


void getArguments(int argc, char *argv[], int * n, int * m, int * iter, 
                  int * verbose, int *numThreads)
{
 
  //char *tvalue;  // number of threads
  char *nvalue;      // number of rows
  char *mvalue;      // number of columns
  char *iter_value;        // number of iterations
  char *numThreads_value; 
  
  int c;        // result from getopt calls

  // for verbose printing output 
  int verbose_flag = 0;


  while ((c = getopt (argc, argv, "n:m:i:t:v")) != -1) {

    switch (c)
      {
    
      case 'n':
        if (isNumber(optarg)) {
          nvalue = optarg;
          *n = atoi(nvalue);
        } else {
          exitWithError(c, argv);
        }
        break;
    
      case 'm':
        if (isNumber(optarg)) {
          mvalue = optarg;
          *m = atoi(mvalue);
        } else {
          exitWithError(c, argv);
        }
        break;

      case 'i':
        if (isNumber(optarg)) {
          iter_value = optarg;
          *iter = atoi(iter_value);
        } else {
          exitWithError(c, argv);
        } 
        break;
      
      case 't':
        if (isNumber(optarg)) {
          numThreads_value = optarg;
          *numThreads = atoi(numThreads_value);
        } else {
          exitWithError(c, argv);
        } 
        break;

      // case 'd':
      //   display_flag = 1;
      //   *display = display_flag;
      //   break;

      case 'v':
        verbose_flag = 1;
        *verbose = verbose_flag;
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (
            (optopt == 'n') ||
            (optopt == 'm') ||
            (optopt == 'i') ||
            (optopt == 'v') ||
            (optopt == 't')
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
    // fprintf(stderr, "Usage: %s [-t numThreads] [-m rows] [-n cols] [-i iterations] [-p threshold] [-d]\n", program);
  fprintf(stderr, "Usage: %s [-n rows] [-m cols] [-i iterations] [-v] [-t numThreads]\n", program);
}
