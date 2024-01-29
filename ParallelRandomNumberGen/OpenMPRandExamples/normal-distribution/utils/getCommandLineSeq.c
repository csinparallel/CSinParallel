// Command line arguments for the histogram example
//

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include "getCommandLineSeq.h"

// Add this if you want to experiment with a distribution of doubles. Also
// make sure that you change the function in utils/getCommandlineSeq.h
// void getDoubleArguments(int argc, char *argv[], int * N, \
//                   int * numBins, double * min, double * max, 
//                   int * print, int * useConstantSeed);
//
// Other things need to change from the int version for error checking also.
// It's easiest to eliminate the check for a number and use strtod() to
// convert the string to a double.

void getArguments(int argc, char *argv[], int * N, int * numBins, int * min, int * max, int * print, int * useConstantSeed)
{

  int c;        // result from getopt calls
  
  while ((c = getopt (argc, argv, "pn:i:a:b:ch")) != -1) {
    switch (c)
      {
      
      case 'n':
        if (isNumber(optarg)) {
          *N = atoi(optarg);
        } else {
          exitWithError(c, argv); 
        } 
        break;

      case 'i':
        if (isNumber(optarg)) {
          *numBins = atoi(optarg);
        } else {
          exitWithError(c, argv); 
        } 
        break;

      case 'a':
        // *min = strtod(optarg, NULL);
        *min = atoi(optarg);
        break;

      case 'b':
        // *max = strtod(optarg, NULL);
        *max = atoi(optarg);
        break;

      case 'p':
        *print = 1;
        break;
      
      // If the -h is encountered, then we provide usage
      case 'h':
        Usage(argv[0]);
        exit(0);  
        break;

      // If the -c is encountered, then we change the constant seed flag
      case 'c':
        *useConstantSeed = 1;
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (isprint (optopt)) {
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

void Usage(char *program) {
  fprintf(stderr, "Usage: %s [-h] [-v] [-n numValues] [-i numBins] [-a min] [-b max] [-p] [-c] [-t numThreads]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -n indicates the number of values to generate.\n");
  fprintf(stderr, "   -a indicates the  minimum in the range of values.\n");
  fprintf(stderr, "   -b indicates the maximum in the range of values.\n");
  fprintf(stderr, "   -i indicates the number of bins to place the values in.\n");
  fprintf(stderr, "   -p print number of values in each bin.\n");
  fprintf(stderr, "   -c indicates that a fixed seed will be used, resulting in the same stream of numbers each time this is run.\n");
}

// Check a string as a number containing all digits
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
