
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>  // C++ string comparison
#include "getCommandLine.h"

#define LEAPFROG 0
#define BLOCKSPLIT 1

// Function to gather command line arguments for 1D loops and arrays
void getArguments(int argc, char *argv[],
                  int * numThreads, int * N, int * useConstantSeed, int * doleOut)
{
  
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "n:t:d:hc")) != -1) {

// getopt implicitly sets a value to a char * (string) called optarg
// to what the user typed after -n
    switch (c)
      {
      // character string entered after the -N needs to be a number
      case 'n':
        if (isNumber(optarg)) {
          *N = atoi(optarg);
        } else {
          exitWithError(c, argv); 
        } 
        break;

      // character string entered after the -t needs to be a number
      case 't':
        if (isNumber(optarg)) {
          *numThreads = atoi(optarg);
        } else {
          exitWithError(c, argv); 
        } 
        break;
      
      case 'd':
        if (strcmp(optarg, "block") == 0) {
          *doleOut = BLOCKSPLIT;
        } else if (strcmp(optarg, "leapfrog") == 0) {
          *doleOut = LEAPFROG;
        } else {
          invalidChoice(c, argv);
        }
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
        if (
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

// Called when isNumber() fails
void exitWithError(char cmdFlag, char ** argv) {
  fprintf(stderr, "Option -%c needs a number value\n", cmdFlag);
  Usage(argv[0]);
  exit(EXIT_FAILURE);
}

void invalidChoice(char cmdFlag, char ** argv) {
  fprintf(stderr, "unrecognized value for Option -%c needs 'block' or 'leapfrog'\n", cmdFlag);
  Usage(argv[0]);
  exit(EXIT_FAILURE);
}

void Usage(char *program) {
  fprintf(stderr, "This program demonstrates use of a loop to create a stream of random numbers in parallel.\n");
  fprintf(stderr, "Usage: %s [-h] [-t numThreads] [-n numReps][-c] [-d block|leapfrog]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -t indicates number of threads to use.\n");
  fprintf(stderr, "   -n indicates the number of repetitions of the loop (default is 8).\n");
  fprintf(stderr, "   -c indicates that a fixed seed will be used, resulting in the same stream of numbers each time this is run.\n");
  fprintf(stderr, "   -d indicates whether the trng generator will dole out numbers in blocks or in leapfrog fashion. default is leapfrog.\n");
}

// Function to gather command line arguments for 2D loops and arrays
void getArguments(int argc, char *argv[],
                  int * numThreads, int * w, int * l, int * useConstantSeed, int * doleOut)
{
  
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "w:l:t:d:hc")) != -1) {

// getopt implicitly sets a value to a char * (string) called optarg
// to what the user typed after -n
    switch (c)
      {
      // character string entered after the -w needs to be a number
      case 'w':
        if (isNumber(optarg)) {
          *w = atoi(optarg);
        } else {
          exitWithError(c, argv); 
        } 
        break;
      // character string entered after the -l needs to be a number
      case 'l':
        if (isNumber(optarg)) {
          *l = atoi(optarg);
        } else {
          exitWithError(c, argv); 
        } 
        break;

      // character string entered after the -t needs to be a number
      case 't':
        if (isNumber(optarg)) {
          *numThreads = atoi(optarg);
        } else {
          exitWithError(c, argv); 
        } 
        break;
      
      case 'd':
        if (strcmp(optarg, "block") == 0) {
          *doleOut = BLOCKSPLIT;
        } else if (strcmp(optarg, "leapfrog") == 0) {
          *doleOut = LEAPFROG;
        } else {
          invalidChoice(c, argv);
        }
        break;
      // If the -h is encountered, then we provide usage
      case 'h':
        Usage2D(argv[0]);
        exit(0);  
        break;

      // If the -c is encountered, then we change the constant seed flag
      case 'c':
        *useConstantSeed = 1;
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage2D(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (
            (optopt == 'v') ||
            (optopt == 't')
           ) 
        {
          Usage2D(argv[0]);
          exit(EXIT_FAILURE);
        } else if (isprint (optopt)) {
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          Usage2D(argv[0]);
          exit(EXIT_FAILURE);
        } else {
          fprintf (stderr,
                   "Unknown non-printable option character `\\x%x'.\n",
                   optopt);
          Usage2D(argv[0]);
          exit(EXIT_FAILURE);
        }
        break;
      
      }
  }
}

// // Check a string as a number containing all digits
// int isNumber(char s[])
// {
//     for (int i = 0; s[i]!= '\0'; i++)
//     {
//         if (isdigit(s[i]) == 0)
//               return 0;
//     }
    
//     return 1;
// }

// // Called when isNumber() fails
// void exitWithError(char cmdFlag, char ** argv) {
//   fprintf(stderr, "Option -%c needs a number value\n", cmdFlag);
//   Usage(argv[0]);
//   exit(EXIT_FAILURE);
// }

// void invalidChoice(char cmdFlag, char ** argv) {
//   fprintf(stderr, "unrecognized value for Option -%c needs 'block' or 'leapfrog'\n", cmdFlag);
//   Usage(argv[0]);
//   exit(EXIT_FAILURE);
// }

void Usage2D(char *program) {
  fprintf(stderr, "This program demonstrates use of a loop to create a stream of random numbers in parallel.\n");
  fprintf(stderr, "Usage: %s [-h] [-t numThreads] [-n numReps][-c] [-d block|leapfrog]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -t indicates number of threads to use.\n");
  fprintf(stderr, "   -w dim         : width of the grid (default: 8)\n");
  fprintf(stderr, "   -l dim         : vertical length of the grid (default: 8)\n");
  fprintf(stderr, "   -c indicates that a fixed seed will be used, resulting in the same stream of numbers each time this is run.\n");
  fprintf(stderr, "   -d indicates whether the trng generator will dole out numbers in blocks or in leapfrog fashion. default is leapfrog.\n");
}
