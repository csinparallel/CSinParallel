
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include "getCommandLine.h"

// Arguments for a roulette wheel simulation using 
// threads and the trng library
//
// -n [spins] highest number of spins to try
// -c sets a constant seed flag for reproducing 
//     the same stream of random numbers
// -t [numThreads] sets the number of threads to use
// -h is to generate a help message and exit
//
void getArguments(int argc, char *argv[],
                  int * N, 
                  int * numThreads, 
                  int * useConstantSeed)
{
  
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "n:t:hc")) != -1) {

// getopt implicitly sets a value to a char * (string) called optarg
// to what the user typed after -n or -t
    switch (c)
      {
      // character string entered after the -n needs to be a number
      case 'n':
        if (isNumber(optarg)) {
          *N = atoi(optarg);
        } else {
          exitWithError(c, argv, 1); 
        } 
        break;
      
      // character string entered after the -t needs to be a number
      case 't':
        if (isNumber(optarg)) {
          *numThreads = atoi(optarg);
        } else {
          exitWithError(c, argv, 1); 
        } 
        break;

      // If the -h is encountered, then we provide usage
      case 'h':
        Usage(argv[0], 1);
        exit(0);  
        break;

      // If the -c is encountered, then we change the constant seed flag
      case 'c':
        *useConstantSeed = 1;
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0], 1);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (isprint (optopt)) {
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          Usage(argv[0], 1);
          exit(EXIT_FAILURE);
        } else {
          fprintf (stderr,
                   "Unknown non-printable option character `\\x%x'.\n",
                   optopt);
          Usage(argv[0], 1);
          exit(EXIT_FAILURE);
        }
        break;
      
      }
  }
}

// Arguments for a sequential roulette wheel simulation using 
//  the trng library
//
// -n [spins] highest number of spins to try
// -c sets a constant seed flag for reproducing 
//     the same stream of random numbers
// -h is to generate a help message and exit
//
void getArguments(int argc, char *argv[],
                  int * N, 
                  int * useConstantSeed)
{
  
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "n:hc")) != -1) {

// getopt implicitly sets a value to a char * (string) called optarg
// to what the user typed after -n or -t
    switch (c)
      {
      // character string entered after the -n needs to be a number
      case 'n':
        if (isNumber(optarg)) {
          *N = atoi(optarg);
        } else {
          exitWithError(c, argv, 0); 
        } 
        break;
      
      // If the -h is encountered, then we provide usage
      case 'h':
        Usage(argv[0], 0);
        exit(0);  
        break;

      // If the -c is encountered, then we change the constant seed flag
      case 'c':
        *useConstantSeed = 1;
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0], 0);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (isprint (optopt)) {
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          Usage(argv[0], 0);
          exit(EXIT_FAILURE);
        } else {
          fprintf (stderr,
                   "Unknown non-printable option character `\\x%x'.\n",
                   optopt);
          Usage(argv[0], 0);
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
void exitWithError(char cmdFlag, char ** argv, int isPar) {
  fprintf(stderr, "Option -%c needs a number value\n", cmdFlag);
  Usage(argv[0], isPar);
  exit(EXIT_FAILURE);
}

// All useful C/C++ programs with command line 
// arguments produce a Usage string to the screen 
// when there is an issue or when help is requested.

void Usage(char *program, int isPar) {
   if (isPar) {
    fprintf(stderr, "Usage: %s [-h] [-n numSpins] [-c] [-t numThreads]\n", program);
   } else {
    fprintf(stderr, "Usage: %s [-h] [-n numSpins] [-c]\n", program);
   }
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -n indicates the highest number of spins to try (default 1<<26).\n");
  fprintf(stderr, "   -c indicates that a fixed seed will be used, resulting in the same stream of numbers each time this is run.\n");
  if (isPar) {
    fprintf(stderr, 
        "   -t indicates the number of threads to use.");
    fprintf(stderr, "Default is 1 without this flag.\n");
  }
}

