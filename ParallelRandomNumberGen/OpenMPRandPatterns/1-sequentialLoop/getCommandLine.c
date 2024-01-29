
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include "getCommandLine.h"

void getArguments(int argc, char *argv[],
                  int * N, int * useConstantSeed)
{
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "n:hc")) != -1) {

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

void Usage(char *program) {
  fprintf(stderr, "This program demonstrates use of a loop to create a stream of random numbers in a sequential fashion.");
  fprintf(stderr, "\nUsage: %s [-h] [-n numReps][-c]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -n indicates the number of repetitions of the loop (default is 8).\n");
  fprintf(stderr, "   -c indicates that a fixed seed will be used, resulting in the same stream of numbers each time this is run.\n");
}