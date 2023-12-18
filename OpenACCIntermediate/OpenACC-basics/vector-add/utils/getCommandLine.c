
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include "getCommandLine.h"

void getArguments(int argc, char *argv[], int * n, int *numThreads)
{
 
  char *nvalue;      // number of elements
  char *numThreads_value; 
  
  int c;        // result from getopt calls

  while ((c = getopt (argc, argv, "n:t:")) != -1) {

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
 
      case 't':
        if (isNumber(optarg)) {
          numThreads_value = optarg;
          *numThreads = atoi(numThreads_value);
        } else {
          exitWithError(c, argv);
        } 
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
  fprintf(stderr, "Usage: %s [-n numElements] [-t numThreads]\n", program);
}
