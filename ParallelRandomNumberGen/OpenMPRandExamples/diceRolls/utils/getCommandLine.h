// functions in getCommandLine.c file

void getArguments(int argc, char *argv[],
                  int * verbose, unsigned int * N, int * numDice,
                  int * numThreads, int * useConstantSeed);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void Usage(char *program);
